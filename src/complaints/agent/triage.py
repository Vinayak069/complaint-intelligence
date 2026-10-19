"""Complaint triage with an LLM: tools, prompts, output parsing and scoring.

The agent gets a complaint and the results of four tools (queue router, relief model, similar
past cases, policy notes) and must return JSON: queue, priority, cited policies and a first reply
to the customer. The same prompt without the tool results is the ablation. Generation itself
runs on a GPU (kaggle/agent_eval.py); everything here is plain Python and tested on CPU.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np

from complaints.agent.policies import load_policies, policies_for

PRIORITIES = ("high", "medium", "low")

# A first reply must not commit the company to an outcome before the investigation. A heuristic:
# "will/ensure ... refund" within one clause, unless the clause is about reviewing or a request.
_PROMISE = re.compile(r"\b(will|we'll|you'll|ensure|make sure|guarantee\w*)\b[^.;!?]{0,60}?\b(refund|reimburs|revers|"
                      r"waive|credit(?:ed)? back|remove|removal|delete|correct(?:ed|ion)?)\w*", re.I)
_HEDGE = re.compile(r"\b(review|investigat|whether|if |request|eligib|determin|look into|assess)", re.I)


class _Unsafe:
    def search(self, text: str) -> bool:
        return any(not _HEDGE.search(m.group(0)) for m in _PROMISE.finditer(text))


UNSAFE = _Unsafe()

SYSTEM = """You triage US credit-card complaints for a card issuer's servicing team.
Return only a JSON object with these keys:
  "queue": one of {queues}
  "priority": "high", "medium" or "low" (high = likely the customer is owed relief or is being harmed now)
  "cited_policies": list of policy ids from the notes you were given that apply
  "reply": a first reply to the customer, 60-120 words, that acknowledges the specific problem, says what
           happens next and when, and does not promise a refund, reversal or any outcome before the review
No text outside the JSON."""

GUARDED = """You triage US credit-card complaints for a card issuer's servicing team.
Return only a JSON object with these keys:
  "queue": one of the three queues the route() tool lists (pick another of the three only if the complaint
           clearly belongs there)
  "escalate": true only if the complaint mentions identity theft or fraud in progress, a servicemember,
           an older or vulnerable consumer, or legal action; otherwise false
  "cited_policies": ids of the policy notes below that apply
  "reply": a first reply to the customer of 70 to 110 words. Restate their specific problem in one sentence,
           say which team is reviewing it, what they may send to help (statements, receipts, letters), and
           that the company will respond within 15 days. Never promise a refund, credit, reversal, removal or
           any other outcome; say the review will decide.
Example reply: "Thank you for contacting us about the two charges of $120 from the same merchant on your
March statement. Our billing disputes team is reviewing them now. While the review is open you do not need to
pay the disputed amount or any interest on it. If you have a receipt, an order confirmation or messages with
the merchant, please reply with copies, as they help us finish sooner. We will write to you with the outcome
within 15 days, and sooner if we can."
No text outside the JSON."""


@dataclass
class Case:
    complaint_id: int
    text: str
    sub_product: str
    company: str
    queue_probs: dict[str, float] | None = None
    relief_p: float | None = None
    similar: list[dict] | None = None


def tool_route(case: Case, k: int = 3) -> list[tuple[str, float]]:
    return sorted(case.queue_probs.items(), key=lambda kv: -kv[1])[:k]


def tool_policies(queue: str) -> list[dict]:
    return [{"id": p.id, "title": p.title, "text": p.text} for p in policies_for(queue)]


def nearest(query: np.ndarray, index: np.ndarray, k: int = 3) -> np.ndarray:
    q = query / np.linalg.norm(query, axis=1, keepdims=True)
    ix = index / np.linalg.norm(index, axis=1, keepdims=True)
    return np.argsort(-(q @ ix.T), axis=1)[:, :k]


def messages(case: Case, queues: list[str], with_tools: bool, max_chars: int = 3000, guarded: bool = False
             ) -> list[dict]:
    sys = GUARDED if guarded else SYSTEM.format(queues=", ".join(queues))
    user = [f"Product: {case.sub_product}\nCompany: {case.company}\nComplaint:\n{case.text[:max_chars]}"]
    if with_tools:
        route = tool_route(case)
        user.append("Tool route() -> " + ", ".join(f"{q} ({p:.2f})" for q, p in route))
        user.append(f"Tool predict_relief() -> probability the complaint ends in relief: {case.relief_p:.2f}")
        sims = "\n".join(f"- [{s['issue']}; outcome: {s['response']}] {s['text'][:240]}" for s in case.similar)
        user.append("Tool search_similar_cases() ->\n" + sims)
        pol = {p["id"]: p for q, _ in route[:2] for p in tool_policies(q)}
        user.append("Tool get_policy() ->\n" + "\n".join(f"{p['id']} {p['title']}: {p['text']}" for p in pol.values()))
    else:
        user.append("Policy notes available:\n" + "\n".join(f"{p.id} {p.title}" for p in load_policies().values()))
    return [{"role": "system", "content": sys}, {"role": "user", "content": "\n\n".join(user)}]


def provided_policies(case: Case, with_tools: bool) -> set[str]:
    if not with_tools:
        return set(load_policies())
    return {p["id"] for q, _ in tool_route(case)[:2] for p in tool_policies(q)}


def guard(raw: str, case: Case, high: float, medium: float) -> tuple[str, dict]:
    """Guardrails around the LLM in the guarded design: the queue must be one of the router's top
    three (else the router's first choice), priority comes from the relief model's calibrated
    thresholds (set on the validation period), and escalation flags can only raise it."""
    out = parse(raw, required=("queue", "reply")) or {}
    top = [q for q, _ in tool_route(case)]
    flags = {"queue_fallback": out.get("queue") not in top, "parse_failed": not out}
    queue = out.get("queue") if not flags["queue_fallback"] else top[0]
    pri = "high" if case.relief_p >= high else "medium" if case.relief_p >= medium else "low"
    if out.get("escalate") is True and pri != "high":
        pri, flags["escalated"] = "high", True
    final = {"queue": queue, "priority": pri, "cited_policies": out.get("cited_policies", []),
             "reply": out.get("reply", "")}
    return json.dumps(final), flags


def parse(raw: str, required=("queue", "priority", "reply")) -> dict | None:
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        out = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(out, dict) or not set(required) <= out.keys():
        return None
    out["cited_policies"] = [str(p) for p in out.get("cited_policies") or []]
    return out


def score(rows: list[dict], queues: list[str]) -> dict:
    """rows: dicts with true_queue, relief, provided (set of policy ids), raw (model text)."""
    valid_ids = set(load_policies())
    parsed = [parse(r["raw"]) for r in rows]
    ok = [(r, p) for r, p in zip(rows, parsed) if p is not None]
    out = {"n": len(rows), "json_valid": len(ok) / max(len(rows), 1)}
    if not ok:
        return out
    out["queue_valid"] = float(np.mean([p["queue"] in queues for _, p in ok]))
    out["queue_accuracy"] = float(np.mean([p["queue"] == r["true_queue"] for r, p in ok]))
    by_pri = {}
    for pri in PRIORITIES:
        rel = [r["relief"] for r, p in ok if str(p["priority"]).lower() == pri]
        by_pri[pri] = {"share": len(rel) / len(ok), "relief_rate": float(np.mean(rel)) if rel else None}
    out["priority"] = by_pri
    cited = [(r, p["cited_policies"]) for r, p in ok]
    out["cites_any_policy"] = float(np.mean([bool(c) for _, c in cited]))
    out["citations_exist"] = float(np.mean([all(c in valid_ids for c in cs) for _, cs in cited]))
    out["citations_grounded"] = float(np.mean([all(c in r["provided"] for c in cs) for r, cs in cited]))
    out["unsafe_promise_rate"] = float(np.mean([bool(UNSAFE.search(str(p["reply"]))) for _, p in ok]))
    words = [len(str(p["reply"]).split()) for _, p in ok]
    out["reply_words_median"] = float(np.median(words))
    out["reply_in_60_120_words"] = float(np.mean([60 <= w <= 120 for w in words]))
    return out
