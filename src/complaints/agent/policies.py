"""The policy notes the agent can cite, parsed from policies.md."""
from __future__ import annotations

import re
from dataclasses import dataclass
from importlib.resources import files


@dataclass(frozen=True)
class Policy:
    id: str
    title: str
    queues: tuple[str, ...]
    text: str


def load_policies() -> dict[str, Policy]:
    md = files("complaints").joinpath("agent/policies.md").read_text()
    out = {}
    for block in md.split("\n## ")[1:]:
        head, body = block.split("\n", 1)
        pid, title = head.split(" ", 1)
        m = re.match(r"Queues: (.*)\n", body)
        queues = tuple(q.strip() for q in m.group(1).split(",")) if m else ("all",)
        out[pid] = Policy(pid, title.strip(), queues, body[m.end():].strip() if m else body.strip())
    return out


def policies_for(queue: str) -> list[Policy]:
    return [p for p in load_policies().values() if queue in p.queues or "all" in p.queues]
