"""GPU job: evaluates the LLM triage agent on a random sample of test complaints, with and
without its tools. Needs the outputs of kaggle/gpu_models.py (branch kaggle-out/gpu_models).

Outputs in --out: agent_eval.json (metrics) and agent_outputs.jsonl (every prompt's answer).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from complaints import data, models
from complaints.agent import triage

LLM = "Qwen/Qwen2.5-3B-Instruct"


def fetch_gpu_outputs(dest: Path) -> Path:
    url = subprocess.run(["git", "remote", "get-url", "origin"], capture_output=True, text=True, check=True).stdout.strip()
    subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", "kaggle-out/gpu_models", url, str(dest)], check=True)
    return dest / "out"


@torch.no_grad()
def generate(tok, llm, convs: list[list[dict]], bs: int = 16, max_new: int = 320) -> list[str]:
    prompts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in convs]
    order = np.argsort([len(p) for p in prompts])
    out = [None] * len(prompts)
    for i in range(0, len(prompts), bs):
        idx = order[i:i + bs]
        b = tok([prompts[j] for j in idx], return_tensors="pt", padding=True).to(llm.device)
        g = llm.generate(**b, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        for j, seq in zip(idx, g[:, b["input_ids"].shape[1]:]):
            out[j] = tok.decode(seq, skip_special_tokens=True)
        print(f"generated {min(i + bs, len(prompts))}/{len(prompts)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="out")
    ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gpu = fetch_gpu_outputs(Path("/kaggle/tmp/gpu_models"))

    df = data.load_features(f"{a.data}/cfpb/**/*.parquet")
    top = df[df.split == "train"]["company"].value_counts().index[:25].tolist()
    df = models.add_columns(df, top).reset_index(drop=True)
    preds = pd.read_parquet(gpu / "transformer_preds.parquet").query("split == 'test'")
    emb = pd.read_parquet(gpu / "embeddings.parquet").set_index("complaint_id")
    queues = sorted(c[2:] for c in preds.columns if c.startswith("q_"))

    tr = df[df.split == "train"]
    te = df[df.split == "test"].merge(preds, on="complaint_id")
    sample = te.sample(a.n, random_state=0).reset_index(drop=True)
    nn_idx = triage.nearest(emb.loc[sample.complaint_id].to_numpy(np.float32),
                            emb.loc[tr.complaint_id].to_numpy(np.float32), k=3)
    cases = []
    for i, r in sample.iterrows():
        sims = [{"issue": tr.issue.iat[j], "text": tr.text.iat[j],
                 "response": "relief" if tr.relief.iat[j] else "closed with explanation"} for j in nn_idx[i]]
        cases.append(triage.Case(int(r.complaint_id), r.text, r.sub_product, r.company,
                                 {q: float(r[f"q_{q}"]) for q in queues}, float(r.relief_p), sims))

    tok = AutoTokenizer.from_pretrained(LLM, padding_side="left")
    llm = AutoModelForCausalLM.from_pretrained(LLM, torch_dtype=torch.float16, device_map="cuda:0").eval()

    res = {"llm": LLM, "n": a.n, "router_alone_accuracy": float(
        np.mean([triage.tool_route(c)[0][0] == q for c, q in zip(cases, sample.queue)])),
        "sample_relief_rate": float(sample.relief.mean())}
    rows_out = []
    for with_tools in (False, True):
        name = "with_tools" if with_tools else "no_tools"
        t = time.time()
        raws = generate(tok, llm, [triage.messages(c, queues, with_tools) for c in cases])
        rows = [{"complaint_id": c.complaint_id, "true_queue": q, "relief": int(rel),
                 "provided": triage.provided_policies(c, with_tools), "raw": raw}
                for c, q, rel, raw in zip(cases, sample.queue, sample.relief, raws)]
        res[name] = triage.score(rows, queues)
        res[name]["seconds"] = time.time() - t
        print(name, res[name], flush=True)
        rows_out += [{**r, "provided": sorted(r["provided"]), "condition": name} for r in rows]

    json.dump(res, open(out / "agent_eval.json", "w"), indent=2, default=float)
    with open(out / "agent_outputs.jsonl", "w") as f:
        for r in rows_out:
            f.write(json.dumps(r) + "\n")
    Path(out, "summary.json").write_text(json.dumps({"done": True}))


if __name__ == "__main__":
    main()
