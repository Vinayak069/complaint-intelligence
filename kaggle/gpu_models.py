"""GPU job (Kaggle, 2x T4): sentence embeddings with retrieval evaluation, then a multi-task
transformer fine-tune for queue routing and relief prediction.

Outputs in --out:
  embeddings.parquet       complaint_id + 64-d PCA of the embeddings (for LightGBM and the agent)
  transformer_preds.parquet val/test predictions: relief probability and queue probabilities
  gpu_models.json          metrics
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from complaints import data, models

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
FT_MODEL = "roberta-base"


def model_text(df: pd.DataFrame) -> list[str]:
    # The routing label is derived from the consumer's chosen issue, so the issue never enters the text.
    return (df.sub_product + " | " + df.company_top + " | " + df.text).tolist()


@torch.no_grad()
def embed(texts: list[str], dev) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
    enc = AutoModel.from_pretrained(EMBED_MODEL, torch_dtype=torch.float16).to(dev).eval()
    if torch.cuda.device_count() > 1:
        enc = nn.DataParallel(enc)
    order = np.argsort([len(t) for t in texts])  # length-sorted batches waste less padding
    out = np.zeros((len(texts), 768), dtype=np.float32)
    for i in range(0, len(texts), 256):
        idx = order[i:i + 256]
        b = tok([texts[j] for j in idx], truncation=True, max_length=384, padding=True, return_tensors="pt").to(dev)
        h = enc(**b).last_hidden_state[:, 0]  # bge uses the CLS vector
        out[idx] = torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy()
    return out


@torch.no_grad()
def knn(query: np.ndarray, index: np.ndarray, k: int, dev) -> tuple[np.ndarray, np.ndarray]:
    ix = torch.from_numpy(index).half().to(dev)
    sims, nbrs = [], []
    for i in range(0, len(query), 2048):
        s = torch.from_numpy(query[i:i + 2048]).half().to(dev) @ ix.T
        v, j = s.topk(k, dim=1)
        sims.append(v.float().cpu().numpy())
        nbrs.append(j.cpu().numpy())
    return np.vstack(sims), np.vstack(nbrs)


def retrieval_eval(tr, te, e_tr, e_te, dev) -> dict:
    sims, nbrs = knn(e_te, e_tr, 20, dev)
    q_tr, q_te = tr.queue.to_numpy(dtype=object), te.queue.to_numpy(dtype=object)
    vote = np.array([pd.Series(q_tr[row]).value_counts().index[0] for row in nbrs])
    relief_knn = tr.relief.to_numpy()[nbrs].mean(1)
    return {"model": EMBED_MODEL, "k": 20,
            "same_queue_at_1": float((q_tr[nbrs[:, 0]] == q_te).mean()),
            "same_queue_precision_at_5": float((q_tr[nbrs[:, :5]] == q_te[:, None]).mean()),
            "knn_routing_accuracy": float((vote == q_te).mean()),
            "knn_relief_auc": float(roc_auc_score(te.relief.values, relief_knn)),
            "mean_top1_similarity": float(sims[:, 0].mean())}


class MultiTask(nn.Module):
    def __init__(self, n_queues: int):
        super().__init__()
        self.enc = AutoModel.from_pretrained(FT_MODEL)
        d = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.queue_head = nn.Linear(d, n_queues)
        self.relief_head = nn.Linear(d, 1)

    def forward(self, input_ids, attention_mask):
        h = self.drop(self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0])
        return self.queue_head(h), self.relief_head(h).squeeze(-1)


def batches(enc, ql, rl, bs, shuffle, rng=None):
    idx = rng.permutation(len(ql)) if shuffle else np.arange(len(ql))
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        yield (torch.from_numpy(enc["input_ids"][j]), torch.from_numpy(enc["attention_mask"][j]),
               torch.from_numpy(ql[j]), torch.from_numpy(rl[j]))


@torch.no_grad()
def predict(net, enc, dev, bs=256):
    net.eval()
    qs, rs = [], []
    n = len(enc["input_ids"])
    for i in range(0, n, bs):
        ids = torch.from_numpy(enc["input_ids"][i:i + bs]).to(dev)
        am = torch.from_numpy(enc["attention_mask"][i:i + bs]).to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            q, r = net(ids, am)
        qs.append(torch.softmax(q.float(), -1).cpu().numpy())
        rs.append(torch.sigmoid(r.float()).cpu().numpy())
    return np.vstack(qs), np.concatenate(rs)


def finetune(tr, va, te, classes, dev, epochs=2, max_len=256, bs=64, lr=3e-5, relief_weight=1.0) -> dict:
    tok = AutoTokenizer.from_pretrained(FT_MODEL)
    enc = {s: tok(model_text(d), truncation=True, max_length=max_len, padding="max_length", return_tensors="np")
           for s, d in (("train", tr), ("val", va), ("test", te))}
    qidx = {c: i for i, c in enumerate(classes)}
    lab = {s: (d.queue.map(qidx).to_numpy(dtype=np.int64), d.relief.to_numpy(dtype=np.float32))
           for s, d in (("train", tr), ("val", va), ("test", te))}
    net = MultiTask(len(classes)).to(dev)
    par = nn.DataParallel(net) if torch.cuda.device_count() > 1 else net
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(tr) + bs - 1) // bs)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.cuda.amp.GradScaler()
    ce, bce = nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(0)
    log, best, best_state = [], -1.0, None
    for ep in range(epochs):
        par.train()
        t, tot, n = time.time(), 0.0, 0
        for ids, am, q, r in batches(enc["train"], *lab["train"], bs, True, rng):
            ids, am, q, r = ids.to(dev), am.to(dev), q.to(dev), r.to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                lq, lr_ = par(ids, am)
                loss = ce(lq.float(), q) + relief_weight * bce(lr_.float(), r)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot, n = tot + loss.item(), n + 1
            if n % 200 == 0:
                print(f"epoch {ep} step {n} loss {tot / n:.4f} {time.time() - t:.0f}s", flush=True)
        qp, rp = predict(par, enc["val"], dev)
        acc = float((qp.argmax(1) == lab["val"][0]).mean())
        auc = float(roc_auc_score(lab["val"][1], rp))
        log.append({"epoch": ep, "train_loss": tot / n, "val_queue_accuracy": acc, "val_relief_auc": auc,
                    "seconds": time.time() - t})
        print(log[-1], flush=True)
        if acc + auc > best:
            best, best_state = acc + auc, {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    preds = {s: predict(par, enc[s], dev) for s in ("val", "test")}
    return {"log": log, "preds": preds, "config": {"model": FT_MODEL, "epochs": epochs, "max_len": max_len,
                                                   "batch_size": bs, "lr": lr, "relief_weight": relief_weight}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="out")
    ap.add_argument("--epochs", type=int, default=2)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    torch.manual_seed(0)

    df = data.load_features(f"{a.data}/cfpb/**/*.parquet")
    top = df[df.split == "train"]["company"].value_counts().index[:25].tolist()
    df = models.add_columns(df, top).reset_index(drop=True)
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    classes = sorted(tr.queue.unique())
    res = {"gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
           "n": {"train": len(tr), "val": len(va), "test": len(te)}}
    print(res, flush=True)

    t = time.time()
    emb = embed(model_text(df), dev)
    res["embedding_seconds"] = time.time() - t
    res["retrieval"] = retrieval_eval(tr, te, emb[tr.index], emb[te.index], dev)
    print(res["retrieval"], flush=True)
    pca = PCA(64, random_state=0).fit(emb[tr.index])
    res["pca_explained_variance"] = float(pca.explained_variance_ratio_.sum())
    e64 = pca.transform(emb).astype(np.float16)
    pd.DataFrame({"complaint_id": df.complaint_id.values,
                  **{f"e{i}": e64[:, i] for i in range(64)}}).to_parquet(out / "embeddings.parquet", index=False)
    del emb
    torch.cuda.empty_cache()

    t = time.time()
    ft = finetune(tr, va, te, classes, dev, epochs=a.epochs)
    res["finetune_seconds"] = time.time() - t
    res["finetune"] = {"config": ft["config"], "log": ft["log"]}
    for s, d in (("val", va), ("test", te)):
        qp, rp = ft["preds"][s]
        pred = np.array(classes)[qp.argmax(1)]
        res["finetune"][f"routing_{s}"] = models.multiclass_metrics(d.queue.values, pred, qp, classes)
        res["finetune"][f"relief_{s}"] = models.binary_metrics(d.relief.values, rp)
    print({k: v for k, v in res["finetune"].items() if k.endswith("test")}, flush=True)
    rows = []
    for s, d in (("val", va), ("test", te)):
        qp, rp = ft["preds"][s]
        f = pd.DataFrame(qp.astype(np.float32), columns=[f"q_{c}" for c in classes])
        f.insert(0, "relief_p", rp.astype(np.float32))
        f.insert(0, "split", s)
        f.insert(0, "complaint_id", d.complaint_id.values)
        rows.append(f)
    pd.concat(rows).to_parquet(out / "transformer_preds.parquet", index=False)
    json.dump(res, open(out / "gpu_models.json", "w"), indent=2, default=float)
    Path(out, "summary.json").write_text(json.dumps({"done": True}))


if __name__ == "__main__":
    main()
