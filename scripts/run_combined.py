"""Combines the classical models with the GPU outputs (kaggle/gpu_models.py): LightGBM with
sentence embeddings, a stacked relief model, and a router ensemble. The blends are fitted on the
validation period and scored on the test period.

    python scripts/run_combined.py --data data --gpu out/gpu_models --out docs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

from complaints import data, models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--gpu", default="out/gpu_models")
    ap.add_argument("--out", default="docs")
    a = ap.parse_args()
    out = Path(a.out, "results")
    out.mkdir(parents=True, exist_ok=True)

    df = data.load_features(f"{a.data}/cfpb/**/*.parquet")
    top = df[df.split == "train"]["company"].value_counts().index[:25].tolist()
    df = models.add_columns(df, top)
    emb = pd.read_parquet(Path(a.gpu, "embeddings.parquet")).set_index("complaint_id")
    tf = pd.read_parquet(Path(a.gpu, "transformer_preds.parquet")).set_index("complaint_id")
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    E = {s: emb.loc[d.complaint_id].to_numpy(np.float32) for s, d in (("train", tr), ("val", va), ("test", te))}
    res = {}

    p = {}
    for name, use_emb in (("lightgbm", False), ("lightgbm_embeddings", True)):
        g = models.GBMRelief().fit(tr, tr.relief.values, va, va.relief.values,
                                   emb=E["train"] if use_emb else None, val_emb=E["val"] if use_emb else None)
        p[name] = {s: g.predict(d, E[s] if use_emb else None) for s, d in (("val", va), ("test", te))}
        res[name] = models.binary_metrics(te.relief.values, p[name]["test"])
        if use_emb:
            res[name]["importance"] = g.importance()
        print(name, {k: v for k, v in res[name].items() if k != "importance"}, flush=True)
    p["transformer"] = {s: tf.loc[d.complaint_id, "relief_p"].to_numpy() for s, d in (("val", va), ("test", te))}
    res["transformer"] = models.binary_metrics(te.relief.values, p["transformer"]["test"])

    cols = ["lightgbm_embeddings", "transformer"]
    X = {s: np.column_stack([logit(np.clip(p[c][s], 1e-6, 1 - 1e-6)) for c in cols]) for s in ("val", "test")}
    st = LogisticRegression().fit(X["val"], va.relief.values)
    res["stacked"] = models.binary_metrics(te.relief.values, st.predict_proba(X["test"])[:, 1])
    res["stacked"]["inputs"] = cols
    res["stacked"]["weights"] = st.coef_[0].tolist()
    print("stacked", res["stacked"], flush=True)

    router = models.QueueRouter().fit(tr.text, tr.queue)
    classes = router.classes
    qcols = [f"q_{c}" for c in classes]
    lr_p = {s: router.predict_proba(d.text) for s, d in (("val", va), ("test", te))}
    tf_p = {s: tf.loc[d.complaint_id, qcols].to_numpy() for s, d in (("val", va), ("test", te))}
    acc = lambda pr, d: float((np.array(classes)[pr.argmax(1)] == d.queue.values).mean())
    w = max(np.linspace(0, 1, 11), key=lambda w: acc(w * tf_p["val"] + (1 - w) * lr_p["val"], va))
    for name, pr in (("logreg", lr_p["test"]), ("transformer", tf_p["test"]),
                     ("ensemble", w * tf_p["test"] + (1 - w) * lr_p["test"])):
        res[f"routing_{name}"] = models.multiclass_metrics(te.queue.values, np.array(classes)[pr.argmax(1)], pr, classes)
    res["routing_ensemble"]["transformer_weight"] = float(w)
    print({k: {m: v[m] for m in ("accuracy", "macro_f1", "top2_accuracy")} for k, v in res.items() if k.startswith("routing")})
    json.dump(res, open(out / "combined.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
