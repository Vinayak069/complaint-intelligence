"""Classical baselines that run on a laptop CPU: relief prediction (TF-IDF logistic regression
and LightGBM) and queue routing (TF-IDF logistic regression), plus monthly monitoring.

    python scripts/run_classical.py --data data --out docs
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from complaints import data, models, monitoring


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="docs")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    df = data.load_features(f"{a.data}/cfpb/**/*.parquet")
    train_raw = df[df.split == "train"]
    top = train_raw["company"].value_counts().index[:25].tolist()
    df = models.add_columns(df, top)
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    res = {"data": {"complaints": len(df), "train": len(tr), "val": len(va), "test": len(te),
                    "train_period": [str(tr.date_received.min()), str(tr.date_received.max())],
                    "test_period": [str(te.date_received.min()), str(te.date_received.max())],
                    "relief_rate": {s: float(g.relief.mean()) for s, g in df.groupby("split")}}}
    print(res["data"], flush=True)

    t = time.time()
    lr = models.TextLogReg().fit(tr, tr.relief.values)
    res["relief_logreg"] = models.binary_metrics(te.relief.values, lr.predict(te))
    res["relief_logreg"]["fit_seconds"] = time.time() - t
    print("logreg", res["relief_logreg"], flush=True)

    t = time.time()
    gbm = models.GBMRelief().fit(tr, tr.relief.values, va, va.relief.values)
    p_val, p_test = gbm.predict(va), gbm.predict(te)
    res["relief_lightgbm"] = models.binary_metrics(te.relief.values, p_test)
    res["relief_lightgbm"]["fit_seconds"] = time.time() - t
    res["relief_lightgbm"]["best_iteration"] = int(gbm.model.best_iteration_ or gbm.params["n_estimators"])
    res["relief_lightgbm"]["importance"] = gbm.importance()
    print("lightgbm", {k: v for k, v in res["relief_lightgbm"].items() if k != "importance"}, flush=True)

    mon = monitoring.monthly_report(te, p_test, p_val)
    res["monitoring"] = mon.to_dict(orient="records")
    res["monitoring_input_psi"] = {
        "company_top": monitoring.categorical_psi(va.company_top, te.company_top),
        "issue": monitoring.categorical_psi(va.issue, te.issue),
        "text_len": monitoring.psi(va.text_len.values, te.text_len.values),
        "company_prior_relief_rate": monitoring.psi(va.company_prior_relief_rate.values, te.company_prior_relief_rate.values)}

    t = time.time()
    router = models.QueueRouter().fit(tr.text, tr.queue)
    proba = router.predict_proba(te.text)
    pred = np.array(router.classes)[proba.argmax(1)]
    res["routing_logreg"] = models.multiclass_metrics(te.queue.values, pred, proba, router.classes)
    res["routing_logreg"]["fit_seconds"] = time.time() - t
    res["routing_logreg"]["majority_baseline"] = float((te.queue == tr.queue.mode()[0]).mean())
    print("routing", {k: v for k, v in res["routing_logreg"].items() if k != "per_queue_f1"}, flush=True)

    np.save(out / "results" / "relief_lightgbm_test_scores.npy", p_test.astype(np.float32))
    json.dump(res, open(out / "results" / "classical.json", "w"), indent=2, default=float)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].plot(mon.month, mon.auc, marker="o")
    ax[0].set_title("Relief model AUC by month (test)")
    ax[0].tick_params(axis="x", rotation=60)
    ax[1].plot(mon.month, mon.base_rate, marker="o", label="actual relief rate")
    ax[1].plot(mon.month, mon.mean_score, marker="o", label="mean predicted")
    ax[1].legend()
    ax[1].set_title("Calibration drift by month")
    ax[1].tick_params(axis="x", rotation=60)
    plt.tight_layout()
    plt.savefig(out / "figures" / "monitoring.png", dpi=140)


if __name__ == "__main__":
    main()
