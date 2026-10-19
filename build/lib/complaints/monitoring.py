"""Model monitoring: population stability (PSI) of the score and of inputs, and performance by month."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index of `actual` against `expected`, on the expected deciles.
    Rule of thumb: < 0.1 stable, 0.1-0.25 watch, > 0.25 investigate."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def categorical_psi(expected: pd.Series, actual: pd.Series) -> float:
    cats = sorted(set(expected) | set(actual))
    e = expected.value_counts(normalize=True).reindex(cats, fill_value=0).values.clip(1e-6)
    a = actual.value_counts(normalize=True).reindex(cats, fill_value=0).values.clip(1e-6)
    return float(np.sum((a - e) * np.log(a / e)))


def monthly_report(df: pd.DataFrame, score: np.ndarray, ref_score: np.ndarray, label: str = "relief") -> pd.DataFrame:
    """One row per month: volume, base rate, mean score, AUC and score PSI against the reference (validation) period."""
    d = df.assign(score=score)
    rows = []
    for m, g in d.groupby("month"):
        rows.append({"month": str(m)[:7], "n": len(g), "base_rate": g[label].mean(), "mean_score": g["score"].mean(),
                     "auc": roc_auc_score(g[label], g["score"]) if g[label].nunique() > 1 else np.nan,
                     "score_psi": psi(ref_score, g["score"].values)})
    return pd.DataFrame(rows)
