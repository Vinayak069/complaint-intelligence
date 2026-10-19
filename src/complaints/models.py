"""Relief prediction (will this complaint end in monetary or non-monetary relief?) and queue
routing (which servicing team should own it?), with classical models.

Relief models see only what is known when the complaint arrives: the narrative, the consumer's
chosen product/issue, the company, state, tags, and the company's relief history before that
month. The outcome itself (company response) is the label and never a feature.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score,
                             roc_auc_score)
from sklearn.preprocessing import OneHotEncoder

from complaints.data import clean_text

CATEGORICAL = ["company_top", "sub_product", "issue", "sub_issue", "state", "tags"]
NUMERIC = ["text_len", "n_amounts", "max_amount", "n_redactions", "company_prior_n", "company_prior_relief_rate",
           "month_of_year"]


def add_columns(df: pd.DataFrame, top_companies: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["company_top"] = np.where(df["company"].isin(top_companies), df["company"], "Other company")
    df["text"] = df["narrative"].map(clean_text)
    df["max_amount"] = np.log1p(df["max_amount"].clip(lower=0))
    df["text_len"] = np.log1p(df["text_len"])
    return df


def tfidf(max_features: int = 200_000) -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.5, max_features=max_features, sublinear_tf=True,
                           strip_accents="unicode", lowercase=True)


def binary_metrics(y: np.ndarray, p: np.ndarray, top_frac: float = 0.2) -> dict:
    """AUC, average precision, Brier score, and the business view: if the team reviews the top
    `top_frac` of complaints by score first, what share of all relief cases does it reach?"""
    order = np.argsort(-p)
    k = int(len(p) * top_frac)
    captured = y[order[:k]].sum() / max(y.sum(), 1)
    return {"auc": float(roc_auc_score(y, p)), "avg_precision": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)), "base_rate": float(y.mean()),
            f"relief_captured_top{int(top_frac * 100)}pct": float(captured),
            f"precision_top{int(top_frac * 100)}pct": float(y[order[:k]].mean()),
            "lift_top20pct": float(y[order[:k]].mean() / y.mean())}


class TextLogReg:
    """TF-IDF on the narrative plus one-hot intake fields, logistic regression."""

    def __init__(self, C: float = 2.0):
        self.vec, self.ohe = tfidf(), OneHotEncoder(handle_unknown="ignore", min_frequency=20)
        self.clf = LogisticRegression(C=C, max_iter=2000, solver="liblinear")

    def _x(self, df, fit=False):
        t = self.vec.fit_transform(df["text"]) if fit else self.vec.transform(df["text"])
        o = self.ohe.fit_transform(df[CATEGORICAL]) if fit else self.ohe.transform(df[CATEGORICAL])
        return sparse.hstack([t, o]).tocsr()

    def fit(self, df, y):
        self.clf.fit(self._x(df, fit=True), y)
        return self

    def predict(self, df):
        return self.clf.predict_proba(self._x(df))[:, 1]


class GBMRelief:
    """LightGBM on intake fields, numeric features, and a 128-dim SVD of the TF-IDF text
    (optionally plus sentence embeddings)."""

    def __init__(self, svd_dim: int = 128, seed: int = 0, params: dict | None = None):
        self.vec, self.svd = tfidf(100_000), TruncatedSVD(svd_dim, random_state=seed)
        self.params = {"n_estimators": 2000, "learning_rate": 0.03, "num_leaves": 63, "min_child_samples": 100,
                       "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.6, "reg_lambda": 1.0,
                       "random_state": seed, "verbose": -1, **(params or {})}

    def _x(self, df, emb=None, fit=False):
        t = self.vec.fit_transform(df["text"]) if fit else self.vec.transform(df["text"])
        z = self.svd.fit_transform(t) if fit else self.svd.transform(t)
        x = pd.DataFrame(z, columns=[f"svd{i}" for i in range(z.shape[1])], index=df.index)
        for c in NUMERIC:
            x[c] = df[c].values
        for c in CATEGORICAL:
            x[c] = pd.Categorical(df[c], categories=self.cats[c]) if not fit else df[c].astype("category")
        if fit:
            self.cats = {c: x[c].cat.categories for c in CATEGORICAL}
        if emb is not None:
            for i in range(emb.shape[1]):
                x[f"emb{i}"] = emb[:, i]
        return x

    def fit(self, df, y, val_df=None, val_y=None, emb=None, val_emb=None):
        x = self._x(df, emb, fit=True)
        self.model = lgb.LGBMClassifier(**self.params)
        kw = {}
        if val_df is not None:
            kw = {"eval_set": [(self._x(val_df, val_emb), val_y)], "eval_metric": "auc",
                  "callbacks": [lgb.early_stopping(100, verbose=False)]}
        self.model.fit(x, y, **kw)
        return self

    def predict(self, df, emb=None):
        return self.model.predict_proba(self._x(df, emb))[:, 1]

    def importance(self, top: int = 15) -> dict:
        imp = self.model.booster_.feature_importance("gain")
        names = self.model.booster_.feature_name()
        svd_total = float(sum(g for n, g in zip(names, imp) if n.startswith("svd")))
        emb_total = float(sum(g for n, g in zip(names, imp) if n.startswith("emb")))
        rest = sorted(((n, float(g)) for n, g in zip(names, imp) if not n.startswith(("svd", "emb"))), key=lambda t: -t[1])
        total = float(imp.sum())
        out = {"text_svd (all)": svd_total / total, "embeddings (all)": emb_total / total}
        out.update({n: g / total for n, g in rest[:top]})
        return out


class QueueRouter:
    """Routes a complaint to a servicing queue from its narrative alone (TF-IDF + logistic regression)."""

    def __init__(self, C: float = 4.0):
        self.vec = tfidf()
        self.clf = LogisticRegression(C=C, max_iter=3000)

    def fit(self, text, labels):
        self.clf.fit(self.vec.fit_transform(text), labels)
        return self

    def predict_proba(self, text):
        return self.clf.predict_proba(self.vec.transform(text))

    def predict(self, text):
        return self.clf.predict(self.vec.transform(text))

    @property
    def classes(self):
        return list(self.clf.classes_)


def multiclass_metrics(y_true, y_pred, proba=None, classes=None) -> dict:
    out = {"accuracy": float(accuracy_score(y_true, y_pred)), "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
           "per_queue_f1": {c: float(f) for c, f in zip(sorted(set(y_true)),
                                                         f1_score(y_true, y_pred, average=None, labels=sorted(set(y_true))))}}
    if proba is not None and classes is not None:
        idx = {c: i for i, c in enumerate(classes)}
        top2 = np.argsort(-proba, axis=1)[:, :2]
        out["top2_accuracy"] = float(np.mean([idx.get(t, -1) in row for t, row in zip(y_true, top2)]))
    return out
