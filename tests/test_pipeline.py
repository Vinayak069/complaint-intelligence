import numpy as np
import pandas as pd
import pytest

from complaints import data, models, monitoring


@pytest.fixture
def source(tmp_path):
    rows = []
    for i, (d, co, resp) in enumerate([
        ("2018-01-10", "A", "Closed with monetary relief"),
        ("2018-01-20", "A", "Closed with explanation"),
        ("2018-02-05", "A", "Closed with non-monetary relief"),
        ("2018-02-06", "B", "Closed with explanation"),
        ("2024-08-01", "A", "Closed with explanation"),
        ("2025-03-01", "A", "Closed with monetary relief"),
        ("2025-03-02", "A", "In progress"),                     # not a final outcome
        ("2017-12-31", "A", "Closed with monetary relief"),     # before the window
    ]):
        rows.append({"complaint_id": i, "date_received": pd.Timestamp(d).date(), "sub_product": "Store credit card",
                     "issue": "Fees or interest", "sub_issue": None, "company": co, "state": "CA", "tags": None,
                     "narrative": "I was charged {$1,250.00} by XXXX on XX/XX/XXXX", "company_response": resp})
    p = tmp_path / "c.parquet"
    pd.DataFrame(rows).to_parquet(p)
    return str(p)


def test_labels_splits_and_filters(source):
    df = data.load_features(source).set_index("complaint_id")
    assert set(df.index) == {0, 1, 2, 3, 4, 5}
    assert df.loc[[0, 2, 5], "relief"].tolist() == [1, 1, 1] and df.loc[0, "monetary_relief"] == 1
    assert df.loc[2, "monetary_relief"] == 0
    assert df.loc[[0, 3], "split"].tolist() == ["train", "train"]
    assert df.loc[4, "split"] == "val" and df.loc[5, "split"] == "test"
    assert df.loc[0, "queue"] == "Fees and interest"
    assert df.loc[0, "max_amount"] == 1250.0 and df.loc[0, "n_amounts"] == 1


def test_company_history_uses_only_earlier_months(source):
    df = data.load_features(source).set_index("complaint_id")
    # January complaints see no history; February sees January's two (one relief).
    assert df.loc[0, "company_prior_n"] == 0 and df.loc[1, "company_prior_n"] == 0
    assert df.loc[2, "company_prior_n"] == 2
    assert df.loc[2, "company_prior_relief_rate"] == pytest.approx((1 + 3) / (2 + 10))
    assert df.loc[3, "company_prior_n"] == 0  # company B has no history


def test_clean_text_collapses_redactions():
    assert data.clean_text("Dear XXXX,  on XX/XX/XXXX  I paid") == "Dear [REDACTED] , on [REDACTED] I paid"


def test_binary_metrics_top_fraction():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    p = np.linspace(1, 0, 10)
    m = models.binary_metrics(y, p)
    assert m["auc"] == 1.0 and m["relief_captured_top20pct"] == 1.0 and m["lift_top20pct"] == pytest.approx(5.0)


def test_psi():
    rng = np.random.default_rng(0)
    a = rng.normal(size=20000)
    assert monitoring.psi(a, rng.normal(size=20000)) < 0.01
    assert monitoring.psi(a, rng.normal(1.0, size=20000)) > 0.25
    s = pd.Series(["x"] * 50 + ["y"] * 50)
    assert monitoring.categorical_psi(s, s) == pytest.approx(0)
