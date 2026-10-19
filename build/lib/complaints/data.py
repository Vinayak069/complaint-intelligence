"""Loads the CFPB extract with DuckDB, builds labels and leakage-safe features in SQL
(sql/base.sql), and maps CFPB issues to servicing queues."""
from __future__ import annotations

import re
from importlib.resources import files

import duckdb
import pandas as pd

SPLITS = {"train_start": "2018-01-01", "val_start": "2024-07-01", "test_start": "2025-01-01"}

# CFPB's consumer-chosen issue -> the servicing queue that would own it. Equivalent issues that
# CFPB renamed over the years go to the same queue.
QUEUES = {
    "Problem with a purchase shown on your statement": "Billing dispute",
    "Problem with a purchase or transfer": "Billing dispute",
    "Fees or interest": "Fees and interest",
    "Unexpected or other fees": "Fees and interest",
    "Getting a credit card": "Applications",
    "Closing your account": "Account closure",
    "Problem getting a card or closing an account": "Account closure",
    "Problem when making payments": "Payments",
    "Struggling to pay your bill": "Payments",
    "Advertising and marketing, including promotional offers": "Rewards and offers",
    "Advertising": "Rewards and offers",
    "Incorrect information on your report": "Credit reporting",
    "Problem with a credit reporting company's investigation into an existing problem": "Credit reporting",
    "Improper use of your report": "Credit reporting",
    "Problem with a company's investigation into an existing problem": "Escalated investigation",
    "Trouble using your card": "Card usage",
    "Trouble using the card": "Card usage",
    "Other features, terms, or problems": "Other",
}


def connect(source: str, **splits) -> duckdb.DuckDBPyConnection:
    params = {**SPLITS, **splits, "source": source}
    sql = files("complaints").joinpath("sql/base.sql").read_text()
    for k, v in params.items():
        sql = sql.replace("{" + k + "}", v)
    sql = "\n".join(line for line in sql.splitlines() if not line.lstrip().startswith("--"))
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    con.execute(sql)
    return con


def load_features(source: str, **splits) -> pd.DataFrame:
    con = connect(source, **splits)
    df = con.execute("SELECT * FROM features ORDER BY date_received, complaint_id").df()
    df["queue"] = df["issue"].map(QUEUES).fillna("Other")
    return df


_MASK = re.compile(r"X{2,}(/X{2,})*")


def clean_text(text: str) -> str:
    """CFPB redacts names, dates and account numbers as XXXX; collapse each redaction to one token."""
    return re.sub(r"\s+", " ", _MASK.sub(" [REDACTED] ", text)).strip()
