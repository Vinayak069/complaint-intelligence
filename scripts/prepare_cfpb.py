"""Filter the CFPB Consumer Complaint Database to credit-card complaints with a narrative.

Input: complaints.csv from https://files.consumerfinance.gov/ccdb/complaints.csv.zip (public domain).
Output: Parquet files partitioned by year and quarter, each well under GitHub's 100 MB file limit.

    python scripts/prepare_cfpb.py complaints.csv out_dir
"""
import csv
import sys
from pathlib import Path

import duckdb


def main(csv_path: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        header = next(csv.reader(f))
    print("header", header)
    norm = {h.strip().lower(): h for h in header}

    def col(name: str) -> str:
        return '"' + norm[name.lower()].replace('"', '""') + '"'

    con.execute(f"""
        CREATE TABLE cc AS
        SELECT
            CAST({col("Complaint ID")} AS BIGINT)                 AS complaint_id,
            CAST({col("Date received")} AS DATE)                  AS date_received,
            CAST({col("Date sent to company")} AS DATE)           AS date_sent,
            {col("Product")}                                      AS product,
            {col("Sub-product")}                                  AS sub_product,
            {col("Issue")}                                        AS issue,
            {col("Sub-issue")}                                    AS sub_issue,
            {col("Consumer complaint narrative")}                 AS narrative,
            {col("Company public response")}                      AS company_public_response,
            {col("Company")}                                      AS company,
            {col("State")}                                        AS state,
            {col("ZIP code")}                                     AS zip,
            {col("Tags")}                                         AS tags,
            {col("Submitted via")}                                AS submitted_via,
            {col("Company response to consumer")}                 AS company_response,
            {col("Timely response?")}                             AS timely,
            {col("Consumer disputed?")}                           AS disputed
        FROM read_csv('{csv_path}', header = true, all_varchar = true, quote = '"', escape = '"',
                      max_line_size = 10000000)
        WHERE {col("Product")} ILIKE '%credit card%'
          AND {col("Consumer complaint narrative")} IS NOT NULL
          AND length({col("Consumer complaint narrative")}) > 0
    """)
    con.execute(f"""
        COPY (SELECT *, year(date_received) AS year, quarter(date_received) AS q FROM cc)
        TO '{out}' (FORMAT parquet, COMPRESSION zstd, PARTITION_BY (year, q), OVERWRITE_OR_IGNORE)
    """)
    summary = con.execute("""
        SELECT year(date_received) AS year, count(*) AS n,
               round(avg(CASE WHEN company_response IN ('Closed with monetary relief',
                   'Closed with non-monetary relief', 'Closed with relief') THEN 1 ELSE 0 END), 3) AS relief_rate
        FROM cc GROUP BY 1 ORDER BY 1
    """).fetchall()
    print("years", summary)
    print("products", con.execute("SELECT product, count(*) FROM cc GROUP BY 1 ORDER BY 2 DESC").fetchall())
    print("responses", con.execute(
        "SELECT company_response, count(*) FROM cc GROUP BY 1 ORDER BY 2 DESC").fetchall())
    print("top companies", con.execute(
        "SELECT company, count(*) FROM cc GROUP BY 1 ORDER BY 2 DESC LIMIT 12").fetchall())
    biggest = max(p.stat().st_size for p in out.rglob("*.parquet"))
    print("files", len(list(out.rglob("*.parquet"))), "largest bytes", biggest)
    if biggest > 90 * 2**20:
        sys.exit("a partition is too large for git")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
