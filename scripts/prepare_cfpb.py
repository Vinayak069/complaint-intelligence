"""Filter CFPB consumer complaints to credit-card complaints that have a consumer narrative.

CFPB stopped publishing narratives in its public download in August 2026, so the narratives
come from a Hugging Face mirror of the database taken before that (Mouwiya/cfpb-consumer-complaints,
complaints through January 2026). Column names are normalised, so the official CSV export, when
it carries narratives, works as input too.

Output: Parquet files partitioned by year and quarter, each well under GitHub's 100 MB file limit.

    python scripts/prepare_cfpb.py "hf://datasets/Mouwiya/cfpb-consumer-complaints/**/*.parquet" out_dir
    python scripts/prepare_cfpb.py complaints.csv out_dir
"""
import re
import sys
from pathlib import Path

import duckdb

FIELDS = {  # output name: normalised source name
    "complaint_id": "complaint_id", "date_received": "date_received", "date_sent": "date_sent_to_company",
    "product": "product", "sub_product": "sub_product", "issue": "issue", "sub_issue": "sub_issue",
    "narrative": "consumer_complaint_narrative", "company_public_response": "company_public_response",
    "company": "company", "state": "state", "zip": "zip_code", "tags": "tags", "submitted_via": "submitted_via",
    "company_response": "company_response_to_consumer", "timely": "timely_response",
}


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def main(source: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    if source.endswith(".csv"):
        con.execute(f"""CREATE VIEW raw AS SELECT * FROM read_csv('{source}', header = true, all_varchar = true,
                        quote = '"', escape = '"', max_line_size = 10000000)""")
    else:
        con.execute("INSTALL httpfs; LOAD httpfs;")
        con.execute(f"CREATE VIEW raw AS SELECT * FROM read_parquet('{source}', union_by_name = true)")
    cols = {norm(r[0]): r[0] for r in con.execute("DESCRIBE raw").fetchall()}
    print("columns", list(cols))
    missing = [v for v in FIELDS.values() if v not in cols]
    if missing:
        sys.exit(f"source lacks columns {missing}")

    def c(key: str) -> str:
        return '"' + cols[FIELDS[key]] + '"'

    con.execute(f"""
        CREATE TABLE cc AS
        SELECT TRY_CAST({c('complaint_id')} AS BIGINT) AS complaint_id,
               TRY_CAST({c('date_received')} AS DATE) AS date_received,
               TRY_CAST({c('date_sent')} AS DATE) AS date_sent,
               {', '.join(f'CAST({c(k)} AS VARCHAR) AS {k}' for k in FIELDS if k not in ('complaint_id', 'date_received', 'date_sent'))}
        FROM raw
        WHERE CAST({c('product')} AS VARCHAR) ILIKE '%credit card%'
          AND {c('narrative')} IS NOT NULL AND length(trim(CAST({c('narrative')} AS VARCHAR))) > 0
    """)
    con.execute("DELETE FROM cc WHERE complaint_id IN (SELECT complaint_id FROM cc GROUP BY 1 HAVING count(*) > 1)"
                " AND rowid NOT IN (SELECT min(rowid) FROM cc GROUP BY complaint_id)")
    con.execute(f"""
        COPY (SELECT *, year(date_received) AS year, quarter(date_received) AS q FROM cc)
        TO '{out}' (FORMAT parquet, COMPRESSION zstd, PARTITION_BY (year, q), OVERWRITE_OR_IGNORE)
    """)
    print("years", con.execute("""
        SELECT year(date_received), count(*),
               round(avg(CASE WHEN company_response IN ('Closed with monetary relief',
                   'Closed with non-monetary relief', 'Closed with relief') THEN 1 ELSE 0 END), 3)
        FROM cc GROUP BY 1 ORDER BY 1""").fetchall())
    print("products", con.execute("SELECT product, count(*) FROM cc GROUP BY 1 ORDER BY 2 DESC").fetchall())
    print("responses", con.execute("SELECT company_response, count(*) FROM cc GROUP BY 1 ORDER BY 2 DESC").fetchall())
    print("top companies", con.execute("SELECT company, count(*) FROM cc GROUP BY 1 ORDER BY 2 DESC LIMIT 12").fetchall())
    biggest = max(p.stat().st_size for p in out.rglob("*.parquet"))
    print("files", len(list(out.rglob("*.parquet"))), "largest bytes", biggest)
    if biggest > 90 * 2**20:
        sys.exit("a partition is too large for git")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
