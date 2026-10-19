-- Credit-card complaints with a narrative, one row per complaint, with labels and split.
-- {source} is a Parquet glob of the CFPB extract (see scripts/prepare_cfpb.py).
CREATE OR REPLACE VIEW raw AS
SELECT * FROM read_parquet('{source}', hive_partitioning = true);

CREATE OR REPLACE TABLE complaints AS
SELECT
    complaint_id,
    date_received,
    date_trunc('month', date_received)::DATE                          AS month,
    coalesce(sub_product, 'Unknown')                                  AS sub_product,
    issue,
    coalesce(sub_issue, 'None')                                       AS sub_issue,
    company,
    coalesce(state, 'NA')                                             AS state,
    coalesce(tags, 'None')                                            AS tags,
    narrative,
    company_response,
    CASE WHEN company_response IN ('Closed with monetary relief', 'Closed with non-monetary relief')
         THEN 1 ELSE 0 END                                            AS relief,
    CASE WHEN company_response = 'Closed with monetary relief' THEN 1 ELSE 0 END AS monetary_relief,
    CASE
        WHEN date_received <  DATE '{val_start}'  THEN 'train'
        WHEN date_received <  DATE '{test_start}' THEN 'val'
        ELSE 'test'
    END                                                               AS split
FROM raw
WHERE date_received >= DATE '{train_start}'
  AND company_response IN ('Closed with explanation', 'Closed with monetary relief',
                           'Closed with non-monetary relief')        -- final outcomes only
  AND coalesce(sub_product, 'Unknown') IN ('General-purpose credit card or charge card',
                                           'Store credit card', 'Unknown');  -- not prepaid or gift cards

-- Company relief history, computed only from complaints closed before each month starts, so a
-- feature never uses an outcome the triage team could not have known yet.
CREATE OR REPLACE TABLE company_month AS
WITH m AS (
    SELECT company, month, count(*) AS n, sum(relief) AS r FROM complaints GROUP BY 1, 2
)
SELECT company, month,
       coalesce(sum(n) OVER w, 0) AS prior_n,
       coalesce(sum(r) OVER w, 0) AS prior_relief
FROM m
WINDOW w AS (PARTITION BY company ORDER BY month ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING);

CREATE OR REPLACE TABLE features AS
SELECT c.*,
       length(c.narrative)                                                AS text_len,
       len(regexp_extract_all(c.narrative, '\{\$[0-9,.]+\}'))              AS n_amounts,
       coalesce(list_max(list_transform(regexp_extract_all(c.narrative, '\{\$([0-9,.]+)\}', 1),
                x -> TRY_CAST(replace(x, ',', '') AS DOUBLE))), 0)          AS max_amount,
       len(regexp_extract_all(c.narrative, 'XXXX'))                       AS n_redactions,
       coalesce(cm.prior_n, 0)                                             AS company_prior_n,
       (coalesce(cm.prior_relief, 0) + 10 * 0.3) / (coalesce(cm.prior_n, 0) + 10)  AS company_prior_relief_rate,
       month(c.date_received)                                              AS month_of_year
FROM complaints c
LEFT JOIN company_month cm USING (company, month);
