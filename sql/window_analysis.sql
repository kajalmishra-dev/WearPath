-- Per-engine sequential analysis using SQL window functions.
-- Demonstrates: LAG, LEAD, ROW_NUMBER, and rolling window aggregates.
-- Partitioned by engine_id and ordered by cycle.

SELECT
    engine_id,
    cycle,
    sensor_04,
    rul,
    ROW_NUMBER() OVER (
        PARTITION BY engine_id
        ORDER BY cycle
    ) AS cycle_row_number,
    LAG(sensor_04, 1) OVER (
        PARTITION BY engine_id
        ORDER BY cycle
    ) AS sensor_04_lag,
    LEAD(sensor_04, 1) OVER (
        PARTITION BY engine_id
        ORDER BY cycle
    ) AS sensor_04_lead,
    sensor_04 - LAG(sensor_04, 1) OVER (
        PARTITION BY engine_id
        ORDER BY cycle
    ) AS sensor_04_sql_delta,
    AVG(sensor_04) OVER (
        PARTITION BY engine_id
        ORDER BY cycle
        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
    ) AS sensor_04_sql_rolling_mean,
    STDDEV_SAMP(sensor_04) OVER (
        PARTITION BY engine_id
        ORDER BY cycle
        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
    ) AS sensor_04_sql_rolling_std
FROM sensor_features_enriched
ORDER BY engine_id, cycle
LIMIT 5000;
