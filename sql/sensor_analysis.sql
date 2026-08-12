-- CTE-based sensor analysis joined to engine lifecycle attributes.
-- Demonstrates: CTE, aggregation, and a meaningful join.

WITH sensor_means AS (
    SELECT
        engine_id,
        AVG(sensor_02) AS mean_sensor_02,
        AVG(sensor_03) AS mean_sensor_03,
        AVG(sensor_04) AS mean_sensor_04,
        AVG(sensor_07) AS mean_sensor_07,
        AVG(sensor_11) AS mean_sensor_11,
        AVG(sensor_12) AS mean_sensor_12,
        AVG(sensor_15) AS mean_sensor_15,
        AVG(sensor_04_rolling_std) AS mean_sensor_04_rolling_std,
        AVG(rul) AS mean_rul
    FROM sensor_features
    GROUP BY engine_id
)
SELECT
    s.engine_id,
    e.n_cycles,
    e.last_cycle,
    s.mean_sensor_02,
    s.mean_sensor_04,
    s.mean_sensor_11,
    s.mean_sensor_15,
    s.mean_sensor_04_rolling_std,
    s.mean_rul,
    CASE
        WHEN e.n_cycles >= (
            SELECT PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY n_cycles)
            FROM engine_lifecycle
        ) THEN 'long_life'
        WHEN e.n_cycles <= (
            SELECT PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY n_cycles)
            FROM engine_lifecycle
        ) THEN 'short_life'
        ELSE 'typical_life'
    END AS life_band
FROM sensor_means AS s
INNER JOIN engine_lifecycle AS e
    ON s.engine_id = e.engine_id
ORDER BY e.n_cycles DESC, s.engine_id;
