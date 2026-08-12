-- Engine-level aggregations for predictive-maintenance review.
-- Demonstrates GROUP BY aggregations over the sensor_features table.

SELECT
    engine_id,
    COUNT(*) AS n_cycles,
    MIN(cycle) AS first_cycle,
    MAX(cycle) AS last_cycle,
    AVG(sensor_02) AS avg_sensor_02,
    MIN(sensor_02) AS min_sensor_02,
    MAX(sensor_02) AS max_sensor_02,
    AVG(sensor_04) AS avg_sensor_04,
    MIN(sensor_04) AS min_sensor_04,
    MAX(sensor_04) AS max_sensor_04,
    AVG(sensor_11) AS avg_sensor_11,
    AVG(rul) AS avg_rul,
    MIN(rul) AS min_rul,
    MAX(rul) AS max_rul
FROM sensor_features
GROUP BY engine_id
ORDER BY engine_id;
