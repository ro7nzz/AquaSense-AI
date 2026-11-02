-- AquaSenseAI: SQL Analytics Queries
-- Database table: water_sensor_records

-- 1. Total records and label distribution
SELECT fault_d7, COUNT(*) AS record_count,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM water_sensor_records), 2) AS percentage
FROM water_sensor_records
GROUP BY fault_d7
ORDER BY fault_d7;

-- 2. Records by day
SELECT DATE(timestamp) AS record_date,
       COUNT(*) AS hourly_records,
       SUM(CASE WHEN fault_d7 = 1 THEN 1 ELSE 0 END) AS positive_labels
FROM water_sensor_records
GROUP BY DATE(timestamp)
ORDER BY record_date;

-- 3. Check timestamp range
SELECT MIN(timestamp) AS first_timestamp,
       MAX(timestamp) AS last_timestamp,
       COUNT(*) AS total_records
FROM water_sensor_records;

-- 4. Inspect recent records
SELECT timestamp, fault_d7
FROM water_sensor_records
ORDER BY timestamp DESC
LIMIT 20;
