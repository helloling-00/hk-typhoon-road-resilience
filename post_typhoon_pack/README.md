# Post-Typhoon Analysis Data Pack

## Data files

### yagiasha_road_timeseries.parquet
Aggregated per-slot timeseries. Columns:
- `road_id`, `dt` (datetime), `slot` (0-47), `ds` (date string)
- `day_type`: WORKDAY / SATURDAY / SUNDAY
- `obs`: mean observed speed (km/h)
- `bl`: baseline speed (km/h)
- `dev`: deviation = obs - bl

Date range: 2025-09-21 ~ 2025-09-26 (6 days)

### regression_table.parquet
One row per road per typhoon phase. **271,649 rows, 29,497 unique roads**.
Contains ALL features: POI density (10 categories), demographics (population, income, age, employment ratios), road structure (intersection_degree, dist_to_coast, road_length, road_category), incidents (incident_count_500m, severe_incident_500m, closure_nearby_500m).

Key columns:
- `road_id`, `typhoon`, `signal_level`, `time_group`, `signal_group`
- `mean_speed`, `mean_deviation`, `n_slots`
- POI: `work_density`, `education_density`, `retail_density`, `food_drink_density`, `recreation_density`, `medical_density`, `transport_density`, `tourism_density`, `finance_density`, `civic_density`
- Demographics (500m buffer): `population_density_500m`, `median_income_500m`, `working_pop_ratio_500m`, `ratio_雇员_500m`, `ratio_学生_500m`, `ratio_退休人士_500m`, `ratio_age_0_14_500m`, `ratio_age_25_44_500m`, `ratio_age_65plus_500m`
- Structure: `intersection_degree`, `dist_to_coast_m`, `road_length_m`, `road_category`, `road_broad` (highway/arterial/local)
- Incidents: `incident_count_500m`, `severe_incident_500m`, `closure_nearby_500m`

### baseline_speed.parquet
Baseline speed by (cluster_id, day_type, slot). Columns: `cluster_id`, `day_type`, `slot`, `mean_speed`, `std_speed`, `n_obs`.

### ep_to_road.parquet
Geometry mapping: `ep_key` → `road_id`. Used to link WKB geometries to road IDs.

### road_registry.parquet
Road registry with `road_id`, `cluster_id`, `canonical_wkb`, etc.

## Typhoon Signal Timeline (Ragasa, Sep 2025)

| Signal | Start | End |
|--------|-------|-----|
| S1 | Sep 22 12:20 | Sep 22 21:40 |
| S3 | Sep 22 21:40 | Sep 23 14:20 |
| S8 | Sep 23 14:20 | Sep 24 01:40 |
| S10 | Sep 24 01:40 | Sep 24 13:20 |
| S8 | Sep 24 13:20 | Sep 24 20:20 |
| S3 | Sep 24 20:20 | Sep 25 08:20 |
| S1 | Sep 25 08:20 | Sep 25 11:20 |

Post-typhoon: Sep 25 11:20 onwards.

## Data gaps
- Sep 22 05:00–20:30: 32 slots missing (S1 period, nearly no data)

## Key definitions
- F (Faster): dev > +0.03
- S (Slower): dev < −0.03
- N (Near baseline): |dev| ≤ 0.03
- Slot 17 = 08:30, Slot 26 = 13:00

## Control workdays
8 clean Mon-Fri days: 2025-09-16, 09-26, 09-29, 09-30, 10-02, 10-06, 10-08, 10-09
