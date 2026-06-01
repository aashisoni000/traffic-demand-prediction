# Baseline Features

## Feature Groups

- Categorical columns: RoadType, Weather
- Numeric columns: hour, minute, day_index, hour_sin, hour_cos, minute_sin, minute_cos, Temperature, NumberOfLanes, LargeVehicles, lag1_demand, lag2_demand, lag3_demand, lag6_demand, rolling_mean_3
- Temporal columns: hour, minute, day_index
- Fold-safe target mean columns: geohash_hour_mean, RoadType_hour_mean
- Fold-safe lag columns: lag1_demand, lag2_demand, lag3_demand, lag6_demand, rolling_mean_3

## Feature Table

| Feature | Train dtype | Train nulls | Test dtype | Test nulls |
| --- | --- | ---: | --- | ---: |
| hour | Int64 | 0 | Int64 | 0 |
| minute | Int64 | 0 | Int64 | 0 |
| day_index | Int64 | 0 | Int64 | 0 |
| hour_sin | float64 | 0 | float64 | 0 |
| hour_cos | float64 | 0 | float64 | 0 |
| minute_sin | float64 | 0 | float64 | 0 |
| minute_cos | float64 | 0 | float64 | 0 |
| RoadType | string | 600 | string | 324 |
| Weather | string | 797 | string | 431 |
| Temperature | float64 | 2,495 | float64 | 1,349 |
| NumberOfLanes | int64 | 0 | int64 | 0 |
| LargeVehicles | float64 | 0 | float64 | 0 |
| lag1_demand | float64 | 1,378 | float64 | 25 |
| lag2_demand | float64 | 2,716 | float64 | 42 |
| lag3_demand | float64 | 3,973 | float64 | 134 |
| lag6_demand | float64 | 7,746 | float64 | 418 |
| rolling_mean_3 | float64 | 3,973 | float64 | 134 |

## Summary

- Train rows: 77,299
- Test rows: 41,778
- Train feature memory: 17.44 MB
- Test feature memory: 9.41 MB
