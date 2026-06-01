# Baseline Features

## Feature Groups

- Categorical columns: RoadType, Weather
- Numeric columns: hour, minute, day_index, hour_sin, hour_cos, minute_sin, minute_cos, Temperature, NumberOfLanes, LargeVehicles
- Temporal columns: hour, minute, day_index

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

## Summary

- Train rows: 77,299
- Test rows: 41,778
- Train feature memory: 14.49 MB
- Test feature memory: 7.82 MB
