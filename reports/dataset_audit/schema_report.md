# Schema Report

## Train

- Status: PASS
- Rows: 77,299
- Columns: 11
- Missing columns: None
- Extra columns: None
- Expected schema: Index, geohash, day, timestamp, demand, RoadType, NumberofLanes, LargeVehicles, Landmarks, Temperature, Weather
- Timestamp helper column: timestamp_parsed

## Test

- Status: PASS
- Rows: 41,778
- Columns: 10
- Missing columns: None
- Extra columns: None
- Expected schema: Index, geohash, day, timestamp, RoadType, NumberofLanes, LargeVehicles, Landmarks, Temperature, Weather
- Timestamp helper column: timestamp_parsed
