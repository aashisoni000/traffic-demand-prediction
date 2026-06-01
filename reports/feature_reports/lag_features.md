# Fold-Safe Lag Features

## Summary

### lag1_demand

- Target: demand
- Group column: geohash
- Lag order: 1
- Ordering: strictly chronological by day, hour, minute within geohash
- Validation lag source: training fold rows only
- Test lag source: full training rows only
- Train coverage rows: 75,921 (98.22%)
- Train NaN rows: 1,378 (1.78%)
- Test coverage rows: 41,753 (99.94%)
- Test NaN rows: 25 (0.06%)
- Validation coverage rows: 61,848 (99.63%)
- Validation NaN rows: 229 (0.37%)

### lag2_demand

- Target: demand
- Group column: geohash
- Lag order: 2
- Ordering: strictly chronological by day, hour, minute within geohash
- Validation lag source: training fold rows only
- Test lag source: full training rows only
- Train coverage rows: 74,583 (96.49%)
- Train NaN rows: 2,716 (3.51%)
- Test coverage rows: 41,736 (99.90%)
- Test NaN rows: 42 (0.10%)
- Validation coverage rows: 61,600 (99.23%)
- Validation NaN rows: 477 (0.77%)

### lag3_demand

- Target: demand
- Group column: geohash
- Lag order: 3
- Ordering: strictly chronological by day, hour, minute within geohash
- Validation lag source: training fold rows only
- Test lag source: full training rows only
- Train coverage rows: 73,326 (94.86%)
- Train NaN rows: 3,973 (5.14%)
- Test coverage rows: 41,644 (99.68%)
- Test NaN rows: 134 (0.32%)
- Validation coverage rows: 61,385 (98.89%)
- Validation NaN rows: 692 (1.11%)

### lag6_demand

- Target: demand
- Group column: geohash
- Lag order: 6
- Ordering: strictly chronological by day, hour, minute within geohash
- Validation lag source: training fold rows only
- Test lag source: full training rows only
- Train coverage rows: 69,553 (89.98%)
- Train NaN rows: 7,746 (10.02%)
- Test coverage rows: 41,360 (99.00%)
- Test NaN rows: 418 (1.00%)
- Validation coverage rows: 60,565 (97.56%)
- Validation NaN rows: 1,512 (2.44%)

### rolling_mean_3

- Target: demand
- Group column: geohash
- Lag order: 3
- Ordering: strictly chronological by day, hour, minute within geohash
- Validation lag source: training fold rows only
- Test lag source: full training rows only
- Train coverage rows: 73,326 (94.86%)
- Train NaN rows: 3,973 (5.14%)
- Test coverage rows: 41,644 (99.68%)
- Test NaN rows: 134 (0.32%)
- Validation coverage rows: 61,385 (98.89%)
- Validation NaN rows: 692 (1.11%)

## Fold Details

| Feature | Fold | Train rows | Validation rows | Train geohashes | Validation geohashes | Matched validation geohashes | Geohash coverage | Lag coverage rows | NaN rows | Chronological check |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| lag1_demand | 1 | 14,308 | 16,210 | 1,140 | 1,164 | 1,099 | 94.42% | 16,066 (99.11%) | 144 (0.89%) | PASS |
| lag1_demand | 2 | 30,537 | 15,956 | 1,205 | 1,162 | 1,133 | 97.50% | 15,906 (99.69%) | 50 (0.31%) | PASS |
| lag1_demand | 3 | 46,556 | 10,581 | 1,234 | 978 | 976 | 99.80% | 10,579 (99.98%) | 2 (0.02%) | PASS |
| lag1_demand | 4 | 57,557 | 5,986 | 1,237 | 666 | 665 | 99.85% | 5,985 (99.98%) | 1 (0.02%) | PASS |
| lag1_demand | 5 | 63,502 | 13,344 | 1,238 | 1,101 | 1,090 | 99.00% | 13,312 (99.76%) | 32 (0.24%) | PASS |
| lag2_demand | 1 | 14,308 | 16,210 | 1,140 | 1,164 | 1,099 | 94.42% | 15,907 (98.13%) | 303 (1.87%) | PASS |
| lag2_demand | 2 | 30,537 | 15,956 | 1,205 | 1,162 | 1,133 | 97.50% | 15,848 (99.32%) | 108 (0.68%) | PASS |
| lag2_demand | 3 | 46,556 | 10,581 | 1,234 | 978 | 976 | 99.80% | 10,576 (99.95%) | 5 (0.05%) | PASS |
| lag2_demand | 4 | 57,557 | 5,986 | 1,237 | 666 | 665 | 99.85% | 5,984 (99.97%) | 2 (0.03%) | PASS |
| lag2_demand | 5 | 63,502 | 13,344 | 1,238 | 1,101 | 1,090 | 99.00% | 13,285 (99.56%) | 59 (0.44%) | PASS |
| lag3_demand | 1 | 14,308 | 16,210 | 1,140 | 1,164 | 1,099 | 94.42% | 15,754 (97.19%) | 456 (2.81%) | PASS |
| lag3_demand | 2 | 30,537 | 15,956 | 1,205 | 1,162 | 1,133 | 97.50% | 15,804 (99.05%) | 152 (0.95%) | PASS |
| lag3_demand | 3 | 46,556 | 10,581 | 1,234 | 978 | 976 | 99.80% | 10,569 (99.89%) | 12 (0.11%) | PASS |
| lag3_demand | 4 | 57,557 | 5,986 | 1,237 | 666 | 665 | 99.85% | 5,984 (99.97%) | 2 (0.03%) | PASS |
| lag3_demand | 5 | 63,502 | 13,344 | 1,238 | 1,101 | 1,090 | 99.00% | 13,274 (99.48%) | 70 (0.52%) | PASS |
| lag6_demand | 1 | 14,308 | 16,210 | 1,140 | 1,164 | 1,099 | 94.42% | 15,193 (93.73%) | 1,017 (6.27%) | PASS |
| lag6_demand | 2 | 30,537 | 15,956 | 1,205 | 1,162 | 1,133 | 97.50% | 15,638 (98.01%) | 318 (1.99%) | PASS |
| lag6_demand | 3 | 46,556 | 10,581 | 1,234 | 978 | 976 | 99.80% | 10,556 (99.76%) | 25 (0.24%) | PASS |
| lag6_demand | 4 | 57,557 | 5,986 | 1,237 | 666 | 665 | 99.85% | 5,982 (99.93%) | 4 (0.07%) | PASS |
| lag6_demand | 5 | 63,502 | 13,344 | 1,238 | 1,101 | 1,090 | 99.00% | 13,196 (98.89%) | 148 (1.11%) | PASS |
| rolling_mean_3 | 1 | 14,308 | 16,210 | 1,140 | 1,164 | 1,099 | 94.42% | 15,754 (97.19%) | 456 (2.81%) | PASS |
| rolling_mean_3 | 2 | 30,537 | 15,956 | 1,205 | 1,162 | 1,133 | 97.50% | 15,804 (99.05%) | 152 (0.95%) | PASS |
| rolling_mean_3 | 3 | 46,556 | 10,581 | 1,234 | 978 | 976 | 99.80% | 10,569 (99.89%) | 12 (0.11%) | PASS |
| rolling_mean_3 | 4 | 57,557 | 5,986 | 1,237 | 666 | 665 | 99.85% | 5,984 (99.97%) | 2 (0.03%) | PASS |
| rolling_mean_3 | 5 | 63,502 | 13,344 | 1,238 | 1,101 | 1,090 | 99.00% | 13,274 (99.48%) | 70 (0.52%) | PASS |

## Verification

- No future leakage: PASS
- Chronological correctness: PASS
- Validation rows use only fold training history.
- Lag values are missing only when no prior row exists in the permitted history for that geohash.
