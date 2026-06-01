# Validation Fold Summary

- Total rows: 77,299
- Total time groups: 105
- Folds: 5
- Embargo groups: 1
- Validation fold size ratio: 2.71
- Smallest validation fold: 5,986
- Largest validation fold: 16,210

## Fold Details

| Fold | Train rows | Validation rows | Validation share | Train groups | Validation groups | Train range | Validation range | Embargo range | Overlap | Leakage |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | --- |
| 1 | 14,308 | 16,210 | 20.97% | 17 | 18 | from 48 00:00 to 48 04:00 | from 48 04:30 to 48 08:45 | 48 04:15 -> 48 04:15 | 0 | PASS |
| 2 | 30,537 | 15,956 | 20.64% | 35 | 18 | from 48 00:00 to 48 08:30 | from 48 09:00 to 48 13:15 | 48 08:45 -> 48 08:45 | 0 | PASS |
| 3 | 46,556 | 10,581 | 13.69% | 53 | 17 | from 48 00:00 to 48 13:00 | from 48 13:30 to 48 17:30 | 48 13:15 -> 48 13:15 | 0 | PASS |
| 4 | 57,557 | 5,986 | 7.74% | 70 | 17 | from 48 00:00 to 48 17:15 | from 48 17:45 to 48 21:45 | 48 17:30 -> 48 17:30 | 0 | PASS |
| 5 | 63,502 | 13,344 | 17.26% | 87 | 17 | from 48 00:00 to 48 21:30 | from 48 22:00 to 49 02:00 | 48 21:45 -> 48 21:45 | 0 | PASS |

## Verification

- Chronological ordering: PASS
- Embargo support: PASS
- Overlap checks: PASS
- Timestamp integrity: PASS
- Future leakage risk: PASS
- Fold imbalance warning: WARN

## Imbalance Warning

Validation fold sizes are imbalanced: smallest fold = 5,986, largest fold = 16,210, ratio = 2.71.
