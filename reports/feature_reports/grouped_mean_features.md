# Fold-Safe Grouped Mean Features

## Summary

## Fold Details

| Feature | Fold | Train buckets | Validation buckets | Matched validation buckets | Validation bucket coverage | Exact rows | Hour fallback rows | Global fallback rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

## Verification

- Fold mappings are fit on training fold rows only.
- Validation rows are transformed from their fold-specific mapping.
- Test rows are transformed from a full-training mapping for train/test parity.
- NaN checks: PASS
- Train/test feature parity checks run in the baseline feature bundle.
