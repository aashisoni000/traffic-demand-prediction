# CatBoost Baseline

## Summary

- Fold RMSE: 0.074474, 0.083417, 0.073131, 0.052378, 0.069026
- Mean CV: 0.070485
- Fold std: 0.011403
- OOF RMSE: 0.073737
- OOF coverage rows: 62,077
- OOF coverage ratio: 80.31%
- Training runtime: 17.90 seconds
- Feature count: 12
- Fold metrics: /home/aashisoni/Codes/traffic-demand-prediction/artifacts/models/catboost_fold_metrics.csv
- OOF predictions: /home/aashisoni/Codes/traffic-demand-prediction/oof/catboost/oof_predictions.csv
- CatBoost parameters: /home/aashisoni/Codes/traffic-demand-prediction/artifacts/models/catboost_params.json

## Fold Scores

| Fold | RMSE | Train rows | Validation rows | Train runtime (s) | Best iteration |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.074474 | 14,308 | 16,210 | 1.37 | 153 |
| 2 | 0.083417 | 30,537 | 15,956 | 2.95 | 191 |
| 3 | 0.073131 | 46,556 | 10,581 | 3.91 | 175 |
| 4 | 0.052378 | 57,557 | 5,986 | 6.35 | 221 |
| 5 | 0.069026 | 63,502 | 13,344 | 2.97 | 59 |

## CatBoost Parameters

```json
{
  "loss_function": "RMSE",
  "eval_metric": "RMSE",
  "iterations": 500,
  "learning_rate": 0.05,
  "depth": 6,
  "l2_leaf_reg": 3.0,
  "random_seed": 42,
  "bootstrap_type": "No",
  "random_strength": 0.0,
  "od_type": "Iter",
  "od_wait": 50,
  "allow_writing_files": false,
  "verbose": false,
  "thread_count": 1
}
```
