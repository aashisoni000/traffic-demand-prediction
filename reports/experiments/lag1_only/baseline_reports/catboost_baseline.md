# CatBoost Baseline

## Summary

- Fold RMSE: 0.049242, 0.046162, 0.068739, 0.041049, 0.071512
- Mean CV: 0.055341
- Fold std: 0.013845
- OOF RMSE: 0.056992
- OOF coverage rows: 62,077
- OOF coverage ratio: 80.31%
- Training runtime: 20.31 seconds
- Feature count: 13
- Enabled features: hour, minute, day_index, hour_sin, hour_cos, minute_sin, minute_cos, RoadType, Weather, Temperature, NumberOfLanes, LargeVehicles, lag1_demand
- Fold metrics: /home/aashisoni/Codes/traffic-demand-prediction/artifacts/experiments/lag1_only/models/catboost_fold_metrics.csv
- OOF predictions: /home/aashisoni/Codes/traffic-demand-prediction/oof/experiments/lag1_only/catboost/oof_predictions.csv
- CatBoost parameters: /home/aashisoni/Codes/traffic-demand-prediction/artifacts/experiments/lag1_only/models/catboost_params.json

## Fold Scores

| Fold | RMSE | Train rows | Validation rows | Train runtime (s) | Best iteration |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.049242 | 14,308 | 16,210 | 2.94 | 337 |
| 2 | 0.046162 | 30,537 | 15,956 | 2.79 | 145 |
| 3 | 0.068739 | 46,556 | 10,581 | 1.87 | 36 |
| 4 | 0.041049 | 57,557 | 5,986 | 3.14 | 67 |
| 5 | 0.071512 | 63,502 | 13,344 | 9.25 | 274 |

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
