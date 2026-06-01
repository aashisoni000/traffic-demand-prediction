# CatBoost Baseline

## Summary

- Fold RMSE: 0.047777, 0.043067, 0.070703, 0.040553, 0.071927
- Mean CV: 0.054805
- Fold std: 0.015299
- OOF RMSE: 0.056529
- OOF coverage rows: 62,077
- OOF coverage ratio: 80.31%
- Training runtime: 21.03 seconds
- Feature count: 14
- Enabled features: hour, minute, day_index, hour_sin, hour_cos, minute_sin, minute_cos, RoadType, Weather, Temperature, NumberOfLanes, LargeVehicles, lag1_demand, rolling_mean_3
- Fold metrics: /home/aashisoni/Codes/traffic-demand-prediction/artifacts/experiments/lag1_plus_rolling/models/catboost_fold_metrics.csv
- OOF predictions: /home/aashisoni/Codes/traffic-demand-prediction/oof/experiments/lag1_plus_rolling/catboost/oof_predictions.csv
- CatBoost parameters: /home/aashisoni/Codes/traffic-demand-prediction/artifacts/experiments/lag1_plus_rolling/models/catboost_params.json

## Fold Scores

| Fold | RMSE | Train rows | Validation rows | Train runtime (s) | Best iteration |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.047777 | 14,308 | 16,210 | 3.13 | 296 |
| 2 | 0.043067 | 30,537 | 15,956 | 4.72 | 253 |
| 3 | 0.070703 | 46,556 | 10,581 | 1.93 | 36 |
| 4 | 0.040553 | 57,557 | 5,986 | 3.31 | 67 |
| 5 | 0.071927 | 63,502 | 13,344 | 7.54 | 187 |

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
