# CatBoost Baseline

## Summary

- Fold RMSE: 0.047246, 0.043680, 0.063848, 0.041187, 0.074742
- Mean CV: 0.054141
- Fold std: 0.014521
- OOF RMSE: 0.055969
- OOF coverage rows: 62,077
- OOF coverage ratio: 80.31%
- Training runtime: 84.72 seconds
- Feature count: 14
- Enabled features: hour, minute, day_index, hour_sin, hour_cos, minute_sin, minute_cos, RoadType, Weather, Temperature, NumberOfLanes, LargeVehicles, lag1_demand, rolling_mean_3
- Fold metrics: /home/aashisoni/Codes/traffic-demand-prediction/artifacts/experiments/seed99/models/catboost_fold_metrics.csv
- OOF predictions: /home/aashisoni/Codes/traffic-demand-prediction/oof/experiments/seed99/catboost/oof_predictions.csv
- CatBoost parameters: /home/aashisoni/Codes/traffic-demand-prediction/artifacts/experiments/seed99/models/catboost_params.json

## Fold Scores

| Fold | RMSE | Train rows | Validation rows | Train runtime (s) | Best iteration |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.047246 | 14,308 | 16,210 | 5.67 | 288 |
| 2 | 0.043680 | 30,537 | 15,956 | 10.52 | 306 |
| 3 | 0.063848 | 46,556 | 10,581 | 6.61 | 74 |
| 4 | 0.041187 | 57,557 | 5,986 | 10.19 | 131 |
| 5 | 0.074742 | 63,502 | 13,344 | 51.39 | 938 |

## CatBoost Parameters

```json
{
  "loss_function": "RMSE",
  "eval_metric": "RMSE",
  "iterations": 1400,
  "learning_rate": 0.025,
  "depth": 8,
  "l2_leaf_reg": 6,
  "random_seed": 99,
  "bootstrap_type": "Bayesian",
  "random_strength": 1,
  "od_type": "Iter",
  "od_wait": 120,
  "allow_writing_files": false,
  "verbose": false,
  "thread_count": 1,
  "bagging_temperature": 1
}
```
