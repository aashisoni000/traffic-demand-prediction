# CatBoost Baseline

## Summary

- Fold RMSE: 0.046572, 0.046602, 0.065166, 0.040902, 0.076594
- Mean CV: 0.055167
- Fold std: 0.015082
- OOF RMSE: 0.057191
- OOF coverage rows: 62,077
- OOF coverage ratio: 80.31%
- Training runtime: 85.19 seconds
- Feature count: 17
- Enabled features: hour, minute, day_index, hour_sin, hour_cos, minute_sin, minute_cos, RoadType, Weather, Temperature, NumberOfLanes, LargeVehicles, lag1_demand, lag2_demand, lag3_demand, lag6_demand, rolling_mean_3
- Fold metrics: /home/aashisoni/Codes/traffic-demand-prediction/artifacts/experiments/multi_lag/models/catboost_fold_metrics.csv
- OOF predictions: /home/aashisoni/Codes/traffic-demand-prediction/oof/experiments/multi_lag/catboost/oof_predictions.csv
- CatBoost parameters: /home/aashisoni/Codes/traffic-demand-prediction/artifacts/experiments/multi_lag/models/catboost_params.json

## Fold Scores

| Fold | RMSE | Train rows | Validation rows | Train runtime (s) | Best iteration |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.046572 | 14,308 | 16,210 | 8.43 | 388 |
| 2 | 0.046602 | 30,537 | 15,956 | 8.97 | 195 |
| 3 | 0.065166 | 46,556 | 10,581 | 7.35 | 72 |
| 4 | 0.040902 | 57,557 | 5,986 | 12.04 | 129 |
| 5 | 0.076594 | 63,502 | 13,344 | 48.03 | 761 |

## CatBoost Parameters

```json
{
  "loss_function": "RMSE",
  "eval_metric": "RMSE",
  "iterations": 1400,
  "learning_rate": 0.025,
  "depth": 8,
  "l2_leaf_reg": 6,
  "random_seed": 42,
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
