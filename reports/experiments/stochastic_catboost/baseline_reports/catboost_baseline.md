# CatBoost Baseline

## Summary

- Fold RMSE: 0.047241, 0.044541, 0.064157, 0.041333, 0.072102
- Mean CV: 0.053875
- Fold std: 0.013476
- OOF RMSE: 0.055467
- OOF coverage rows: 62,077
- OOF coverage ratio: 80.31%
- Training runtime: 82.68 seconds
- Feature count: 14
- Enabled features: hour, minute, day_index, hour_sin, hour_cos, minute_sin, minute_cos, RoadType, Weather, Temperature, NumberOfLanes, LargeVehicles, lag1_demand, rolling_mean_3
- Fold metrics: /home/aashisoni/Codes/traffic-demand-prediction/artifacts/experiments/stochastic_catboost/models/catboost_fold_metrics.csv
- OOF predictions: /home/aashisoni/Codes/traffic-demand-prediction/oof/experiments/stochastic_catboost/catboost/oof_predictions.csv
- CatBoost parameters: /home/aashisoni/Codes/traffic-demand-prediction/artifacts/experiments/stochastic_catboost/models/catboost_params.json

## Fold Scores

| Fold | RMSE | Train rows | Validation rows | Train runtime (s) | Best iteration |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.047241 | 14,308 | 16,210 | 4.56 | 201 |
| 2 | 0.044541 | 30,537 | 15,956 | 8.42 | 221 |
| 3 | 0.064157 | 46,556 | 10,581 | 6.49 | 73 |
| 4 | 0.041333 | 57,557 | 5,986 | 11.81 | 172 |
| 5 | 0.072102 | 63,502 | 13,344 | 51.03 | 961 |

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
