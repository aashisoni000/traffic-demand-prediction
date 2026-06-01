Traffic Demand Prediction – Solution Overview

Problem:
The task is to predict traffic demand using temporal, categorical, and historical traffic behavior features.

Approach:
I built a leakage-safe machine learning pipeline using CatBoost regression with purged time-series cross validation.

Main steps:

1. Data validation and reporting
2. Fold-safe time-series validation
3. Baseline temporal feature engineering
4. Lag-based historical demand features
5. Experiment tracking using YAML configs
6. Ensemble inference using fold averaging

Validation Strategy:
A purged cross-validation setup with embargo groups was used to avoid future leakage between train and validation splits.

Main Features Used:

* hour
* minute
* day_index
* cyclical time encodings
* RoadType
* Weather
* Temperature
* NumberOfLanes
* LargeVehicles
* lag1_demand
* rolling_mean_3

Key Engineering Decisions:

* Strict chronological feature generation
* Fold-safe lag features
* No future target leakage
* Train/inference feature parity
* Ensemble averaging across folds

Model:

* CatBoostRegressor
* RMSE optimization
* Fold ensemble inference

Best Experiment:
Configuration:
exp04_lag1_plus_rolling.yaml

Best Validation Metrics:

* Mean CV RMSE: 0.054805
* OOF RMSE: 0.056529

Tools & Libraries:

* Python
* Pandas
* NumPy
* CatBoost
* YAML
* VS Code

Important Files:

* train.py → training entrypoint
* predict.py → inference + submission generation
* features/temporal.py → feature engineering
* validation/purged_cv.py → fold generation
* models/train_catboost.py → CatBoost training
* configs/exp04_lag1_plus_rolling.yaml → best experiment config

Inference:
Predictions were generated using fold model averaging and exported in competition submission format.
