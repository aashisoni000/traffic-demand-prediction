# Traffic Demand Prediction Solution

## Overview

This project solves the traffic demand prediction problem using a fold-safe machine learning pipeline built with CatBoost and time-aware validation.

The system focuses on:

* leakage prevention
* temporal consistency
* robust feature engineering
* stable cross-validation
* ensemble-based inference

---

# Model Architecture

## Core Model

* CatBoostRegressor

## Validation Strategy

* Purged Time-Series Cross Validation
* Fold embargo protection
* Strict chronological separation

This prevents future information leakage during training and validation.

---

# Feature Engineering

## Temporal Features

* hour
* minute
* day_index
* hour_sin
* hour_cos
* minute_sin
* minute_cos

## Road and Traffic Features

* RoadType
* Weather
* Temperature
* NumberOfLanes
* LargeVehicles

## Fold-Safe Lag Features

* lag1_demand
* lag2_demand
* lag3_demand
* lag6_demand
* rolling_mean_3

All lag features are generated strictly from past observations within each geohash region.

---

# Training Improvements

## CatBoost Tuning

The final experiments used:

* Bayesian bootstrap
* stochastic regularization
* deeper trees
* lower learning rate
* extended boosting iterations

## Ensemble Strategy

Final predictions are generated using a weighted ensemble of multiple stochastic CatBoost experiments.

Weights:

* exp07: 0.2
* exp08: 0.2
* exp09: 0.6

---

# Project Structure

## Important Files

### Training

* train.py

### Inference

* predict.py
* weighted_ensemble.py

### Feature Engineering

* features/temporal.py

### Validation

* validation/

### Configurations

* configs/

---

# Final Output

The final submission file:

* preserves original row ordering
* contains no NaN predictions
* matches competition submission schema

Generated file:

* final_weighted_ensemble_submission.csv

---

# Libraries Used

* Python
* Pandas
* NumPy
* CatBoost
* Scikit-learn
* PyYAML

---

# Notes

The pipeline was designed to maintain strict train/test parity and avoid data leakage during feature engineering and validation.
