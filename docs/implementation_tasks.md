
# Implementation Tasks

PHASE 0 — Setup
- create folder structure
- create configs
- setup seeds
- setup logging
- setup train.py

PHASE 1 — Data Pipeline
- build load_data()
- schema validation
- timestamp parsing
- sorting
- duplicate checks

PHASE 2 — Validation
- create purged folds
- save fold artifacts
- verify embargo
- dummy CV run

PHASE 3 — Baseline Features
- temporal features
- cyclical encodings
- categoricals
- baseline feature report

PHASE 4 — Baseline Model
- CatBoost baseline
- OOF generation
- fold metrics
- reproducibility test

PHASE 5 — Grouped Features
- geohash-hour means
- weather-hour means
- road-hour means
- fold-safe implementation

PHASE 6 — Lag Features
- lag1
- lag24
- rolling mean
- rolling std

PHASE 7 — Geospatial
- geohash neighbors
- neighbor demand lag
- congestion propagation

PHASE 8 — Ensembles
- model diversity
- hill climbing
- weighted blends

Daily execution rule:
- one stable improvement at a time
- verify leakage every phase
- never stack unverified features
