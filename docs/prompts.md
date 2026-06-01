
# AI Prompts

## Fold-Safe Grouped Features

Objective:
Build leakage-safe grouped aggregations.

Requirements:
- compute inside fold only
- no future leakage
- deterministic
- config-driven
- parquet outputs

Forbidden:
- global grouped means
- random CV
- fit_transform on full dataset

---

## Lag Features

Objective:
Build lag and rolling features safely.

Requirements:
- sort by [geohash, timestamp]
- use shift(1)+ only
- deterministic
- fold-safe

Outputs:
- lag1
- lag24
- rolling mean
- rolling std

Forbidden:
- shift(0)
- unsorted rolling windows

---

## Geospatial Features

Objective:
Build neighbor congestion features.

Requirements:
- use geohash neighbors
- no future leakage
- lagged neighbor demand only
- deterministic execution

Outputs:
- neighbor_mean_lag1
- neighbor_density
- shockwave proxies

---

## Model Training

Objective:
Train deterministic fold-safe models.

Requirements:
- fixed seed
- purged CV
- OOF generation
- fold metrics
- config snapshots

Outputs:
- fold scores
- OOF predictions
- feature importance
- SHAP summaries
