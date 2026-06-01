
# Baseline Execution

Goal:
Build a fully reproducible baseline before advanced modeling.

DO NOT:
- ensemble early
- pseudo-label early
- tune aggressively
- optimize public leaderboard early

Baseline phases:

PHASE 0
- setup environment
- deterministic seeds
- config system
- logging

PHASE 1
- centralized data loading
- schema validation
- timestamp parsing
- duplicate checks

PHASE 2
- purged time-series folds
- embargo verification
- dummy CV

PHASE 3
- baseline temporal features
- categorical handling
- feature audit

PHASE 4
- CatBoost baseline
- OOF generation
- metrics logging

Required baseline features:
- hour
- weekday
- weekend
- cyclical encodings
- RoadType
- NumberofLanes
- Weather
- Temperature
- LargeVehicles

Mandatory outputs:
- fold scores
- OOF predictions
- config snapshot
- feature list
- runtime metrics

Hard fail conditions:
- leakage
- fold overlap
- timestamp corruption
- rerun instability
- train/inference mismatch

Acceptance criteria:
- deterministic reruns
- fold std <= 0.01
- reproducible artifacts
- stable CV

After baseline:
1. grouped stats
2. lag features
3. rolling features
4. geospatial features
5. interactions
6. ensembles
