
# Competition Playbook

Mission:
Build a robust traffic demand prediction system optimized for hidden leaderboard generalization.

Winning principles:
- validation quality > model complexity
- stable gains > flashy gains
- reproducibility > chaos
- fold-safe engineering > brute force

Core competition insight:
Traffic =
WHERE + WHEN + CAPACITY + HUMAN BEHAVIOR

Most important signals:
- rush hour patterns
- geohash behavior
- grouped aggregations
- lag demand
- weather interactions
- bottleneck effects
- neighbor congestion

Critical feature priorities:
1. temporal features
2. grouped OOF stats
3. lag features
4. rolling windows
5. geohash neighbors
6. capacity proxies
7. interaction features

Mandatory leakage rules:
- grouped stats inside fold only
- rolling features sorted by timestamp
- no future rows
- no random CV
- no train/test concatenation

Feature promotion rules:
- CV gain >= +0.008
- fold std increase <= +0.003
- leakage audit passes
- pseudo-private stable
- no single-fold miracle

Reject immediately if:
- unstable across folds
- suspicious CV jump
- public LB only improvement
- train/inference mismatch

Recommended models:
- CatBoost
- LightGBM
- XGBoost

Most important engineering rule:
small stable gains compound.

The winner is usually:
- disciplined experimentation
- strong validation
- leakage control
- robust feature engineering

NOT:
- exotic deep learning.
