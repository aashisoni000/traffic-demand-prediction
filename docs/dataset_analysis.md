
# Dataset Analysis

Competition dataset:
Spatiotemporal tabular traffic demand prediction.

Dataset focus:
- feature engineering
- validation quality
- grouped aggregations
- temporal intelligence

Mandatory audit sequence:
1. schema audit
2. null audit
3. duplicate audit
4. timestamp audit
5. target audit
6. geohash audit
7. regime discovery
8. drift analysis
9. leakage analysis

Critical analyses:

Temporal:
- rush hours
- weekends
- weather shifts
- night patterns
- timestamp continuity

Geospatial:
- geohash density
- unstable regions
- sparse geohashes
- hotspot discovery

Capacity:
- lane bottlenecks
- LargeVehicles friction
- congestion pressure

Required visualizations:
- hourly demand
- weekday demand
- rolling volatility
- geohash heatmaps
- weather interactions
- correlation matrix

Mandatory questions:
- where does demand spike?
- which geohashes unstable?
- what regimes hardest?
- where drift exists?
- what leakage risks exist?

Required outputs:
- schema report
- drift report
- regime report
- feature hypotheses
- leakage audit report

Important:
Dataset analysis is NOT generic EDA.
It is:
- leakage investigation
- validation planning
- feature discovery
- leaderboard robustness analysis.
