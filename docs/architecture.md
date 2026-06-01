
# Architecture

Purpose:
Define the deterministic, leakage-safe, modular system architecture for the traffic demand prediction competition.

Core priorities:
1. Validation correctness
2. Reproducibility
3. Train/inference parity
4. Fast experimentation
5. Modular feature engineering

Project structure:

project_root/
├── configs/
├── data/
├── validation/
├── features/
├── models/
├── oof/
├── artifacts/
├── reports/
├── docs/
├── utils/
└── train.py

Rules:
- Single entrypoint only
- No notebook-only pipelines
- No hidden preprocessing
- Config-driven execution only
- All artifacts versioned

Pipeline flow:
config → schema validation → fold generation → feature generation → leakage audit → training → OOF → evaluation → artifact saving

Mandatory guarantees:
- deterministic reruns
- fixed seeds
- fold-safe grouped stats
- no future leakage
- exact train/inference parity

Required systems:
- feature registry
- experiment registry
- artifact hashing
- OOF storage
- runtime logging
- leakage auditing

Important:
Do not overengineer infra.
This is a competition system, not a distributed production platform.
