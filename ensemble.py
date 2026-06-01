"""Ensemble script to blend stochastic CatBoost experiment submissions."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

def main() -> None:
    # Define absolute paths based on project structure and experiment configs
    root = Path("/home/aashisoni/Codes/traffic-demand-prediction")
    
    # Exp 07 path: submissions/experiments/stochastic_catboost/exp07_stochastic_catboost_submission.csv
    sub_07_path = root / "submissions/experiments/stochastic_catboost/exp07_stochastic_catboost_submission.csv"
    
    # Exp 08 path: submissions/experiments/seed99/exp08_seed99_submission.csv
    sub_08_path = root / "submissions/experiments/seed99/exp08_seed99_submission.csv"
    
    output_path = root / "submissions/ensemble_exp07_exp08_submission.csv"

    # Load submissions
    if not sub_07_path.exists() or not sub_08_path.exists():
        print(f"Error: One or both submission files are missing.\nChecked:\n  {sub_07_path}\n  {sub_08_path}")
        return

    df_07 = pd.read_csv(sub_07_path)
    df_08 = pd.read_csv(sub_08_path)

    # Requirement: Row counts must match
    if len(df_07) != len(df_08):
        raise ValueError(f"Row count mismatch: exp07 has {len(df_07)} rows, exp08 has {len(df_08)} rows.")

    # Requirement: Preserve original test ordering (verified via Index column alignment)
    if not (df_07["Index"] == df_08["Index"]).all():
        raise ValueError("Index alignment mismatch: The submission files are not in the same order.")

    # Generate final ensemble prediction: 0.5 * exp07 + 0.5 * exp08
    ensemble_demand = 0.5 * df_07["demand"] + 0.5 * df_08["demand"]

    # Requirement: No NaNs
    if ensemble_demand.isna().any():
        raise ValueError("NaN values detected in the calculated ensemble predictions.")

    # Create final submission
    ensemble_df = pd.DataFrame({
        "Index": df_07["Index"],
        "demand": ensemble_demand
    })

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ensemble_df.to_csv(output_path, index=False)

    # Reporting
    correlation = np.corrcoef(df_07["demand"], df_08["demand"])[0, 1]
    
    print(f"\nEnsemble generated successfully: {output_path.name}")
    print(f"Prediction Stats: Mean={ensemble_demand.mean():.6f}, Min={ensemble_demand.min():.6f}, Max={ensemble_demand.max():.6f}")
    print(f"Model Correlation: {correlation:.6f}")
    print(f"Total Rows: {len(ensemble_df)}")
    print(f"Output Path: {output_path}")

if __name__ == "__main__":
    main()