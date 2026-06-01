"""Weighted ensemble script to blend multiple experiment submissions."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

def main() -> None:
    # Define absolute paths based on project structure and experiment configs
    root = Path("/home/aashisoni/Codes/traffic-demand-prediction")
    
    # Define paths to submission CSVs based on experiment configurations
    sub_paths = {
        "exp07": root / "submissions/experiments/stochastic_catboost/submission.csv",
        "exp08": root / "submissions/experiments/seed99/submission.csv",
        "exp09": root / "submissions/experiments/multi_lag/submission.csv"
    }
    
    # Define weights for the ensemble
    weights = {
        "exp07": 0.2,
        "exp08": 0.2,
        "exp09": 0.6
    }
    
    output_path = root / "submissions/final_weighted_ensemble_submission.csv"

    # Check existence of source files
    for name, path in sub_paths.items():
        if not path.exists():
            print(f"Error: Submission file for {name} is missing at {path}")
            return

    # Load submissions
    dfs = {name: pd.read_csv(path) for name, path in sub_paths.items()}

    # Validation: Row counts must match across all input files
    row_counts = {name: len(df) for name, df in dfs.items()}
    if len(set(row_counts.values())) > 1:
        raise ValueError(f"Row count mismatch across submissions: {row_counts}")

    # Validation: Index columns must align perfectly to preserve row ordering
    first_name = list(dfs.keys())[0]
    base_index = dfs[first_name]["Index"]
    for name in list(dfs.keys())[1:]:
        if not (dfs[name]["Index"] == base_index).all():
            raise ValueError(f"Index alignment mismatch for {name}. Submissions must be in the same order.")

    # Generate final ensemble prediction using weighted averaging
    weighted_preds = pd.Series(0.0, index=base_index.index)
    for name, df in dfs.items():
        weighted_preds += df["demand"] * weights[name]

    # Validation: Ensure no NaN values were introduced during calculation
    if weighted_preds.isna().any():
        raise ValueError("NaN values detected in the calculated weighted ensemble predictions.")

    # Create final submission DataFrame
    ensemble_df = pd.DataFrame({
        "Index": base_index,
        "demand": weighted_preds
    })

    # Save the ensemble results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ensemble_df.to_csv(output_path, index=False)

    # Reporting
    print(f"\nWeighted ensemble generated successfully: {output_path.name}")
    print(f"Prediction Stats: Mean={weighted_preds.mean():.6f}, Min={weighted_preds.min():.6f}, Max={weighted_preds.max():.6f}")
    
    # Print pairwise prediction correlations to check model diversity
    print("\nPairwise Prediction Correlations:")
    names = list(dfs.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            corr = np.corrcoef(dfs[names[i]]["demand"], dfs[names[j]]["demand"])[0, 1]
            print(f"  {names[i]} vs {names[j]}: {corr:.6f}")
            
    print(f"\nTotal Rows: {len(ensemble_df)}")
    print(f"Output Path: {output_path}")

if __name__ == "__main__":
    main()