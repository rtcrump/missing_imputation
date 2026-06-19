"""Quick-start example for the missing_imputation package.

Generates a synthetic FACT-E dataset, holds out a fraction of observed values,
imputes with each classical method, and prints the held-out reconstruction
error (MAE / accuracy) per method.

Run with::

    python examples/quickstart.py
"""

import numpy as np

import missing_imputation as mi
from missing_imputation.columns import FACTE_COLUMNS


def hold_out_validation(df, cols, frac=0.2, seed=0):
    """Blank a random fraction of observed cells to score reconstruction."""
    rng = np.random.default_rng(seed)
    validation_df = df.copy()
    masks, originals = {}, {}
    for col in cols:
        observed = df[col].notna().to_numpy()
        mask = observed & (rng.random(len(df)) < frac)
        masks[col] = df[col].notna() & mask
        originals[col] = df[col].copy()
        validation_df.loc[mask, col] = np.nan
    return validation_df, masks, originals


def main():
    # 1. Make synthetic, PHI-free demo data.
    df = mi.make_synthetic_facte(n_patients=80, n_visits=5, missing_rate=0.2)
    cols = FACTE_COLUMNS
    print(f"Synthetic data: {len(df)} rows, {len(cols)} FACT-E items")
    print(f"Missing cells: {int(df[cols].isna().sum().sum())}\n")

    # 2. Hold out some observed values to measure imputation quality.
    validation_df, masks, originals = hold_out_validation(df, cols, frac=0.2)

    # 3. Run each method and report mean MAE / accuracy across items.
    print(f"{'method':<12}{'mean MAE':>10}{'mean acc':>10}")
    print("-" * 32)
    for name, method in mi.METHODS.items():
        _, results = method(validation_df, cols, validation_df, masks, originals)
        maes = [r["mae"] for r in results.values() if "mae" in r]
        accs = [r["accuracy"] for r in results.values() if "accuracy" in r]
        print(f"{name:<12}{np.mean(maes):>10.3f}{np.mean(accs):>10.3f}")


if __name__ == "__main__":
    main()
