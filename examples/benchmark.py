"""Benchmark imputation methods with the evaluation harness.

Runs all four evaluators in ``missing_imputation.evaluation`` on a synthetic
FACT-E dataset and prints a leaderboard for each. Replace ``make_synthetic_facte``
with your own long-format DataFrame (columns ``id``, ``redcap_event_name``, and
the FACT-E items) to benchmark on real data.

Run with::

    python examples/benchmark.py
"""

import missing_imputation as mi
from missing_imputation.columns import FACTE_COLUMNS

# Use the fast, dependency-light methods so the example runs in seconds.
FAST_METHODS = {k: mi.METHODS[k] for k in ("mean", "median", "knn", "softimpute")}


def main():
    df = mi.make_synthetic_facte(n_patients=120, n_visits=6, missing_rate=0.25)
    cols = FACTE_COLUMNS
    print(f"Synthetic data: {len(df)} rows, {len(cols)} FACT-E items\n")

    # 1. Cross-validated reconstruction error + ordinal metrics.
    print("== Sparse cross-validation ==")
    sparse = mi.evaluate_with_sparse_validation(df, cols, methods=FAST_METHODS, n_folds=5)
    print(f"{'method':<12}{'MAE':>8}{'RMSE':>8}{'acc':>8}{'QWK':>8}{'within1':>9}")
    print("-" * 53)
    for name, r in sorted(sparse.items(), key=lambda kv: kv[1]["avg_mae"]):
        print(f"{name:<12}{r['avg_mae']:>8.3f}{r['avg_rmse']:>8.3f}"
              f"{r['avg_accuracy']:>8.3f}{r['avg_qwk']:>8.3f}"
              f"{r['avg_within1_accuracy']:>9.3f}")

    # 2. Trajectory fidelity (per-patient middle-visit recovery).
    print("\n== Trajectory fidelity ==")
    traj = mi.evaluate_trajectory_fidelity(df, cols, methods=FAST_METHODS)
    print(f"{'method':<12}{'traj MAE':>10}{'corr':>8}{'n':>6}")
    print("-" * 36)
    for name, r in sorted(traj.items(), key=lambda kv: kv[1]["trajectory_mae"]):
        print(f"{name:<12}{r['trajectory_mae']:>10.3f}"
              f"{r['trajectory_correlation']:>8.3f}{r['n_patients']:>6}")

    # 3. Temporal smoothness of the imputed trajectories.
    print("\n== Temporal smoothness (lower = smoother) ==")
    imputed = mi.run_all_methods(df, cols, methods=FAST_METHODS)
    smooth = mi.evaluate_temporal_smoothness(df, imputed, cols)
    print(f"{'method':<12}{'smoothness':>12}{'n':>6}")
    print("-" * 30)
    for name, r in sorted(smooth.items(), key=lambda kv: kv[1]["mean_smoothness"]):
        print(f"{name:<12}{r['mean_smoothness']:>12.3f}{r['n_patients']:>6}")

    # 4. Robustness across missingness patterns.
    print("\n== Missing-pattern robustness (MAE by pattern) ==")
    robust = mi.evaluate_missing_pattern_robustness(df, cols, methods=FAST_METHODS)
    for pattern, group in robust.items():
        if not group:
            print(f"  {pattern}: (no patients in this group)")
            continue
        ranked = ", ".join(f"{m}={v['mae']:.3f}" for m, v in sorted(group.items()))
        print(f"  {pattern}: {ranked}")


if __name__ == "__main__":
    main()
