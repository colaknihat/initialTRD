from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_testing import run_walk_forward_test
from project_paths import (
    FEATURES_PATH,
    WALK_FORWARD_RESULTS_PATH,
    WEIGHTED_FEATURES_PATH,
    first_existing_path,
    resolve_project_path,
)


DEFAULT_FEATURES = (
    "bist_ret",
    "fx_ret",
    "real_rate",
    "cds_velocity",
    "fx_volatility",
    "market_breadth",
)


class MeanReturnModel:
    def fit(self, x: np.ndarray, y: np.ndarray):
        self.prediction = float(np.mean(y))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(len(x), self.prediction)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run purged walk-forward validation on an engineered CSV."
    )
    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Engineered input CSV path. Defaults to artifacts/features.csv if it "
            "exists, otherwise artifacts/features_weighted.csv."
        ),
    )
    parser.add_argument(
        "--output",
        default=WALK_FORWARD_RESULTS_PATH,
        help="Results CSV path. Relative paths are resolved from the project root.",
    )
    parser.add_argument(
        "--features",
        default=",".join(DEFAULT_FEATURES),
        help="Comma-separated feature columns.",
    )
    parser.add_argument("--target", default="target", help="Target column.")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--embargo-days", type=int, default=15)
    parser.add_argument(
        "--model",
        choices=("mean", "linear", "ridge", "random-forest"),
        default="mean",
    )
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = (
        resolve_project_path(args.input)
        if args.input
        else first_existing_path(FEATURES_PATH, WEIGHTED_FEATURES_PATH)
    )
    feature_columns = _parse_columns(args.features)

    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} does not exist. Run scripts/run_feature_engineering.py "
            "first or pass --input."
        )

    df = pd.read_csv(input_path)
    results = run_walk_forward_test(
        df,
        model_factory=build_model_factory(args),
        features=feature_columns,
        target=args.target,
        n_splits=args.n_splits,
        embargo_days=args.embargo_days,
    )

    print(results.to_string(index=False))
    print()
    print("Average metrics:")
    print(results[["test_sharpe", "test_max_dd", "rmse", "directional_accuracy"]].mean())

    output_path = resolve_project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"Wrote results to {output_path}")


def build_model_factory(args: argparse.Namespace):
    if args.model == "mean":
        return MeanReturnModel

    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression, Ridge
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for linear, ridge, and random-forest models"
        ) from exc

    if args.model == "linear":
        return LinearRegression
    if args.model == "ridge":
        return lambda: Ridge(alpha=args.ridge_alpha)
    if args.model == "random-forest":
        return lambda: RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=5,
            random_state=args.random_state,
        )

    raise ValueError(f"unsupported model: {args.model}")


def _parse_columns(value: str) -> list[str]:
    columns = [column.strip() for column in value.split(",") if column.strip()]
    if not columns:
        raise ValueError("at least one feature column is required")
    return columns


if __name__ == "__main__":
    main()
