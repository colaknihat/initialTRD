from __future__ import annotations

import argparse

import pandas as pd

from initial_trd.paths import (
    FEATURES_PATH,
    RAW_MARKET_PATH,
    WEIGHTED_FEATURES_PATH,
    resolve_project_path,
)
from initial_trd.training import engineer_turkish_features, generate_regime_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Turkish macro features from a raw market CSV."
    )
    parser.add_argument(
        "--input",
        default=RAW_MARKET_PATH,
        help="Raw input CSV path. Relative paths are resolved from the current working directory.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Engineered output CSV path. Defaults to artifacts/features.csv, "
            "or artifacts/features_weighted.csv with --with-regime-weights."
        ),
    )
    parser.add_argument(
        "--with-regime-weights",
        action="store_true",
        help="Also add HMM regime and sample_weight columns. Requires hmmlearn.",
    )
    parser.add_argument("--target-horizon", type=int, default=1)
    parser.add_argument("--n-components", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = resolve_project_path(args.input)
    if args.output:
        output_path = resolve_project_path(args.output)
    elif args.with_regime_weights:
        output_path = WEIGHTED_FEATURES_PATH
    else:
        output_path = FEATURES_PATH

    if not source_path.exists():
        raise FileNotFoundError(
            f"{source_path} does not exist. Run trd-fetch-data first "
            "or pass --input."
        )

    df = pd.read_csv(source_path)
    features = engineer_turkish_features(df, target_horizon=args.target_horizon)

    if args.with_regime_weights:
        features = generate_regime_weights(
            features,
            n_components=args.n_components,
            random_state=args.random_state,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    print(f"Wrote {len(features)} rows to {output_path}")


if __name__ == "__main__":
    main()
