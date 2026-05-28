from __future__ import annotations

import argparse
from dataclasses import asdict
import json

import numpy as np
import pandas as pd
import torch

from initial_trd.cli.train_lstm import (
    DEFAULT_FEATURES,
    _parse_columns,
    build_loaders,
    build_sequence_arrays,
)
from initial_trd.cli.signal import predict_from_saved_lstm
from initial_trd.cli.walk_forward import build_model_factory
from initial_trd.data_fetch import (
    DEFAULT_STOCK_A_TICKER,
    DEFAULT_STOCK_B_TICKER,
    fetch_and_align_data,
)
from initial_trd.evaluation import run_walk_forward_test
from initial_trd.paths import (
    FEATURES_PATH,
    MODEL_PATH,
    PIPELINE_SUMMARY_PATH,
    RAW_MARKET_PATH,
    STOCK_A_PATH,
    STOCK_B_PATH,
    STRATEGY_SIGNAL_PATH,
    WALK_FORWARD_RESULTS_PATH,
    WEIGHTED_FEATURES_PATH,
    resolve_project_path,
)
from initial_trd.strategy import execute_pairs_trade
from initial_trd.training import (
    BISTResilientLSTM,
    engineer_turkish_features,
    generate_regime_weights,
    train_bist_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full Initial TRD research pipeline."
    )
    parser.add_argument("--stock-a-ticker", default=DEFAULT_STOCK_A_TICKER)
    parser.add_argument("--stock-b-ticker", default=DEFAULT_STOCK_B_TICKER)
    parser.add_argument(
        "--stock-a-name",
        default=None,
        help="Display name in signal output. Defaults to --stock-a-ticker.",
    )
    parser.add_argument(
        "--stock-b-name",
        default=None,
        help="Display name in signal output. Defaults to --stock-b-ticker.",
    )
    parser.add_argument(
        "--fetch-random-state",
        type=int,
        default=7,
        help="Seed for generated macro proxy data.",
    )
    parser.add_argument(
        "--hmm-random-state",
        type=int,
        default=7,
        help="Seed for HMM regime weighting. This is for reproducibility, not a guaranteed optimum.",
    )
    parser.add_argument(
        "--no-regime-weights",
        action="store_true",
        help="Write artifacts/features.csv instead of weighted features.",
    )
    parser.add_argument(
        "--features",
        default=",".join(DEFAULT_FEATURES),
        help="Comma-separated feature columns for LSTM and walk-forward tests.",
    )
    parser.add_argument("--target", default="target")
    parser.add_argument("--weight-column", default="sample_weight")
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=66)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--walk-forward-model",
        choices=("mean", "linear", "ridge", "random-forest"),
        default="ridge",
    )
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--embargo-days", type=int, default=15)
    parser.add_argument("--regime", type=int, default=1)
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--entry-z", type=float, default=-2.0)
    parser.add_argument("--exit-z", type=float, default=0.5)
    parser.add_argument(
        "--summary-output",
        default=PIPELINE_SUMMARY_PATH,
        help="Pipeline summary JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    stock_a_name = args.stock_a_name or args.stock_a_ticker
    stock_b_name = args.stock_b_name or args.stock_b_ticker
    feature_columns = _parse_columns(args.features)
    use_regime_weights = not args.no_regime_weights

    print("Pipeline configuration")
    print(f"Pair: {stock_a_name} ({args.stock_a_ticker}) vs {stock_b_name} ({args.stock_b_ticker})")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Feature regime weights: {'yes' if use_regime_weights else 'no'}")
    print(f"HMM random state: {args.hmm_random_state}")
    print(f"Walk-forward model: {args.walk_forward_model}")
    print()

    print("Step 1/5: Fetching data")
    fetch_and_align_data(
        stock_a_ticker=args.stock_a_ticker,
        stock_b_ticker=args.stock_b_ticker,
        random_state=args.fetch_random_state,
    )

    print()
    print("Step 2/5: Engineering features")
    raw_market_path = resolve_project_path(RAW_MARKET_PATH)
    features_path = resolve_project_path(
        WEIGHTED_FEATURES_PATH if use_regime_weights else FEATURES_PATH
    )
    raw_df = pd.read_csv(raw_market_path)
    features_df = engineer_turkish_features(raw_df)
    if use_regime_weights:
        features_df = generate_regime_weights(
            features_df,
            random_state=args.hmm_random_state,
        )
    features_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(features_path, index=False)
    print(f"Wrote {len(features_df)} rows to {features_path}")

    print()
    print("Step 3/5: Training LSTM")
    model_path = resolve_project_path(MODEL_PATH)
    x, y, weights = build_sequence_arrays(
        features_df,
        feature_columns=feature_columns,
        target_column=args.target,
        weight_column=args.weight_column,
        sequence_length=args.sequence_length,
    )
    train_loader, val_loader = build_loaders(
        x,
        y,
        weights,
        validation_size=args.validation_size,
        batch_size=args.batch_size,
    )
    model = BISTResilientLSTM(
        input_dim=len(feature_columns),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    history = train_bist_model(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "features": feature_columns,
            "target": args.target,
            "sequence_length": args.sequence_length,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "history": history,
            "stock_a_ticker": args.stock_a_ticker,
            "stock_b_ticker": args.stock_b_ticker,
        },
        model_path,
    )
    model_metadata_path = model_path.with_suffix(model_path.suffix + ".json")
    model_metadata_path.write_text(
        json.dumps(
            {
                "features": feature_columns,
                "target": args.target,
                "sequence_length": args.sequence_length,
                "rows_used": int(len(x)),
                "epochs": args.epochs,
                "final_metrics": history[-1],
                "stock_a_ticker": args.stock_a_ticker,
                "stock_b_ticker": args.stock_b_ticker,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved model to {model_path}")

    print()
    print("Step 4/5: Running walk-forward validation")
    walk_forward_args = argparse.Namespace(
        model=args.walk_forward_model,
        ridge_alpha=args.ridge_alpha,
        random_state=args.seed,
    )
    walk_forward_results = run_walk_forward_test(
        features_df,
        model_factory=build_model_factory(walk_forward_args),
        features=feature_columns,
        target=args.target,
        n_splits=args.n_splits,
        embargo_days=args.embargo_days,
    )
    walk_forward_path = resolve_project_path(WALK_FORWARD_RESULTS_PATH)
    walk_forward_path.parent.mkdir(parents=True, exist_ok=True)
    walk_forward_results.to_csv(walk_forward_path, index=False)
    print(walk_forward_results.to_string(index=False))
    print(f"Wrote results to {walk_forward_path}")

    print()
    print("Step 5/5: Generating signal")
    stock_a_path = resolve_project_path(STOCK_A_PATH)
    stock_b_path = resolve_project_path(STOCK_B_PATH)
    prediction = predict_from_saved_lstm(
        model_path=model_path,
        features_path=features_path,
        feature_columns=None,
        device=args.device,
    )
    instruction = execute_pairs_trade(
        pd.read_csv(stock_a_path),
        pd.read_csv(stock_b_path),
        regime=args.regime,
        lstm_prediction=prediction,
        window=args.window,
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        stock_a_name=stock_a_name,
        stock_b_name=stock_b_name,
    )
    signal_payload = asdict(instruction)
    signal_payload["stock_a_name"] = stock_a_name
    signal_payload["stock_b_name"] = stock_b_name
    signal_payload["stock_a_ticker"] = args.stock_a_ticker
    signal_payload["stock_b_ticker"] = args.stock_b_ticker
    signal_payload["stock_a_path"] = str(stock_a_path)
    signal_payload["stock_b_path"] = str(stock_b_path)
    signal_payload["prediction_source"] = "model"
    signal_path = resolve_project_path(STRATEGY_SIGNAL_PATH)
    signal_path.parent.mkdir(parents=True, exist_ok=True)
    signal_path.write_text(json.dumps(signal_payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(signal_payload, indent=2))
    print(f"Wrote instruction to {signal_path}")

    summary_path = resolve_project_path(args.summary_output)
    summary = {
        "stock_a_ticker": args.stock_a_ticker,
        "stock_b_ticker": args.stock_b_ticker,
        "stock_a_name": stock_a_name,
        "stock_b_name": stock_b_name,
        "epochs": args.epochs,
        "device": args.device,
        "hmm_random_state": args.hmm_random_state,
        "features_path": str(features_path),
        "model_path": str(model_path),
        "walk_forward_results_path": str(walk_forward_path),
        "signal_path": str(signal_path),
        "signal_action": instruction.action,
        "predicted_return": float(prediction),
        "walk_forward_average_metrics": {
            column: float(walk_forward_results[column].mean())
            for column in (
                "test_sharpe",
                "test_max_dd",
                "rmse",
                "directional_accuracy",
                "mean_strategy_return",
            )
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote pipeline summary to {summary_path}")


if __name__ == "__main__":
    main()
