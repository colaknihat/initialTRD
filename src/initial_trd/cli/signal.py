from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from initial_trd.paths import (
    FEATURES_PATH,
    MODEL_PATH,
    STOCK_A_PATH,
    STOCK_B_PATH,
    STRATEGY_SIGNAL_PATH,
    WEIGHTED_FEATURES_PATH,
    first_existing_path,
    resolve_project_path,
)
from initial_trd.strategy import execute_pairs_trade
from initial_trd.training import BISTResilientLSTM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a pair-trade instruction from close prices and a momentum prediction."
    )
    parser.add_argument(
        "--stock-a",
        default=STOCK_A_PATH,
        help="CSV for stock A. Must include close.",
    )
    parser.add_argument(
        "--stock-b",
        default=STOCK_B_PATH,
        help="CSV for stock B. Must include close.",
    )
    parser.add_argument("--regime", type=int, default=1, help="Current macro regime id.")
    parser.add_argument("--prediction", type=float, default=None, help="Predicted return.")
    parser.add_argument(
        "--model",
        default=MODEL_PATH,
        help="Optional .pt model produced by trd-train-lstm.",
    )
    parser.add_argument(
        "--features-input",
        default=None,
        help=(
            "Engineered CSV used for model prediction. Defaults to "
            "artifacts/features_weighted.csv if it exists, otherwise artifacts/features.csv."
        ),
    )
    parser.add_argument(
        "--features",
        default=None,
        help="Optional comma-separated feature columns. Defaults to model metadata.",
    )
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--entry-z", type=float, default=-2.0)
    parser.add_argument("--exit-z", type=float, default=0.5)
    parser.add_argument("--stock-a-name", default="stock_A")
    parser.add_argument("--stock-b-name", default="stock_B")
    parser.add_argument(
        "--output",
        default=STRATEGY_SIGNAL_PATH,
        help="JSON output path. Relative paths are resolved from the current working directory.",
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stock_a_path = resolve_project_path(args.stock_a)
    stock_b_path = resolve_project_path(args.stock_b)
    if not stock_a_path.exists() or not stock_b_path.exists():
        raise FileNotFoundError(
            f"{stock_a_path} or {stock_b_path} does not exist. "
            "Run trd-fetch-data first or pass --stock-a and --stock-b."
        )

    stock_a = pd.read_csv(stock_a_path)
    stock_b = pd.read_csv(stock_b_path)
    prediction = resolve_prediction(args)

    instruction = execute_pairs_trade(
        stock_a,
        stock_b,
        regime=args.regime,
        lstm_prediction=prediction,
        window=args.window,
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        stock_a_name=args.stock_a_name,
        stock_b_name=args.stock_b_name,
    )

    payload = asdict(instruction)
    payload["stock_a_name"] = args.stock_a_name
    payload["stock_b_name"] = args.stock_b_name
    payload["stock_a_path"] = str(stock_a_path)
    payload["stock_b_path"] = str(stock_b_path)
    payload["prediction_source"] = "argument" if args.prediction is not None else "model"
    output = json.dumps(payload, indent=2)
    print(output)

    output_path = resolve_project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output + "\n", encoding="utf-8")
    print(f"Wrote instruction to {output_path}")


def resolve_prediction(args: argparse.Namespace) -> float:
    if args.prediction is not None:
        return args.prediction

    model_path = resolve_project_path(args.model)
    features_path = (
        resolve_project_path(args.features_input)
        if args.features_input
        else first_existing_path(WEIGHTED_FEATURES_PATH, FEATURES_PATH)
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} does not exist. Provide --prediction or run "
            "trd-train-lstm first."
        )
    if not features_path.exists():
        raise FileNotFoundError(
            f"{features_path} does not exist. Run trd-engineer-features "
            "first or pass --features-input."
        )

    return predict_from_saved_lstm(
        model_path=model_path,
        features_path=features_path,
        feature_columns=_parse_optional_columns(args.features),
        device=args.device,
    )


def predict_from_saved_lstm(
    *,
    model_path: Path,
    features_path: Path,
    feature_columns: list[str] | None,
    device: str,
) -> float:
    device_obj = torch.device(device)
    checkpoint = load_checkpoint(model_path, device_obj)
    features = feature_columns or list(checkpoint["features"])
    sequence_length = int(checkpoint["sequence_length"])

    df = pd.read_csv(features_path)
    missing = [column for column in features if column not in df.columns]
    if missing:
        raise ValueError(f"features input is missing columns: {', '.join(missing)}")

    model = BISTResilientLSTM(
        input_dim=len(features),
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_layers=int(checkpoint["num_layers"]),
        dropout=float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device_obj)
    model.eval()

    values = (
        df.loc[:, features]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .tail(sequence_length)
        .to_numpy(dtype=np.float32)
    )
    if len(values) < sequence_length:
        raise ValueError("features input does not contain enough rows for prediction")

    x = torch.from_numpy(values.reshape(1, sequence_length, len(features))).to(device_obj)
    with torch.no_grad():
        prediction = model(x)

    return float(prediction.detach().cpu().numpy().reshape(-1)[0])


def load_checkpoint(model_path: Path, device: torch.device) -> dict:
    try:
        return torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(model_path, map_location=device)


def _parse_optional_columns(value: str | None) -> list[str] | None:
    if value is None:
        return None

    columns = [column.strip() for column in value.split(",") if column.strip()]
    if not columns:
        raise ValueError("features must include at least one column")
    return columns


if __name__ == "__main__":
    main()
