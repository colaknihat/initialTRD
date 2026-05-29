from __future__ import annotations

import argparse
import json
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from initial_trd.paths import (
    FEATURES_PATH,
    MODEL_PATH,
    WEIGHTED_FEATURES_PATH,
    first_existing_path,
    resolve_project_path,
)
from initial_trd.training import (
    BISTResilientLSTM,
    calculate_split_safe_regime_weights,
    train_bist_model,
)


DEFAULT_FEATURES = (
    "bist_ret",
    "fx_ret",
    "real_rate",
    "cds_velocity",
    "fx_volatility",
    "market_breadth",
)
MAX_SEQUENCE_GAP = pd.Timedelta(days=5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the PyTorch attention LSTM on an engineered CSV."
    )
    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Engineered input CSV path. Defaults to artifacts/features.csv "
            "if it exists, otherwise artifacts/features_weighted.csv."
        ),
    )
    parser.add_argument(
        "--output",
        default=MODEL_PATH,
        help="Output .pt model path. Relative paths are resolved from the current working directory.",
    )
    parser.add_argument(
        "--features",
        default=",".join(DEFAULT_FEATURES),
        help="Comma-separated feature columns.",
    )
    parser.add_argument("--target", default="target", help="Target column.")
    parser.add_argument(
        "--weight-column",
        default="sample_weight",
        help="Sample weight column. Uses 1.0 if absent.",
    )
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default=None, help="Example: cpu or cuda.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    input_path = (
        resolve_project_path(args.input)
        if args.input
        else first_existing_path(FEATURES_PATH, WEIGHTED_FEATURES_PATH)
    )
    output_path = resolve_project_path(args.output)
    feature_columns = _parse_columns(args.features)

    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} does not exist. Run trd-engineer-features "
            "first or pass --input."
        )

    df = pd.read_csv(input_path)
    x, y, weights = build_sequence_arrays(
        df,
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
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
        },
        output_path,
    )

    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    metadata_path.write_text(
        json.dumps(
            {
                "features": feature_columns,
                "target": args.target,
                "sequence_length": args.sequence_length,
                "rows_used": int(len(x)),
                "epochs": args.epochs,
                "final_metrics": history[-1],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved model to {output_path}")
    print(f"Saved metadata to {metadata_path}")


def build_sequence_arrays(
    df: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    weight_column: str,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1")

    required_columns = feature_columns + [target_column]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"input is missing required columns: {', '.join(missing)}")

    working = df.copy()
    if weight_column not in working.columns:
        working[weight_column] = 1.0

    columns = feature_columns + [target_column, weight_column]
    has_date_column = "date" in working.columns
    if has_date_column and "date" not in columns:
        columns.append("date")
    working = working.loc[:, columns].replace([np.inf, -np.inf], np.nan).dropna()
    if len(working) <= sequence_length:
        raise ValueError("not enough rows to build sequences")

    x_rows = []
    y_rows = []
    weight_rows = []
    for segment in _contiguous_segments(working, has_date_column=has_date_column):
        if len(segment) <= sequence_length:
            continue

        feature_values = segment.loc[:, feature_columns].to_numpy(dtype=np.float32)
        targets = segment.loc[:, target_column].to_numpy(dtype=np.float32)
        weights = segment.loc[:, weight_column].to_numpy(dtype=np.float32)

        for end in range(sequence_length - 1, len(segment)):
            start = end - sequence_length + 1
            x_rows.append(feature_values[start : end + 1])
            y_rows.append(targets[end])
            weight_rows.append(weights[end])

    if not x_rows:
        raise ValueError("not enough contiguous rows to build sequences")

    return (
        np.asarray(x_rows, dtype=np.float32),
        np.asarray(y_rows, dtype=np.float32).reshape(-1, 1),
        np.asarray(weight_rows, dtype=np.float32),
    )


def _contiguous_segments(
    working: pd.DataFrame,
    *,
    has_date_column: bool,
) -> list[pd.DataFrame]:
    if not has_date_column:
        return [working]

    dated = working.copy()
    dated["date"] = pd.to_datetime(dated["date"])
    dated = dated.sort_values("date")
    gap_starts = dated["date"].diff() > MAX_SEQUENCE_GAP
    return [segment for _, segment in dated.groupby(gap_starts.cumsum(), sort=False)]


def build_loaders(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    validation_size: float,
    batch_size: int,
    feature_columns: Sequence[str] | None = None,
    use_regime_weights: bool = False,
    regime_model: Any = None,
    hmm_random_state: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    if not 0.0 < validation_size < 1.0:
        raise ValueError("validation_size must be between 0 and 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    split_index = int(len(x) * (1.0 - validation_size))
    if split_index <= 0 or split_index >= len(x):
        raise ValueError("validation_size leaves an empty train or validation split")

    effective_weights = np.asarray(weights, dtype=np.float32)
    if use_regime_weights:
        effective_weights = _split_safe_regime_weights(
            x,
            split_index=split_index,
            feature_columns=feature_columns,
            model=regime_model,
            random_state=hmm_random_state,
        )

    train_dataset = TensorDataset(
        torch.from_numpy(x[:split_index]),
        torch.from_numpy(y[:split_index]),
        torch.from_numpy(effective_weights[:split_index]),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(x[split_index:]),
        torch.from_numpy(y[split_index:]),
        torch.from_numpy(effective_weights[split_index:]),
    )

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
    )


def _split_safe_regime_weights(
    x: np.ndarray,
    *,
    split_index: int,
    feature_columns: Sequence[str] | None,
    model: Any,
    random_state: int | None,
) -> np.ndarray:
    if feature_columns is None:
        raise ValueError("feature_columns are required for split-safe regime weights")

    columns = list(feature_columns)
    try:
        bist_index = columns.index("bist_ret")
        volatility_index = columns.index("fx_volatility")
    except ValueError as exc:
        raise ValueError(
            "split-safe regime weights require bist_ret and fx_volatility features"
        ) from exc

    regime_values = x[:, -1, [bist_index, volatility_index]]
    _, sample_weights = calculate_split_safe_regime_weights(
        regime_values[:split_index],
        regime_values,
        model=model,
        random_state=random_state,
    )
    return np.asarray(sample_weights, dtype=np.float32)


def _parse_columns(value: str) -> list[str]:
    columns = [column.strip() for column in value.split(",") if column.strip()]
    if not columns:
        raise ValueError("at least one feature column is required")
    return columns


if __name__ == "__main__":
    main()
