from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import pandas as pd
import torch

from initial_trd.cli.train_lstm import (
    DEFAULT_FEATURES,
    _parse_columns,
    build_loaders,
    build_sequence_arrays,
)
from initial_trd.paths import (
    EPOCH_TUNING_PATH,
    EPOCH_TUNING_SUMMARY_PATH,
    FEATURES_PATH,
    WEIGHTED_FEATURES_PATH,
    first_existing_path,
    resolve_project_path,
)
from initial_trd.training import BISTResilientLSTM, train_bist_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find the best LSTM epoch count from validation loss."
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
        default=EPOCH_TUNING_PATH,
        help="Epoch history CSV path. Relative paths are resolved from the current working directory.",
    )
    parser.add_argument(
        "--summary-output",
        default=EPOCH_TUNING_SUMMARY_PATH,
        help="Best-epoch summary JSON path. Relative paths are resolved from the current working directory.",
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
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Stop after this many epochs without validation improvement. Use 0 to disable.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="Minimum validation-loss improvement needed to reset patience.",
    )
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
    if args.max_epochs < 1:
        raise ValueError("max-epochs must be at least 1")
    if args.patience < 0:
        raise ValueError("patience cannot be negative")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    input_path = (
        resolve_project_path(args.input)
        if args.input
        else first_existing_path(FEATURES_PATH, WEIGHTED_FEATURES_PATH)
    )
    output_path = resolve_project_path(args.output)
    summary_output_path = resolve_project_path(args.summary_output)
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
    patience = None if args.patience == 0 else args.patience
    history = train_bist_model(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.max_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        early_stopping_patience=patience,
        min_delta=args.min_delta,
    )

    history_df = pd.DataFrame(history)
    summary = summarize_history(history, requested_max_epochs=args.max_epochs)
    history_df["is_best_epoch"] = history_df["epoch"] == summary["best_epoch"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(output_path, index=False)

    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Best epoch: {summary['best_epoch']}")
    print(f"Best validation loss: {summary['best_val_loss']:.8f}")
    print(f"Final epoch evaluated: {summary['final_epoch']}")
    print(f"Wrote epoch history to {output_path}")
    print(f"Wrote summary to {summary_output_path}")
    print()
    print(
        "Suggested training command: "
        f"trd-train-lstm --epochs {summary['best_epoch']}"
        + (f" --device {args.device}" if args.device else "")
    )


def summarize_history(
    history: list[dict[str, float]],
    *,
    requested_max_epochs: int,
) -> dict[str, Any]:
    if not history:
        raise ValueError("history cannot be empty")

    best = min(history, key=lambda row: row["val_loss"])
    final = history[-1]
    best_epoch = int(best["epoch"])
    final_epoch = int(final["epoch"])

    return {
        "best_epoch": best_epoch,
        "best_val_loss": float(best["val_loss"]),
        "best_train_loss": float(best["train_loss"]),
        "final_epoch": final_epoch,
        "final_val_loss": float(final["val_loss"]),
        "final_train_loss": float(final["train_loss"]),
        "epochs_after_best": final_epoch - best_epoch,
        "stopped_early": final_epoch < requested_max_epochs,
    }


if __name__ == "__main__":
    main()
