from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from initial_trd.backtesting import (
    resolve_device,
    select_common_trading_dates,
    simulate_pair_backtest,
    train_lstm_predictions_by_date,
)
from initial_trd.cli.train_lstm import DEFAULT_FEATURES, _parse_columns
from initial_trd.data_fetch import (
    DEFAULT_BREADTH_TICKERS,
    END_DATE,
    START_DATE,
    fetch_and_align_data,
    fetch_stock_closes,
    _parse_tickers,
)
from initial_trd.paths import ARTIFACTS_DIR, RAW_MARKET_PATH, resolve_project_path
from initial_trd.training import engineer_turkish_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest symmetric pair trades over the latest trading intervals."
    )
    parser.add_argument(
        "--tickers",
        default=",".join(DEFAULT_BREADTH_TICKERS),
        help="Comma-separated stock tickers. Defaults to the repo's 10 BIST tickers.",
    )
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=END_DATE)
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
        help="Seed for HMM regime weighting.",
    )
    parser.add_argument(
        "--no-regime-weights",
        action="store_true",
        help="Train without HMM sample weights.",
    )
    parser.add_argument(
        "--features",
        default=",".join(DEFAULT_FEATURES),
        help="Comma-separated feature columns for the LSTM.",
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
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda.")
    parser.add_argument("--regime", type=int, default=1)
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument(
        "--entry-z",
        type=float,
        default=2.0,
        help="Absolute entry z-score threshold.",
    )
    parser.add_argument("--exit-z", type=float, default=0.5)
    parser.add_argument(
        "--output",
        default=None,
        help="Pair-level CSV output path. Defaults to artifacts/pair_backtest_<days>d.csv.",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Summary JSON output path. Defaults to artifacts/pair_backtest_<days>d_summary.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tickers = _parse_tickers(args.tickers)
    if len(tickers) < 2:
        raise ValueError("at least two tickers are required")

    device = resolve_device(args.device)
    feature_columns = _parse_columns(args.features)
    use_regime_weights = not args.no_regime_weights

    print("Backtest configuration")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Pairs: {len(tickers) * (len(tickers) - 1) // 2}")
    print(f"Days: {args.days}")
    print(f"Device: {device}")
    print(f"Epochs per daily retrain: {args.epochs}")
    print(f"Feature regime weights: {'yes' if use_regime_weights else 'no'}")
    print()

    print("Step 1/4: Fetching market data")
    fetch_and_align_data(
        stock_a_ticker=tickers[0],
        stock_b_ticker=tickers[1],
        breadth_tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        random_state=args.fetch_random_state,
    )
    stock_closes = fetch_stock_closes(
        tickers,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    print()
    print("Step 2/4: Engineering features")
    raw_market_path = resolve_project_path(RAW_MARKET_PATH)
    raw_df = pd.read_csv(raw_market_path)
    features_df = engineer_turkish_features(raw_df)
    print(f"Feature rows: {len(features_df)}")

    print()
    print("Step 3/4: Daily LSTM retraining")
    test_dates = select_common_trading_dates(stock_closes, tickers, days=args.days)
    predictions = train_lstm_predictions_by_date(
        features_df,
        test_dates[:-1],
        feature_columns=feature_columns,
        target_column=args.target,
        weight_column=args.weight_column,
        sequence_length=args.sequence_length,
        validation_size=args.validation_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=device,
        use_regime_weights=use_regime_weights,
        hmm_random_state=args.hmm_random_state,
    )

    print()
    print("Step 4/4: Simulating pair trades")
    result = simulate_pair_backtest(
        stock_closes,
        predictions,
        tickers=tickers,
        days=args.days,
        initial_capital=100.0,
        regime=args.regime,
        window=args.window,
        entry_z=args.entry_z,
        exit_z=args.exit_z,
    )

    output_path = resolve_project_path(args.output or _default_results_path(args.days))
    summary_path = resolve_project_path(
        args.summary_output or _default_summary_path(args.days)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    result.rows.to_csv(output_path, index=False)
    summary_path.write_text(
        json.dumps(result.summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print_backtest_summary(result.summary)
    print(f"Wrote pair-level results to {output_path}")
    print(f"Wrote summary to {summary_path}")


def print_backtest_summary(summary: dict) -> None:
    print()
    print("Backtest summary")
    print(f"Window: {summary['start_date']} to {summary['end_date']}")
    print(f"Pairs tested: {summary['pairs_tested']}")
    print(f"Trades opened: {summary['trades_opened']}")
    print(f"Trades closed: {summary['trades_closed']}")
    print(f"Active pair-days: {summary['active_pair_days']}")
    print()
    for name, values in summary["account_values"].items():
        ending = values["ending_value"]
        pnl = values["total_pnl"]
        return_pct = values["return_pct"] * 100.0
        print(f"{name}: ending=${ending:.2f}, pnl=${pnl:.2f}, return={return_pct:.2f}%")


def _default_results_path(days: int) -> Path:
    return ARTIFACTS_DIR / f"pair_backtest_{days}d.csv"


def _default_summary_path(days: int) -> Path:
    return ARTIFACTS_DIR / f"pair_backtest_{days}d_summary.json"


if __name__ == "__main__":
    main()
