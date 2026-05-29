from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from initial_trd.backtesting import (
    DEFAULT_COINTEGRATION_PVALUE_THRESHOLD,
    DEFAULT_MIN_COINTEGRATION_OBSERVATIONS,
    DEFAULT_SLIPPAGE_PER_LEG,
    DEFAULT_TRANSACTION_COST_PER_LEG,
    classify_backtest_regimes_by_date,
    resolve_device,
    select_common_trading_dates,
    simulate_pair_backtest,
    train_lstm_predictions_by_date,
)
from initial_trd.cli.train_lstm import DEFAULT_FEATURES, _parse_columns
from initial_trd.data_fetch import (
    DEFAULT_CBRT_RATE_URL,
    DEFAULT_BREADTH_TICKERS,
    DEFAULT_CDS_CSV_PATH,
    DEFAULT_TURKSTAT_CPI_URL,
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
        description="Backtest hedge-adjusted pair trades over the latest trading intervals."
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
        "--cpi-url",
        default=DEFAULT_TURKSTAT_CPI_URL,
        help="TCMB/TURKSTAT CPI page URL.",
    )
    parser.add_argument(
        "--cbrt-rate-url",
        default=DEFAULT_CBRT_RATE_URL,
        help="TCMB policy-rate page URL or FRED-compatible CSV URL.",
    )
    parser.add_argument(
        "--cds-csv",
        default=str(DEFAULT_CDS_CSV_PATH),
        help="Turkey 5Y CDS CSV from Bloomberg, Refinitiv, or Investing.com.",
    )
    parser.add_argument(
        "--hmm-random-state",
        type=int,
        default=7,
        help="Seed for HMM regime weighting and backtest regime classification.",
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
    parser.add_argument("--target-horizon", type=int, default=2)
    parser.add_argument("--weight-column", default="sample_weight")
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda.")
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument(
        "--entry-z",
        type=float,
        default=2.0,
        help="Absolute entry z-score threshold.",
    )
    parser.add_argument("--exit-z", type=float, default=0.5)
    parser.add_argument(
        "--transaction-cost-per-leg",
        type=float,
        default=DEFAULT_TRANSACTION_COST_PER_LEG,
        help=(
            "Per-leg tax/commission cost as a decimal. Defaults to 0.0010 "
            "before slippage."
        ),
    )
    parser.add_argument(
        "--slippage-per-leg",
        type=float,
        default=DEFAULT_SLIPPAGE_PER_LEG,
        help="Per-leg bid/ask or execution slippage as a decimal.",
    )
    parser.add_argument(
        "--shortable-tickers",
        default="",
        help=(
            "Comma-separated tickers with confirmed borrow availability. "
            "When omitted, short entries are blocked unless --allow-unlisted-shorts is set."
        ),
    )
    parser.add_argument(
        "--allow-unlisted-shorts",
        action="store_true",
        help="Research-only mode: allow shorts even when a ticker is not in --shortable-tickers.",
    )
    parser.add_argument(
        "--disable-cointegration-filter",
        action="store_true",
        help="Research-only mode: allow opens without passing the Engle-Granger filter.",
    )
    parser.add_argument(
        "--cointegration-pvalue-threshold",
        type=float,
        default=DEFAULT_COINTEGRATION_PVALUE_THRESHOLD,
        help="Maximum Engle-Granger residual unit-root p-value proxy for opening a pair.",
    )
    parser.add_argument(
        "--min-cointegration-observations",
        type=int,
        default=DEFAULT_MIN_COINTEGRATION_OBSERVATIONS,
        help="Minimum historical observations required before a pair can open.",
    )
    parser.add_argument(
        "--cointegration-lookback",
        type=int,
        default=None,
        help="Optional trailing observation count for hedge-ratio and cointegration estimates.",
    )
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
    shortable_tickers = (
        _parse_tickers(args.shortable_tickers)
        if args.shortable_tickers.strip()
        else ()
    )

    device = resolve_device(args.device)
    feature_columns = _parse_columns(args.features)
    use_regime_weights = not args.no_regime_weights

    print("Backtest configuration")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Pairs: {len(tickers) * (len(tickers) - 1) // 2}")
    print(f"Days: {args.days}")
    print(f"Device: {device}")
    print(f"Epochs per daily retrain: {args.epochs}")
    print(f"Target horizon: {args.target_horizon}")
    print(f"Feature regime weights: {'yes' if use_regime_weights else 'no'}")
    print(
        "All-in cost per leg: "
        f"{(args.transaction_cost_per_leg + args.slippage_per_leg) * 100.0:.3f}%"
    )
    print(
        "Short availability: "
        + (
            ", ".join(shortable_tickers)
            if shortable_tickers
            else "none supplied; short entries blocked"
            if not args.allow_unlisted_shorts
            else "not enforced"
        )
    )
    cointegration_text = (
        "disabled"
        if args.disable_cointegration_filter
        else (
            f"p<={args.cointegration_pvalue_threshold}, "
            f"min_obs={args.min_cointegration_observations}"
        )
    )
    print(f"Cointegration filter: {cointegration_text}")
    print()

    print("Step 1/5: Fetching market data")
    fetch_and_align_data(
        stock_a_ticker=tickers[0],
        stock_b_ticker=tickers[1],
        breadth_tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        cpi_url=args.cpi_url,
        cbrt_rate_url=args.cbrt_rate_url,
        cds_csv=args.cds_csv,
    )
    stock_closes = fetch_stock_closes(
        tickers,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    print()
    print("Step 2/5: Engineering features")
    raw_market_path = resolve_project_path(RAW_MARKET_PATH)
    raw_df = pd.read_csv(raw_market_path)
    features_df = engineer_turkish_features(
        raw_df,
        target_horizon=args.target_horizon,
    )
    print(f"Feature rows: {len(features_df)}")

    print()
    print("Step 3/5: Classifying daily HMM regimes")
    test_dates = select_common_trading_dates(stock_closes, tickers, days=args.days + 1)
    signal_dates = test_dates[:-2]
    regimes = classify_backtest_regimes_by_date(
        features_df,
        signal_dates,
        random_state=args.hmm_random_state,
    )
    print(f"Regime classifications: {len(regimes)}")

    print()
    print("Step 4/5: Daily LSTM retraining")
    predictions = train_lstm_predictions_by_date(
        features_df,
        signal_dates,
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
        target_horizon=args.target_horizon,
    )

    print()
    print("Step 5/5: Simulating pair trades")
    result = simulate_pair_backtest(
        stock_closes,
        predictions,
        tickers=tickers,
        days=args.days,
        initial_capital=100.0,
        regime=regimes,
        window=args.window,
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        transaction_cost_per_leg=args.transaction_cost_per_leg,
        slippage_per_leg=args.slippage_per_leg,
        shortable_tickers=shortable_tickers,
        require_short_availability=not args.allow_unlisted_shorts,
        require_cointegration=not args.disable_cointegration_filter,
        cointegration_pvalue_threshold=args.cointegration_pvalue_threshold,
        min_cointegration_observations=args.min_cointegration_observations,
        cointegration_lookback=args.cointegration_lookback,
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
    print(f"Blocked by short availability: {summary['trades_blocked_short']}")
    print(f"Blocked by cointegration: {summary['trades_blocked_cointegration']}")
    print(f"Forced closed at end: {summary['positions_forced_closed_end']}")
    print(f"Active pair-days: {summary['active_pair_days']}")
    print(
        "All-in per-leg cost: "
        f"{summary['cost_model']['all_in_cost_per_leg'] * 100.0:.3f}%"
    )
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
