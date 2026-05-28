"""Pair-trade backtesting helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
import torch

from initial_trd.cli.train_lstm import build_loaders, build_sequence_arrays
from initial_trd.strategy import Regime, TradeInstruction
from initial_trd.training import (
    BISTResilientLSTM,
    generate_regime_weights,
    train_bist_model,
)


PredictionSource = Mapping[pd.Timestamp, float] | Callable[[pd.Timestamp], float]


@dataclass(frozen=True)
class PairBacktestResult:
    rows: pd.DataFrame
    summary: dict[str, Any]


def generate_symmetric_pair_signal(
    stock_a: Any,
    stock_b: Any,
    regime: Any,
    lstm_prediction: Any,
    *,
    window: int = 30,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stock_a_name: str = "stock_A",
    stock_b_name: str = "stock_B",
) -> TradeInstruction:
    """Return a two-sided pair signal from the current spread z-score."""

    if window < 2:
        raise ValueError("window must be at least 2")
    entry_threshold = abs(float(entry_z))
    exit_threshold = abs(float(exit_z))
    if entry_threshold <= 0.0:
        raise ValueError("entry_z must be non-zero")

    close_a = pd.Series(stock_a, dtype=float).rename("stock_a")
    close_b = pd.Series(stock_b, dtype=float).rename("stock_b")
    prices = pd.concat([close_a, close_b], axis=1).dropna()
    if len(prices) < window:
        raise ValueError(f"at least {window} aligned close prices are required")

    spread = prices["stock_a"] - prices["stock_b"]
    rolling_mean = spread.rolling(window).mean()
    rolling_std = spread.rolling(window).std()
    current_z = float(((spread - rolling_mean) / rolling_std).iloc[-1])
    regime_value = int(np.asarray(regime, dtype=int).reshape(-1)[0])
    predicted_return = float(np.asarray(lstm_prediction, dtype=float).reshape(-1)[0])

    if not np.isfinite(current_z):
        return TradeInstruction(
            action="HOLD",
            reason="z-score is unavailable for the latest window",
            z_score=current_z,
            regime=regime_value,
            predicted_return=predicted_return,
        )

    if (
        regime_value == Regime.DISINFLATION
        and predicted_return > 0.0
        and current_z < -entry_threshold
    ):
        return TradeInstruction(
            action="OPEN_PAIR",
            reason="negative spread z-score, positive momentum",
            z_score=current_z,
            regime=regime_value,
            predicted_return=predicted_return,
            long_leg=stock_a_name,
            short_leg=stock_b_name,
        )

    if (
        regime_value == Regime.DISINFLATION
        and predicted_return > 0.0
        and current_z > entry_threshold
    ):
        return TradeInstruction(
            action="OPEN_PAIR",
            reason="positive spread z-score, positive momentum",
            z_score=current_z,
            regime=regime_value,
            predicted_return=predicted_return,
            long_leg=stock_b_name,
            short_leg=stock_a_name,
        )

    if abs(current_z) < exit_threshold:
        return TradeInstruction(
            action="CLOSE",
            reason="spread mean reversion target reached",
            z_score=current_z,
            regime=regime_value,
            predicted_return=predicted_return,
        )

    return TradeInstruction(
        action="HOLD",
        reason="entry and exit conditions are not met",
        z_score=current_z,
        regime=regime_value,
        predicted_return=predicted_return,
    )


def select_common_trading_dates(
    closes: pd.DataFrame,
    tickers: Sequence[str],
    *,
    days: int,
) -> pd.DatetimeIndex:
    """Return the latest days + 1 dates where every requested ticker has a close."""

    if days < 1:
        raise ValueError("days must be at least 1")

    normalized = _normalize_close_frame(closes, tickers)
    common = normalized.loc[:, list(tickers)].dropna()
    required_dates = days + 1
    if len(common) < required_dates:
        raise ValueError(f"at least {required_dates} common trading dates are required")

    return pd.DatetimeIndex(common.index[-required_dates:])


def simulate_pair_backtest(
    closes: pd.DataFrame,
    predictions: PredictionSource,
    *,
    tickers: Sequence[str] | None = None,
    days: int = 30,
    initial_capital: float = 100.0,
    regime: int = int(Regime.DISINFLATION),
    window: int = 30,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
) -> PairBacktestResult:
    """Simulate symmetric pair-trade signals over the latest close-to-close window."""

    ticker_list = list(tickers or closes.columns)
    if len(ticker_list) < 2:
        raise ValueError("at least two tickers are required")

    normalized = _normalize_close_frame(closes, ticker_list)
    test_dates = select_common_trading_dates(normalized, ticker_list, days=days)
    common = normalized.loc[:, ticker_list].dropna()
    pairs = list(combinations(ticker_list, 2))
    positions: dict[tuple[str, str], tuple[str, str] | None] = {
        pair: None for pair in pairs
    }
    rows: list[dict[str, Any]] = []

    for signal_date, next_date in zip(test_dates[:-1], test_dates[1:]):
        predicted_return = _prediction_for_date(predictions, signal_date)

        for stock_a, stock_b in pairs:
            pair_key = (stock_a, stock_b)
            instruction = generate_symmetric_pair_signal(
                common.loc[:signal_date, stock_a],
                common.loc[:signal_date, stock_b],
                regime,
                predicted_return,
                window=window,
                entry_z=entry_z,
                exit_z=exit_z,
                stock_a_name=stock_a,
                stock_b_name=stock_b,
            )
            previous_position = positions[pair_key]
            executed_action = "HOLD"

            if instruction.action == "CLOSE":
                if previous_position is not None:
                    executed_action = "CLOSE"
                positions[pair_key] = None
            elif instruction.action == "OPEN_PAIR" and previous_position is None:
                positions[pair_key] = (instruction.long_leg, instruction.short_leg)  # type: ignore[arg-type]
                executed_action = "OPEN_PAIR"

            current_position = positions[pair_key]
            if current_position is None:
                long_return = 0.0
                short_return = 0.0
                pair_return = 0.0
                position_long = None
                position_short = None
                active_position = False
            else:
                position_long, position_short = current_position
                long_return, short_return, pair_return = calculate_pair_return(
                    common,
                    signal_date=signal_date,
                    next_date=next_date,
                    long_leg=position_long,
                    short_leg=position_short,
                )
                active_position = True

            rows.append(
                {
                    "signal_date": _date_string(signal_date),
                    "next_date": _date_string(next_date),
                    "pair": f"{stock_a}/{stock_b}",
                    "stock_a": stock_a,
                    "stock_b": stock_b,
                    "signal_action": instruction.action,
                    "executed_action": executed_action,
                    "reason": instruction.reason,
                    "z_score": instruction.z_score,
                    "regime": instruction.regime,
                    "predicted_return": instruction.predicted_return,
                    "position_long": position_long,
                    "position_short": position_short,
                    "active_position": active_position,
                    "long_return": long_return,
                    "short_return": short_return,
                    "pair_return": pair_return,
                    "gross_100_pnl": initial_capital * pair_return
                    if active_position
                    else 0.0,
                    "notional_200_pnl": initial_capital * 2.0 * pair_return
                    if active_position
                    else 0.0,
                }
            )

    row_frame = pd.DataFrame(rows)
    summary = build_backtest_summary(
        row_frame,
        tickers=ticker_list,
        days=days,
        pairs_tested=len(pairs),
        initial_capital=initial_capital,
    )
    return PairBacktestResult(rows=row_frame, summary=summary)


def calculate_pair_return(
    closes: pd.DataFrame,
    *,
    signal_date: pd.Timestamp,
    next_date: pd.Timestamp,
    long_leg: str,
    short_leg: str,
) -> tuple[float, float, float]:
    """Return long, short, and gross-balanced pair returns for one interval."""

    current_long = float(closes.loc[signal_date, long_leg])
    next_long = float(closes.loc[next_date, long_leg])
    current_short = float(closes.loc[signal_date, short_leg])
    next_short = float(closes.loc[next_date, short_leg])
    if current_long <= 0.0 or current_short <= 0.0:
        raise ValueError("close prices must be positive to calculate returns")

    long_return = next_long / current_long - 1.0
    short_return = -(next_short / current_short - 1.0)
    pair_return = 0.5 * long_return + 0.5 * short_return
    return float(long_return), float(short_return), float(pair_return)


def calculate_backtest_account_values(
    rows: pd.DataFrame,
    *,
    initial_capital: float = 100.0,
) -> dict[str, dict[str, float]]:
    """Calculate the three requested account views from pair-level rows."""

    if rows.empty:
        return {
            name: _account_values(initial_capital, initial_capital)
            for name in (
                "portfolio_100",
                "gross_100_per_signal",
                "notional_200_per_signal",
            )
        }

    active = rows.loc[rows["active_position"].astype(bool)]
    portfolio_value = float(initial_capital)
    for _, day_rows in rows.groupby("signal_date", sort=True):
        active_day = day_rows.loc[day_rows["active_position"].astype(bool)]
        daily_return = (
            float(active_day["pair_return"].mean()) if not active_day.empty else 0.0
        )
        portfolio_value *= 1.0 + daily_return

    gross_ending = initial_capital + float(active["gross_100_pnl"].sum())
    notional_ending = initial_capital + float(active["notional_200_pnl"].sum())

    return {
        "portfolio_100": _account_values(initial_capital, portfolio_value),
        "gross_100_per_signal": _account_values(initial_capital, gross_ending),
        "notional_200_per_signal": _account_values(initial_capital, notional_ending),
    }


def build_backtest_summary(
    rows: pd.DataFrame,
    *,
    tickers: Sequence[str],
    days: int,
    pairs_tested: int,
    initial_capital: float,
) -> dict[str, Any]:
    account_values = calculate_backtest_account_values(
        rows,
        initial_capital=initial_capital,
    )
    active = rows.loc[rows["active_position"].astype(bool)] if not rows.empty else rows
    active_counts = (
        active.groupby("signal_date").size()
        if not active.empty
        else pd.Series(dtype=float)
    )

    return {
        "days": int(days),
        "tickers": list(tickers),
        "pairs_tested": int(pairs_tested),
        "initial_capital": float(initial_capital),
        "start_date": None if rows.empty else str(rows["signal_date"].iloc[0]),
        "end_date": None if rows.empty else str(rows["next_date"].iloc[-1]),
        "trades_opened": int((rows["executed_action"] == "OPEN_PAIR").sum())
        if not rows.empty
        else 0,
        "trades_closed": int((rows["executed_action"] == "CLOSE").sum())
        if not rows.empty
        else 0,
        "active_pair_days": int(len(active)),
        "max_active_pairs": int(active_counts.max()) if not active_counts.empty else 0,
        "account_values": account_values,
    }


def train_lstm_predictions_by_date(
    features_df: pd.DataFrame,
    signal_dates: Sequence[pd.Timestamp],
    *,
    feature_columns: Sequence[str],
    target_column: str,
    weight_column: str,
    sequence_length: int,
    validation_size: float,
    batch_size: int,
    epochs: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    seed: int,
    device: str,
    use_regime_weights: bool,
    hmm_random_state: int | None,
    verbose: bool = True,
) -> dict[pd.Timestamp, float]:
    predictions: dict[pd.Timestamp, float] = {}
    total = len(signal_dates)
    for index, signal_date in enumerate(signal_dates, start=1):
        if verbose:
            print(f"Training daily LSTM {index}/{total} for {_date_string(signal_date)}")
        predictions[pd.Timestamp(signal_date)] = train_lstm_prediction_for_date(
            features_df,
            pd.Timestamp(signal_date),
            feature_columns=feature_columns,
            target_column=target_column,
            weight_column=weight_column,
            sequence_length=sequence_length,
            validation_size=validation_size,
            batch_size=batch_size,
            epochs=epochs,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            seed=seed,
            device=device,
            use_regime_weights=use_regime_weights,
            hmm_random_state=hmm_random_state,
        )
    return predictions


def train_lstm_prediction_for_date(
    features_df: pd.DataFrame,
    signal_date: pd.Timestamp,
    *,
    feature_columns: Sequence[str],
    target_column: str,
    weight_column: str,
    sequence_length: int,
    validation_size: float,
    batch_size: int,
    epochs: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    seed: int,
    device: str,
    use_regime_weights: bool,
    hmm_random_state: int | None,
) -> float:
    if "date" not in features_df.columns:
        raise ValueError("features_df must include a date column")

    history = features_df.copy()
    history["date"] = pd.to_datetime(history["date"])
    history = history.loc[history["date"] <= signal_date].copy()
    if history.empty:
        raise ValueError(f"no feature rows are available through {_date_string(signal_date)}")
    if use_regime_weights:
        history = generate_regime_weights(history, random_state=hmm_random_state)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    x, y, weights = build_sequence_arrays(
        history,
        feature_columns=list(feature_columns),
        target_column=target_column,
        weight_column=weight_column,
        sequence_length=sequence_length,
    )
    train_loader, val_loader = build_loaders(
        x,
        y,
        weights,
        validation_size=validation_size,
        batch_size=batch_size,
    )
    model = BISTResilientLSTM(
        input_dim=len(feature_columns),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    train_bist_model(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
        verbose=False,
    )

    device_obj = torch.device(device)
    prediction_features = (
        history.loc[:, list(feature_columns)]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .tail(sequence_length)
        .to_numpy(dtype=np.float32)
    )
    if len(prediction_features) < sequence_length:
        raise ValueError("not enough feature rows are available for prediction")

    model.eval()
    x_pred = torch.from_numpy(
        prediction_features.reshape(1, sequence_length, len(feature_columns))
    ).to(device_obj)
    with torch.no_grad():
        prediction = model(x_pred)

    return float(prediction.detach().cpu().numpy().reshape(-1)[0])


def resolve_device(value: str | None) -> str:
    if value is None or value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return value


def _normalize_close_frame(
    closes: pd.DataFrame,
    tickers: Sequence[str],
) -> pd.DataFrame:
    missing = [ticker for ticker in tickers if ticker not in closes.columns]
    if missing:
        raise ValueError(f"close data is missing tickers: {', '.join(missing)}")

    normalized = closes.copy()
    if "date" in normalized.columns:
        normalized = normalized.set_index("date")
    if not isinstance(normalized.index, pd.DatetimeIndex):
        raise ValueError("close data must use a DatetimeIndex or include a date column")

    normalized.index = pd.to_datetime(normalized.index)
    normalized = normalized.sort_index()
    if normalized.index.has_duplicates:
        raise ValueError("close data dates must be unique")
    return normalized


def _prediction_for_date(predictions: PredictionSource, date: pd.Timestamp) -> float:
    timestamp = pd.Timestamp(date)
    if callable(predictions):
        return float(predictions(timestamp))
    if timestamp in predictions:
        return float(predictions[timestamp])

    normalized = pd.Timestamp(timestamp.date())
    if normalized in predictions:
        return float(predictions[normalized])
    raise KeyError(f"missing prediction for {_date_string(timestamp)}")


def _account_values(initial_capital: float, ending_value: float) -> dict[str, float]:
    total_pnl = ending_value - initial_capital
    return {
        "ending_value": float(ending_value),
        "total_pnl": float(total_pnl),
        "return_pct": float(total_pnl / initial_capital),
    }


def _date_string(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")
