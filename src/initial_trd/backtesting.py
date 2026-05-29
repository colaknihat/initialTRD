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
from initial_trd.strategy import Regime, TradeInstruction, generate_pairs_trade_signal
from initial_trd.training import (
    BISTResilientLSTM,
    generate_regime_weights,
    train_bist_model,
)


PredictionSource = Mapping[pd.Timestamp, float] | Callable[[pd.Timestamp], float]
RegimeSource = int | Mapping[pd.Timestamp, int] | Callable[[pd.Timestamp], int]
DEFAULT_TRANSACTION_COST_PER_LEG = 0.0010
DEFAULT_SLIPPAGE_PER_LEG = 0.0005
DEFAULT_COINTEGRATION_PVALUE_THRESHOLD = 0.05
DEFAULT_MIN_COINTEGRATION_OBSERVATIONS = 60


@dataclass(frozen=True)
class PairBacktestResult:
    rows: pd.DataFrame
    summary: dict[str, Any]


@dataclass(frozen=True)
class PairRelationship:
    hedge_ratio: float
    intercept: float
    t_stat: float
    p_value: float
    critical_value: float
    is_cointegrated: bool
    observations: int
    reason: str


@dataclass(frozen=True)
class PairPosition:
    long_leg: str
    short_leg: str
    hedge_ratio: float


@dataclass(frozen=True)
class PairReturnComponents:
    long_return: float
    short_return: float
    gross_pair_return: float
    pair_return: float
    trade_cost_return: float
    long_weight: float
    short_weight: float


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
    return generate_pairs_trade_signal(
        stock_a,
        stock_b,
        regime,
        lstm_prediction,
        window=window,
        entry_z=entry_z,
        exit_z=exit_z,
        stock_a_name=stock_a_name,
        stock_b_name=stock_b_name,
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
    regime: RegimeSource = int(Regime.DISINFLATION),
    window: int = 30,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    transaction_cost_per_leg: float = DEFAULT_TRANSACTION_COST_PER_LEG,
    slippage_per_leg: float = DEFAULT_SLIPPAGE_PER_LEG,
    shortable_tickers: Sequence[str] | None = None,
    require_short_availability: bool = True,
    require_cointegration: bool = True,
    cointegration_pvalue_threshold: float = DEFAULT_COINTEGRATION_PVALUE_THRESHOLD,
    min_cointegration_observations: int = DEFAULT_MIN_COINTEGRATION_OBSERVATIONS,
    cointegration_lookback: int | None = None,
) -> PairBacktestResult:
    """Simulate hedge-adjusted pair trades over the latest close-to-close window."""

    ticker_list = list(tickers or closes.columns)
    if len(ticker_list) < 2:
        raise ValueError("at least two tickers are required")
    _validate_backtest_execution_inputs(
        transaction_cost_per_leg=transaction_cost_per_leg,
        slippage_per_leg=slippage_per_leg,
        cointegration_pvalue_threshold=cointegration_pvalue_threshold,
        min_cointegration_observations=min_cointegration_observations,
        cointegration_lookback=cointegration_lookback,
    )

    normalized = _normalize_close_frame(closes, ticker_list)
    test_dates = select_common_trading_dates(normalized, ticker_list, days=days + 1)
    common = normalized.loc[:, ticker_list].dropna()
    pairs = list(combinations(ticker_list, 2))
    shortable = set(shortable_tickers or ())
    positions: dict[tuple[str, str], PairPosition | None] = {
        pair: None for pair in pairs
    }
    rows: list[dict[str, Any]] = []

    for signal_date, execution_date, next_date in zip(
        test_dates[:-2],
        test_dates[1:-1],
        test_dates[2:],
    ):
        predicted_return = _prediction_for_date(predictions, signal_date)
        regime_value = _regime_for_date(regime, signal_date)

        for stock_a, stock_b in pairs:
            pair_key = (stock_a, stock_b)
            relationship = estimate_pair_relationship(
                common.loc[:signal_date, [stock_a, stock_b]],
                stock_a,
                stock_b,
                lookback=cointegration_lookback,
                min_observations=min_cointegration_observations,
                pvalue_threshold=cointegration_pvalue_threshold,
            )
            signal_hedge_ratio = (
                relationship.hedge_ratio
                if _is_positive_finite(relationship.hedge_ratio)
                else 1.0
            )
            instruction = generate_symmetric_pair_signal(
                _log_close_series(common.loc[:signal_date, stock_a], stock_a),
                signal_hedge_ratio
                * _log_close_series(common.loc[:signal_date, stock_b], stock_b),
                regime_value,
                predicted_return,
                window=window,
                entry_z=entry_z,
                exit_z=exit_z,
                stock_a_name=stock_a,
                stock_b_name=stock_b,
            )
            previous_position = positions[pair_key]
            executed_action = "HOLD"
            active_position = False
            return_applied = False
            position_long = None
            position_short = None
            position_hedge_ratio = float("nan")
            long_return = 0.0
            short_return = 0.0
            gross_pair_return = 0.0
            pair_return = 0.0
            trade_cost_return = 0.0
            long_weight = 0.0
            short_weight = 0.0
            forced_close_end = False

            if instruction.action == "CLOSE":
                if previous_position is not None:
                    executed_action = "CLOSE"
                    position_long = previous_position.long_leg
                    position_short = previous_position.short_leg
                    position_hedge_ratio = previous_position.hedge_ratio
                    close_components = calculate_close_cost_components(
                        long_leg=position_long,
                        short_leg=position_short,
                        stock_a=stock_a,
                        stock_b=stock_b,
                        hedge_ratio=position_hedge_ratio,
                        transaction_cost_per_leg=transaction_cost_per_leg,
                        slippage_per_leg=slippage_per_leg,
                    )
                    long_return = close_components.long_return
                    short_return = close_components.short_return
                    gross_pair_return = close_components.gross_pair_return
                    pair_return = close_components.pair_return
                    trade_cost_return = close_components.trade_cost_return
                    long_weight = close_components.long_weight
                    short_weight = close_components.short_weight
                    return_applied = True
                positions[pair_key] = None
            elif instruction.action == "OPEN_PAIR" and previous_position is None:
                short_leg = str(instruction.short_leg)
                if require_short_availability and short_leg not in shortable:
                    executed_action = "BLOCKED_SHORT_UNAVAILABLE"
                elif require_cointegration and not relationship.is_cointegrated:
                    executed_action = "BLOCKED_COINTEGRATION"
                else:
                    position_hedge_ratio = signal_hedge_ratio
                    positions[pair_key] = PairPosition(
                        long_leg=str(instruction.long_leg),
                        short_leg=short_leg,
                        hedge_ratio=position_hedge_ratio,
                    )
                    executed_action = "OPEN_PAIR"

            current_position = positions[pair_key]
            if current_position is not None:
                forced_close_end = bool(next_date == test_dates[-1])
                position_long = current_position.long_leg
                position_short = current_position.short_leg
                position_hedge_ratio = current_position.hedge_ratio
                interval_components = calculate_pair_return_components(
                    common,
                    signal_date=execution_date,
                    next_date=next_date,
                    long_leg=position_long,
                    short_leg=position_short,
                    stock_a=stock_a,
                    stock_b=stock_b,
                    hedge_ratio=position_hedge_ratio,
                    transaction_cost_per_leg=transaction_cost_per_leg,
                    slippage_per_leg=slippage_per_leg,
                    charge_entry_cost=executed_action == "OPEN_PAIR",
                    charge_exit_cost=forced_close_end,
                )
                long_return = interval_components.long_return
                short_return = interval_components.short_return
                gross_pair_return = interval_components.gross_pair_return
                pair_return = interval_components.pair_return
                trade_cost_return = interval_components.trade_cost_return
                long_weight = interval_components.long_weight
                short_weight = interval_components.short_weight
                active_position = True
                return_applied = True
                if forced_close_end:
                    positions[pair_key] = None

            rows.append(
                {
                    "signal_date": _date_string(signal_date),
                    "execution_date": _date_string(execution_date),
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
                    "gross_pair_return": gross_pair_return,
                    "pair_return": pair_return,
                    "trade_cost_return": trade_cost_return,
                    "long_weight": long_weight,
                    "short_weight": short_weight,
                    "hedge_ratio": position_hedge_ratio
                    if _is_positive_finite(position_hedge_ratio)
                    else relationship.hedge_ratio,
                    "cointegration_pvalue": relationship.p_value,
                    "cointegration_t_stat": relationship.t_stat,
                    "cointegration_critical_value": relationship.critical_value,
                    "cointegration_observations": relationship.observations,
                    "cointegrated": relationship.is_cointegrated,
                    "cointegration_reason": relationship.reason,
                    "return_applied": return_applied,
                    "forced_close_end": forced_close_end,
                    "gross_100_pnl": initial_capital * pair_return
                    if return_applied
                    else 0.0,
                    "notional_200_pnl": initial_capital * 2.0 * pair_return
                    if return_applied
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
        transaction_cost_per_leg=transaction_cost_per_leg,
        slippage_per_leg=slippage_per_leg,
        require_short_availability=require_short_availability,
        shortable_tickers=sorted(shortable),
        require_cointegration=require_cointegration,
        cointegration_pvalue_threshold=cointegration_pvalue_threshold,
        min_cointegration_observations=min_cointegration_observations,
        cointegration_lookback=cointegration_lookback,
    )
    return PairBacktestResult(rows=row_frame, summary=summary)


def classify_backtest_regimes_by_date(
    features_df: pd.DataFrame,
    signal_dates: Sequence[pd.Timestamp],
    *,
    model_factory: Callable[[], Any] | None = None,
    n_components: int = 4,
    covariance_type: str = "diag",
    n_iter: int = 200,
    random_state: int | None = None,
) -> dict[pd.Timestamp, int]:
    """Classify the latest semantic HMM regime available for each signal date."""

    if "date" not in features_df.columns:
        raise ValueError("features_df must include a date column")
    if n_components < 1:
        raise ValueError("n_components must be at least 1")

    history = features_df.copy()
    history["date"] = pd.to_datetime(history["date"])
    history = history.sort_values("date")

    regimes: dict[pd.Timestamp, int] = {}
    for signal_date in signal_dates:
        timestamp = pd.Timestamp(signal_date)
        available = history.loc[history["date"] <= timestamp].copy()
        if len(available) < n_components:
            raise ValueError(
                "not enough feature rows are available through "
                f"{_date_string(timestamp)} to classify regimes"
            )

        model = model_factory() if model_factory is not None else None
        weighted = generate_regime_weights(
            available,
            model=model,
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
        )
        regimes[timestamp] = int(weighted["regime"].iloc[-1])

    return regimes


def estimate_pair_relationship(
    closes: pd.DataFrame,
    stock_a: str,
    stock_b: str,
    *,
    lookback: int | None = None,
    min_observations: int = DEFAULT_MIN_COINTEGRATION_OBSERVATIONS,
    pvalue_threshold: float = DEFAULT_COINTEGRATION_PVALUE_THRESHOLD,
) -> PairRelationship:
    """Estimate a log-price hedge ratio and Engle-Granger cointegration status."""

    _validate_cointegration_inputs(
        pvalue_threshold=pvalue_threshold,
        min_observations=min_observations,
        lookback=lookback,
    )
    normalized = _normalize_close_frame(closes, [stock_a, stock_b])
    pair_closes = normalized.loc[:, [stock_a, stock_b]].dropna()
    if lookback is not None:
        pair_closes = pair_closes.tail(lookback)

    observations = int(len(pair_closes))
    invalid = _invalid_relationship(
        observations=observations,
        min_observations=min_observations,
        pvalue_threshold=pvalue_threshold,
        reason=f"at least {min_observations} observations are required",
    )
    if observations < min_observations:
        return invalid
    if (pair_closes <= 0.0).any().any():
        return _invalid_relationship(
            observations=observations,
            min_observations=min_observations,
            pvalue_threshold=pvalue_threshold,
            reason="close prices must be positive for log-price cointegration",
        )

    log_a = np.log(pair_closes[stock_a].to_numpy(dtype=float))
    log_b = np.log(pair_closes[stock_b].to_numpy(dtype=float))
    if np.isclose(float(np.var(log_b)), 0.0):
        return _invalid_relationship(
            observations=observations,
            min_observations=min_observations,
            pvalue_threshold=pvalue_threshold,
            reason=f"{stock_b} log prices have no variance",
        )

    design = np.column_stack([np.ones_like(log_b), log_b])
    intercept, hedge_ratio = np.linalg.lstsq(design, log_a, rcond=None)[0]
    if not _is_positive_finite(float(hedge_ratio)):
        return _invalid_relationship(
            observations=observations,
            min_observations=min_observations,
            pvalue_threshold=pvalue_threshold,
            reason="estimated hedge ratio is not positive and finite",
            intercept=float(intercept),
            hedge_ratio=float(hedge_ratio),
        )

    residual = log_a - float(intercept) - float(hedge_ratio) * log_b
    t_stat, p_value, critical_value, reason = _engle_granger_residual_test(
        residual,
        pvalue_threshold=pvalue_threshold,
    )
    is_cointegrated = bool(np.isfinite(t_stat) and t_stat <= critical_value)
    if is_cointegrated:
        reason = "cointegration accepted"

    return PairRelationship(
        hedge_ratio=float(hedge_ratio),
        intercept=float(intercept),
        t_stat=float(t_stat),
        p_value=float(p_value),
        critical_value=float(critical_value),
        is_cointegrated=is_cointegrated,
        observations=observations,
        reason=reason,
    )


def calculate_pair_return_components(
    closes: pd.DataFrame,
    *,
    signal_date: pd.Timestamp,
    next_date: pd.Timestamp,
    long_leg: str,
    short_leg: str,
    stock_a: str | None = None,
    stock_b: str | None = None,
    hedge_ratio: float = 1.0,
    transaction_cost_per_leg: float = 0.0,
    slippage_per_leg: float = 0.0,
    charge_entry_cost: bool = False,
    charge_exit_cost: bool = False,
) -> PairReturnComponents:
    """Return hedge-weighted pair-return details for one close-to-close interval."""

    if not _is_positive_finite(hedge_ratio):
        raise ValueError("hedge_ratio must be positive and finite")
    if transaction_cost_per_leg < 0.0:
        raise ValueError("transaction_cost_per_leg cannot be negative")
    if slippage_per_leg < 0.0:
        raise ValueError("slippage_per_leg cannot be negative")

    current_long = float(closes.loc[signal_date, long_leg])
    next_long = float(closes.loc[next_date, long_leg])
    current_short = float(closes.loc[signal_date, short_leg])
    next_short = float(closes.loc[next_date, short_leg])
    if current_long <= 0.0 or current_short <= 0.0:
        raise ValueError("close prices must be positive to calculate returns")

    long_weight, short_weight = _hedge_weights(
        long_leg=long_leg,
        short_leg=short_leg,
        hedge_ratio=hedge_ratio,
        stock_a=stock_a,
        stock_b=stock_b,
    )
    long_return = next_long / current_long - 1.0
    short_return = -(next_short / current_short - 1.0)
    gross_pair_return = long_weight * long_return + short_weight * short_return
    trade_cost_return = _trade_cost_return(
        long_weight=long_weight,
        short_weight=short_weight,
        transaction_cost_per_leg=transaction_cost_per_leg,
        slippage_per_leg=slippage_per_leg,
        charge_entry_cost=charge_entry_cost,
        charge_exit_cost=charge_exit_cost,
    )
    pair_return = gross_pair_return - trade_cost_return
    return PairReturnComponents(
        long_return=float(long_return),
        short_return=float(short_return),
        gross_pair_return=float(gross_pair_return),
        pair_return=float(pair_return),
        trade_cost_return=float(trade_cost_return),
        long_weight=float(long_weight),
        short_weight=float(short_weight),
    )


def calculate_close_cost_components(
    *,
    long_leg: str,
    short_leg: str,
    stock_a: str,
    stock_b: str,
    hedge_ratio: float,
    transaction_cost_per_leg: float,
    slippage_per_leg: float,
) -> PairReturnComponents:
    """Return exit-cost-only components for a position closed at the signal date."""

    long_weight, short_weight = _hedge_weights(
        long_leg=long_leg,
        short_leg=short_leg,
        hedge_ratio=hedge_ratio,
        stock_a=stock_a,
        stock_b=stock_b,
    )
    trade_cost_return = _trade_cost_return(
        long_weight=long_weight,
        short_weight=short_weight,
        transaction_cost_per_leg=transaction_cost_per_leg,
        slippage_per_leg=slippage_per_leg,
        charge_entry_cost=False,
        charge_exit_cost=True,
    )
    return PairReturnComponents(
        long_return=0.0,
        short_return=0.0,
        gross_pair_return=0.0,
        pair_return=float(-trade_cost_return),
        trade_cost_return=float(trade_cost_return),
        long_weight=float(long_weight),
        short_weight=float(short_weight),
    )


def calculate_pair_return(
    closes: pd.DataFrame,
    *,
    signal_date: pd.Timestamp,
    next_date: pd.Timestamp,
    long_leg: str,
    short_leg: str,
    stock_a: str | None = None,
    stock_b: str | None = None,
    hedge_ratio: float = 1.0,
    transaction_cost_per_leg: float = 0.0,
    slippage_per_leg: float = 0.0,
    charge_entry_cost: bool = False,
    charge_exit_cost: bool = False,
) -> tuple[float, float, float]:
    """Return long, short, and net hedge-weighted pair returns for one interval."""

    components = calculate_pair_return_components(
        closes,
        signal_date=signal_date,
        next_date=next_date,
        long_leg=long_leg,
        short_leg=short_leg,
        stock_a=stock_a,
        stock_b=stock_b,
        hedge_ratio=hedge_ratio,
        transaction_cost_per_leg=transaction_cost_per_leg,
        slippage_per_leg=slippage_per_leg,
        charge_entry_cost=charge_entry_cost,
        charge_exit_cost=charge_exit_cost,
    )
    return (
        components.long_return,
        components.short_return,
        components.pair_return,
    )



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

    pnl_rows = rows.loc[_return_applied_mask(rows)]
    portfolio_value = float(initial_capital)
    for _, day_rows in rows.groupby("signal_date", sort=True):
        active_day = day_rows.loc[_return_applied_mask(day_rows)]
        daily_return = (
            float(active_day["pair_return"].mean()) if not active_day.empty else 0.0
        )
        portfolio_value *= 1.0 + daily_return

    gross_ending = initial_capital + float(pnl_rows["gross_100_pnl"].sum())
    notional_ending = initial_capital + float(pnl_rows["notional_200_pnl"].sum())

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
    transaction_cost_per_leg: float = 0.0,
    slippage_per_leg: float = 0.0,
    require_short_availability: bool = False,
    shortable_tickers: Sequence[str] | None = None,
    require_cointegration: bool = False,
    cointegration_pvalue_threshold: float = DEFAULT_COINTEGRATION_PVALUE_THRESHOLD,
    min_cointegration_observations: int = DEFAULT_MIN_COINTEGRATION_OBSERVATIONS,
    cointegration_lookback: int | None = None,
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
        "trades_blocked_short": int(
            (rows["executed_action"] == "BLOCKED_SHORT_UNAVAILABLE").sum()
        )
        if not rows.empty
        else 0,
        "trades_blocked_cointegration": int(
            (rows["executed_action"] == "BLOCKED_COINTEGRATION").sum()
        )
        if not rows.empty
        else 0,
        "positions_forced_closed_end": int(rows["forced_close_end"].sum())
        if not rows.empty and "forced_close_end" in rows.columns
        else 0,
        "active_pair_days": int(len(active)),
        "max_active_pairs": int(active_counts.max()) if not active_counts.empty else 0,
        "cost_model": {
            "transaction_cost_per_leg": float(transaction_cost_per_leg),
            "slippage_per_leg": float(slippage_per_leg),
            "all_in_cost_per_leg": float(
                transaction_cost_per_leg + slippage_per_leg
            ),
            "round_trip_cost_gross_100": float(
                2.0 * (transaction_cost_per_leg + slippage_per_leg)
            ),
            "round_trip_cost_notional_200": float(
                4.0 * (transaction_cost_per_leg + slippage_per_leg)
            ),
        },
        "short_selling": {
            "required": bool(require_short_availability),
            "shortable_tickers": list(shortable_tickers or []),
        },
        "cointegration_filter": {
            "required": bool(require_cointegration),
            "pvalue_threshold": float(cointegration_pvalue_threshold),
            "min_observations": int(min_cointegration_observations),
            "lookback": None
            if cointegration_lookback is None
            else int(cointegration_lookback),
        },
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
    target_horizon: int = 1,
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
            target_horizon=target_horizon,
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
    target_horizon: int = 1,
) -> float:
    if target_horizon < 1:
        raise ValueError("target_horizon must be at least 1")
    if "date" not in features_df.columns:
        raise ValueError("features_df must include a date column")

    history = features_df.copy()
    history["date"] = pd.to_datetime(history["date"])
    history = history.loc[history["date"] <= signal_date].copy()
    if history.empty:
        raise ValueError(f"no feature rows are available through {_date_string(signal_date)}")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    x, y, weights = build_sequence_arrays(
        _drop_unrevealed_target_rows(history, target_horizon=target_horizon),
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
        feature_columns=feature_columns,
        use_regime_weights=use_regime_weights,
        hmm_random_state=hmm_random_state,
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


def _drop_unrevealed_target_rows(
    history: pd.DataFrame,
    *,
    target_horizon: int,
) -> pd.DataFrame:
    if len(history) <= target_horizon:
        raise ValueError("not enough feature rows have revealed targets")
    return history.iloc[:-target_horizon].copy()


def _validate_backtest_execution_inputs(
    *,
    transaction_cost_per_leg: float,
    slippage_per_leg: float,
    cointegration_pvalue_threshold: float,
    min_cointegration_observations: int,
    cointegration_lookback: int | None,
) -> None:
    if transaction_cost_per_leg < 0.0:
        raise ValueError("transaction_cost_per_leg cannot be negative")
    if slippage_per_leg < 0.0:
        raise ValueError("slippage_per_leg cannot be negative")
    _validate_cointegration_inputs(
        pvalue_threshold=cointegration_pvalue_threshold,
        min_observations=min_cointegration_observations,
        lookback=cointegration_lookback,
    )


def _validate_cointegration_inputs(
    *,
    pvalue_threshold: float,
    min_observations: int,
    lookback: int | None,
) -> None:
    if not 0.0 < pvalue_threshold < 1.0:
        raise ValueError("cointegration p-value threshold must be between 0 and 1")
    if min_observations < 4:
        raise ValueError("min_cointegration_observations must be at least 4")
    if lookback is not None and lookback < min_observations:
        raise ValueError("cointegration_lookback must be at least min_observations")


def _invalid_relationship(
    *,
    observations: int,
    min_observations: int,
    pvalue_threshold: float,
    reason: str,
    intercept: float = float("nan"),
    hedge_ratio: float = float("nan"),
) -> PairRelationship:
    del min_observations
    critical_value = _critical_value_for_threshold(pvalue_threshold)
    return PairRelationship(
        hedge_ratio=float(hedge_ratio),
        intercept=float(intercept),
        t_stat=float("nan"),
        p_value=1.0,
        critical_value=critical_value,
        is_cointegrated=False,
        observations=int(observations),
        reason=reason,
    )


def _engle_granger_residual_test(
    residual: np.ndarray,
    *,
    pvalue_threshold: float,
) -> tuple[float, float, float, str]:
    return _fallback_residual_adf_test(
        residual,
        pvalue_threshold=pvalue_threshold,
    )


def _fallback_residual_adf_test(
    residual: np.ndarray,
    *,
    pvalue_threshold: float,
) -> tuple[float, float, float, str]:
    values = np.asarray(residual, dtype=float).reshape(-1)
    critical_value = _critical_value_for_threshold(pvalue_threshold)
    if len(values) < 4:
        return float("nan"), 1.0, critical_value, "not enough residual observations"
    if not np.isfinite(values).all():
        return float("nan"), 1.0, critical_value, "residuals are not finite"

    lagged = values[:-1]
    delta = np.diff(values)
    if np.isclose(float(np.var(lagged)), 0.0):
        return float("nan"), 1.0, critical_value, "residuals have no variance"

    design = np.column_stack([np.ones_like(lagged), lagged])
    coefficients = np.linalg.lstsq(design, delta, rcond=None)[0]
    errors = delta - design @ coefficients
    dof = len(delta) - design.shape[1]
    if dof <= 0:
        return float("nan"), 1.0, critical_value, "not enough degrees of freedom"

    xtx_inv = np.linalg.pinv(design.T @ design)
    residual_variance = float((errors @ errors) / dof)
    gamma_variance = float(residual_variance * xtx_inv[1, 1])
    if gamma_variance <= 0.0:
        return float("nan"), 1.0, critical_value, "ADF coefficient variance is zero"

    t_stat = float(coefficients[1] / np.sqrt(gamma_variance))
    p_value = _approximate_cointegration_pvalue(t_stat)
    if t_stat <= critical_value:
        return t_stat, p_value, critical_value, "cointegration accepted"
    return (
        t_stat,
        p_value,
        critical_value,
        "residual unit-root test does not reject non-cointegration",
    )


def _critical_value_for_threshold(pvalue_threshold: float) -> float:
    if pvalue_threshold <= 0.01:
        return -3.96
    if pvalue_threshold <= 0.05:
        return -3.37
    if pvalue_threshold <= 0.10:
        return -3.07
    return -2.86


def _approximate_cointegration_pvalue(t_stat: float) -> float:
    if not np.isfinite(t_stat):
        return 1.0
    if t_stat <= -3.96:
        return 0.01
    if t_stat <= -3.37:
        return 0.05
    if t_stat <= -3.07:
        return 0.10
    if t_stat <= -2.86:
        return 0.15
    return 0.50


def _log_close_series(series: pd.Series, name: str) -> pd.Series:
    values = pd.Series(series, dtype=float)
    if (values <= 0.0).any():
        raise ValueError(f"{name} close prices must be positive")
    return np.log(values)


def _hedge_weights(
    *,
    long_leg: str,
    short_leg: str,
    hedge_ratio: float,
    stock_a: str | None = None,
    stock_b: str | None = None,
) -> tuple[float, float]:
    if not _is_positive_finite(hedge_ratio):
        raise ValueError("hedge_ratio must be positive and finite")

    denominator = 1.0 + float(hedge_ratio)
    if stock_a is None or stock_b is None:
        return 1.0 / denominator, float(hedge_ratio) / denominator
    if long_leg == stock_a and short_leg == stock_b:
        return 1.0 / denominator, float(hedge_ratio) / denominator
    if long_leg == stock_b and short_leg == stock_a:
        return float(hedge_ratio) / denominator, 1.0 / denominator
    raise ValueError("long_leg and short_leg must match stock_a/stock_b")


def _trade_cost_return(
    *,
    long_weight: float,
    short_weight: float,
    transaction_cost_per_leg: float,
    slippage_per_leg: float,
    charge_entry_cost: bool,
    charge_exit_cost: bool,
) -> float:
    if transaction_cost_per_leg < 0.0:
        raise ValueError("transaction_cost_per_leg cannot be negative")
    if slippage_per_leg < 0.0:
        raise ValueError("slippage_per_leg cannot be negative")

    turnover_count = int(charge_entry_cost) + int(charge_exit_cost)
    if turnover_count == 0:
        return 0.0
    all_in_cost = transaction_cost_per_leg + slippage_per_leg
    return float(turnover_count * all_in_cost * (abs(long_weight) + abs(short_weight)))


def _return_applied_mask(rows: pd.DataFrame) -> pd.Series:
    if "return_applied" in rows.columns:
        return rows["return_applied"].astype(bool)
    return rows["active_position"].astype(bool)


def _is_positive_finite(value: float) -> bool:
    return bool(np.isfinite(value) and value > 0.0)


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


def _regime_for_date(regimes: RegimeSource, date: pd.Timestamp) -> int:
    timestamp = pd.Timestamp(date)
    if callable(regimes):
        return int(regimes(timestamp))
    if isinstance(regimes, Mapping):
        if timestamp in regimes:
            return int(regimes[timestamp])

        normalized = pd.Timestamp(timestamp.date())
        if normalized in regimes:
            return int(regimes[normalized])
        raise KeyError(f"missing regime for {_date_string(timestamp)}")

    values = np.asarray(regimes, dtype=int).reshape(-1)
    if values.size != 1:
        raise ValueError("regime must contain exactly one value")
    return int(values[0])


def _account_values(initial_capital: float, ending_value: float) -> dict[str, float]:
    total_pnl = ending_value - initial_capital
    return {
        "ending_value": float(ending_value),
        "total_pnl": float(total_pnl),
        "return_pct": float(total_pnl / initial_capital),
    }


def _date_string(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")
