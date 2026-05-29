"""Walk-forward model testing utilities from test.md."""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd


ModelFactory = Callable[[], Any]
FitFunction = Callable[[Any, np.ndarray, np.ndarray], Any]
PredictFunction = Callable[[Any, np.ndarray], Any]


class PurgedWalkForward:
    """Generate expanding train/test splits with an embargo before each test fold."""

    def __init__(self, n_splits: int = 5, embargo_days: int = 10):
        if n_splits < 1:
            raise ValueError("n_splits must be at least 1")
        if embargo_days < 0:
            raise ValueError("embargo_days cannot be negative")

        self.n_splits = n_splits
        self.embargo_days = embargo_days

    def split(
        self,
        x: Sequence[Any] | np.ndarray | pd.DataFrame | pd.Series,
        y: Optional[Sequence[Any] | np.ndarray | pd.Series] = None,
    ):
        """Yield train/test indices while skipping the embargo period."""

        n_samples = len(x)
        if y is not None and len(y) != n_samples:
            raise ValueError("x and y must have the same length")

        test_size = n_samples // (self.n_splits + 1)
        if test_size <= 0:
            raise ValueError("not enough samples for the requested number of splits")

        indices = np.arange(n_samples)

        for i in range(self.n_splits):
            test_start = (i + 1) * test_size
            test_end = test_start + test_size
            train_end = test_start - self.embargo_days

            if train_end <= 0:
                continue

            test_indices = indices[test_start:test_end]
            if len(test_indices) == 0:
                continue

            yield indices[:train_end], test_indices


def run_walk_forward_test(
    df: pd.DataFrame,
    model_factory: ModelFactory,
    features: Sequence[str],
    target: str,
    *,
    n_splits: int = 10,
    embargo_days: int = 15,
    fit_fn: Optional[FitFunction] = None,
    predict_fn: Optional[PredictFunction] = None,
    fit_kwargs: Optional[dict[str, Any]] = None,
    periods_per_year: int = 252,
    verbose: bool = True,
) -> pd.DataFrame:
    """Train fresh models on purged walk-forward folds and report real returns."""

    _require_columns(df, tuple(features) + (target,))
    columns = list(dict.fromkeys([*features, target]))
    working = df.loc[:, columns].replace([np.inf, -np.inf], np.nan).dropna()
    x = working.loc[:, features].to_numpy(dtype=float)
    y = working.loc[:, target].to_numpy(dtype=float)

    if len(x) == 0:
        raise ValueError("df must contain at least one row")

    results: list[dict[str, float]] = []
    splitter = PurgedWalkForward(n_splits=n_splits, embargo_days=embargo_days)

    if verbose:
        print("Starting Purged Walk-Forward Validation...")

    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y)):
        x_train, y_train = x[train_idx], y[train_idx]
        x_test, y_test = x[test_idx], y[test_idx]

        model = model_factory()
        _fit_model(model, x_train, y_train, fit_fn, fit_kwargs or {})
        predictions = _predict_model(model, x_test, predict_fn)

        strategy_returns = calculate_strategy_returns(predictions, y_test)
        sharpe = calculate_sharpe(strategy_returns, periods_per_year=periods_per_year)
        max_drawdown = calculate_max_drawdown(strategy_returns)
        rmse = calculate_rmse(y_test, predictions)
        directional_accuracy = calculate_directional_accuracy(predictions, y_test)

        row = {
            "fold": float(fold),
            "train_size": float(len(train_idx)),
            "test_size": float(len(test_idx)),
            "test_start": float(test_idx[0]),
            "test_end": float(test_idx[-1]),
            "test_sharpe": sharpe,
            "test_max_dd": max_drawdown,
            "rmse": rmse,
            "directional_accuracy": directional_accuracy,
            "mean_strategy_return": float(np.mean(strategy_returns)),
        }
        results.append(row)

        if verbose:
            print(
                f"Fold {fold} Complete | "
                f"Sharpe: {sharpe:.2f} | "
                f"Max DD: {max_drawdown:.2%}"
            )

    if not results:
        raise ValueError("no valid folds were produced; reduce n_splits or embargo_days")

    return pd.DataFrame(results)


def calculate_strategy_returns(predictions: Any, actuals: Any) -> np.ndarray:
    """Return FX-adjusted strategy PnL from directional model predictions."""

    predicted = _to_1d_float_array(predictions, "predictions")
    realized = _to_1d_float_array(actuals, "actuals")
    _require_same_length(predicted, realized, "predictions", "actuals")

    return np.sign(predicted) * realized


def calculate_directional_accuracy(predictions: Any, actuals: Any) -> float:
    """Calculate how often predicted and realized return signs match."""

    predicted = _to_1d_float_array(predictions, "predictions")
    realized = _to_1d_float_array(actuals, "actuals")
    _require_same_length(predicted, realized, "predictions", "actuals")

    return float(np.mean(np.sign(predicted) == np.sign(realized)))


def calculate_sharpe(
    returns: Any,
    *,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> float:
    """Calculate annualized Sharpe ratio for a return series."""

    values = _to_1d_float_array(returns, "returns")
    if len(values) < 2:
        return 0.0

    per_period_risk_free = risk_free_rate / periods_per_year
    excess = values - per_period_risk_free
    volatility = float(np.std(excess, ddof=1))
    if volatility == 0.0:
        return 0.0

    return float(np.sqrt(periods_per_year) * np.mean(excess) / volatility)


def calculate_max_drawdown(returns: Any) -> float:
    """Calculate maximum drawdown from a return series."""

    values = _to_1d_float_array(returns, "returns")
    if len(values) == 0:
        return 0.0

    equity_curve = np.cumprod(1.0 + values)
    running_peak = np.maximum.accumulate(equity_curve)
    drawdowns = equity_curve / running_peak - 1.0
    return float(np.min(drawdowns))


def calculate_rmse(actuals: Any, predictions: Any) -> float:
    """Calculate root mean squared prediction error."""

    realized = _to_1d_float_array(actuals, "actuals")
    predicted = _to_1d_float_array(predictions, "predictions")
    _require_same_length(realized, predicted, "actuals", "predictions")

    return float(np.sqrt(np.mean((realized - predicted) ** 2)))


def _fit_model(
    model: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    fit_fn: Optional[FitFunction],
    fit_kwargs: dict[str, Any],
) -> Any:
    if fit_fn is not None:
        return fit_fn(model, x_train, y_train)
    if not hasattr(model, "fit"):
        raise TypeError("model must define fit() or a fit_fn must be provided")
    return model.fit(x_train, y_train, **fit_kwargs)


def _predict_model(
    model: Any,
    x_test: np.ndarray,
    predict_fn: Optional[PredictFunction],
) -> np.ndarray:
    if predict_fn is not None:
        predictions = predict_fn(model, x_test)
    else:
        if not hasattr(model, "predict"):
            raise TypeError("model must define predict() or a predict_fn must be provided")
        predictions = model.predict(x_test)

    return _to_1d_float_array(predictions, "predictions")


def _to_1d_float_array(values: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc

    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")

    return array


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"df is missing required columns: {', '.join(missing)}")


def _require_same_length(
    left: np.ndarray,
    right: np.ndarray,
    left_name: str,
    right_name: str,
) -> None:
    if len(left) != len(right):
        raise ValueError(f"{left_name} and {right_name} must have the same length")
