"""Trading strategy components from pseudocode.md.

The module keeps broker side effects behind optional callbacks so strategy
logic can be tested without placing real orders.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable, Literal, Optional

import numpy as np
import pandas as pd


class Regime(IntEnum):
    HIGH_INFLATION = 0
    DISINFLATION = 1
    CRISIS = 2


@dataclass(frozen=True)
class TradeInstruction:
    action: Literal["OPEN_PAIR", "CLOSE", "HOLD"]
    reason: str
    z_score: float
    regime: int
    predicted_return: float
    long_leg: Optional[str] = None
    short_leg: Optional[str] = None


OrderExecutor = Callable[..., Any]
PositionCloser = Callable[[], Any]


def classify_regime(
    macro_data: Any,
    model: Optional[Any] = None,
    *,
    n_components: int = 3,
    covariance_type: str = "diag",
    n_iter: int = 100,
    random_state: Optional[int] = None,
) -> int:
    """Fit an HMM on macro data and return the latest inferred regime."""

    values = _to_2d_float_array(macro_data, "macro_data")
    fitted_model = model or _build_hmm_model(
        n_components=n_components,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
    )

    fitted_model.fit(values)
    prediction = np.asarray(fitted_model.predict(values[-1:])).reshape(-1)
    if prediction.size != 1:
        raise ValueError("HMM prediction must contain exactly one regime")

    return int(prediction[0])


def build_lstm_model(timesteps: int, feature_count: int) -> Any:
    """Build the LSTM architecture from the pseudocode."""

    try:
        from tensorflow.keras.layers import LSTM, Dense
        from tensorflow.keras.models import Sequential
    except ImportError as exc:
        raise ImportError("tensorflow is required to build the LSTM model") from exc

    model = Sequential(
        [
            LSTM(64, return_sequences=True, input_shape=(timesteps, feature_count)),
            LSTM(32),
            Dense(1, activation="linear"),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def make_lstm_features(price_data: Any, sentiment_scores: Any) -> np.ndarray:
    """Combine price and sentiment tensors for LSTM prediction."""

    price_features = _to_3d_float_array(price_data, "price_data")
    sentiment_features = _to_3d_float_array(sentiment_scores, "sentiment_scores")

    if price_features.shape[:2] != sentiment_features.shape[:2]:
        raise ValueError(
            "price_data and sentiment_scores must share sample and timestep dimensions"
        )

    return np.concatenate([price_features, sentiment_features], axis=2)


def predict_momentum(
    price_data: Any,
    sentiment_scores: Any,
    model: Optional[Any] = None,
) -> float:
    """Predict the next-period return using price action and sentiment.

    Injected models may be Keras/sklearn-style objects with ``predict()``,
    PyTorch ``nn.Module`` instances, or plain callables that accept a 3D numpy
    array shaped as ``(batch, timesteps, features)``.
    """

    features = make_lstm_features(price_data, sentiment_scores)
    prediction_model = model or build_lstm_model(features.shape[1], features.shape[2])
    prediction = _predict_model_output(prediction_model, features[-1:])

    if prediction.size != 1:
        raise ValueError("momentum prediction must contain exactly one value")

    return float(prediction[0])


def generate_pairs_trade_signal(
    stock_a: Any,
    stock_b: Any,
    regime: Any,
    lstm_prediction: Any,
    *,
    window: int = 30,
    entry_z: float = -2.0,
    exit_z: float = 0.5,
    stock_a_name: str = "stock_A",
    stock_b_name: str = "stock_B",
) -> TradeInstruction:
    """Return the pair-trade instruction implied by the current inputs."""

    if window < 2:
        raise ValueError("window must be at least 2")

    close_a = _extract_close_series(stock_a, "stock_a")
    close_b = _extract_close_series(stock_b, "stock_b")
    prices = pd.concat(
        [close_a.rename("stock_a"), close_b.rename("stock_b")],
        axis=1,
    ).dropna()

    if len(prices) < window:
        raise ValueError(f"at least {window} aligned close prices are required")

    spread = prices["stock_a"] - prices["stock_b"]
    rolling_mean = spread.rolling(window).mean()
    rolling_std = spread.rolling(window).std()
    z_score = (spread - rolling_mean) / rolling_std
    current_z = float(z_score.iloc[-1])
    regime_value = _to_scalar_int(regime, "regime")
    predicted_return = _to_scalar_float(lstm_prediction, "lstm_prediction")

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
        and current_z < entry_z
        and predicted_return > 0.0
    ):
        return TradeInstruction(
            action="OPEN_PAIR",
            reason="disinflation regime, negative spread z-score, positive momentum",
            z_score=current_z,
            regime=regime_value,
            predicted_return=predicted_return,
            long_leg=stock_a_name,
            short_leg=stock_b_name,
        )

    if abs(current_z) < exit_z:
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


def execute_pairs_trade(
    stock_a: Any,
    stock_b: Any,
    regime: Any,
    lstm_prediction: Any,
    *,
    order_executor: Optional[OrderExecutor] = None,
    position_closer: Optional[PositionCloser] = None,
    window: int = 30,
    entry_z: float = -2.0,
    exit_z: float = 0.5,
    stock_a_name: str = "stock_A",
    stock_b_name: str = "stock_B",
) -> TradeInstruction:
    """Generate a pair-trade instruction and run optional execution callbacks."""

    instruction = generate_pairs_trade_signal(
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

    if instruction.action == "OPEN_PAIR" and order_executor is not None:
        order_executor("BUY", stock_a, hedge="SHORT", hedge_asset=stock_b)
    elif instruction.action == "CLOSE" and position_closer is not None:
        position_closer()

    return instruction


def _build_hmm_model(
    *,
    n_components: int,
    covariance_type: str,
    n_iter: int,
    random_state: Optional[int],
) -> Any:
    try:
        from hmmlearn import hmm
    except ImportError as exc:
        raise ImportError("hmmlearn is required to classify macro regimes") from exc

    return hmm.GaussianHMM(
        n_components=n_components,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
    )


def _predict_model_output(model: Any, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict"):
        try:
            prediction = model.predict(features, verbose=0)
        except TypeError:
            prediction = model.predict(features)
        return np.asarray(prediction).reshape(-1)

    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None and isinstance(model, torch.nn.Module):
        model.eval()
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
        with torch.no_grad():
            prediction = model(tensor)
        return prediction.detach().cpu().numpy().reshape(-1)

    if callable(model):
        return np.asarray(model(features)).reshape(-1)

    raise TypeError("model must define predict(), be a torch nn.Module, or be callable")


def _to_2d_float_array(data: Any, name: str) -> np.ndarray:
    values = _to_numpy(data, name)
    if values.ndim == 1:
        values = values.reshape(-1, 1)

    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty 2D array or dataframe")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")

    return values


def _to_3d_float_array(data: Any, name: str) -> np.ndarray:
    values = _to_numpy(data, name)
    if values.ndim == 1:
        values = values.reshape(1, values.shape[0], 1)
    elif values.ndim == 2:
        values = values.reshape(1, values.shape[0], values.shape[1])

    if values.ndim != 3 or 0 in values.shape:
        raise ValueError(f"{name} must be a non-empty 1D, 2D, or 3D numeric input")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")

    return values


def _to_numpy(data: Any, name: str) -> np.ndarray:
    try:
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.to_numpy(dtype=float)
        return np.asarray(data, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc


def _extract_close_series(stock: Any, name: str) -> pd.Series:
    if isinstance(stock, pd.Series):
        close = stock
    else:
        try:
            close = stock["close"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{name} must be a Series or include a 'close' column") from exc

    try:
        series = pd.Series(close, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} close prices must be numeric") from exc

    if series.empty:
        raise ValueError(f"{name} close prices cannot be empty")

    return series


def _to_scalar_float(value: Any, name: str) -> float:
    values = np.asarray(value, dtype=float).reshape(-1)
    if values.size != 1:
        raise ValueError(f"{name} must contain exactly one value")
    return float(values[0])


def _to_scalar_int(value: Any, name: str) -> int:
    values = np.asarray(value, dtype=int).reshape(-1)
    if values.size != 1:
        raise ValueError(f"{name} must contain exactly one value")
    return int(values[0])
