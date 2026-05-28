"""Training utilities from trainModel_pseudocode.md."""

from __future__ import annotations

import copy
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


REQUIRED_FEATURE_INPUT_COLUMNS = (
    "BIST100",
    "USD_TRY",
    "CBRT_Rate",
    "CPI",
    "5Y_CDS_Spread",
    "advancing_stocks",
    "declining_stocks",
)


def engineer_turkish_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create Turkish macro and market features for model training."""

    _require_columns(df, REQUIRED_FEATURE_INPUT_COLUMNS)
    features = df.copy()

    features["bist_ret"] = features["BIST100"].pct_change()
    features["fx_ret"] = features["USD_TRY"].pct_change()
    features["target"] = features["bist_ret"] - features["fx_ret"]
    features["real_rate"] = features["CBRT_Rate"] - features["CPI"]
    features["cds_velocity"] = features["5Y_CDS_Spread"].diff()
    features["fx_volatility"] = features["USD_TRY"].rolling(14).std()
    features["market_breadth"] = (
        features["advancing_stocks"] / features["declining_stocks"]
    )

    features = features.replace([np.inf, -np.inf], np.nan)
    return features.dropna()


def generate_regime_weights(
    df: pd.DataFrame,
    model: Optional[Any] = None,
    *,
    n_components: int = 4,
    covariance_type: str = "diag",
    n_iter: int = 200,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Assign inverse-frequency sample weights from HMM market regimes."""

    _require_columns(df, ("bist_ret", "fx_volatility"))
    weighted = df.copy()
    x_hmm = weighted[["bist_ret", "fx_volatility"]].to_numpy(dtype=float)
    if len(x_hmm) == 0:
        raise ValueError("df must contain at least one row")

    hmm_model = model or _build_hmm_model(
        n_components=n_components,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
    )
    hmm_model.fit(x_hmm)

    regimes = np.asarray(hmm_model.predict(x_hmm)).reshape(-1)
    if len(regimes) != len(weighted):
        raise ValueError("HMM must return one regime per input row")

    weighted["regime"] = regimes.astype(int)
    regime_counts = weighted["regime"].value_counts(normalize=True)
    inverse_frequency = 1.0 / regime_counts
    weighted["sample_weight"] = weighted["regime"].map(inverse_frequency)
    weighted["sample_weight"] = (
        weighted["sample_weight"] / weighted["sample_weight"].max()
    )

    return weighted


class BISTResilientLSTM(nn.Module):
    """Attention-based LSTM for next-period BIST relative-return prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(lstm_out * attention_weights, dim=1)
        return self.fc(context_vector)


BIST_ResilientLSTM = BISTResilientLSTM


class RegimeWeightedHuberLoss(nn.Module):
    """Huber loss multiplied by per-sample regime weights."""

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta, reduction="none")

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        predictions = predictions.reshape(-1, 1)
        targets = targets.reshape(-1, 1)
        weights = weights.reshape(-1, 1)
        if predictions.shape != targets.shape or predictions.shape != weights.shape:
            raise ValueError("predictions, targets, and weights must align by sample")

        loss = self.huber(predictions, targets)
        return (loss * weights).mean()


def train_bist_model(
    model: nn.Module,
    train_loader: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    val_loader: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    epochs: int = 100,
    *,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    max_grad_norm: float = 1.0,
    device: Optional[torch.device | str] = None,
    early_stopping_patience: Optional[int] = None,
    min_delta: float = 0.0,
    restore_best_state: bool = False,
    verbose: bool = True,
) -> list[dict[str, float]]:
    """Train the BIST model with weighted Huber loss and gradient clipping."""

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if early_stopping_patience is not None and early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be at least 1")
    if min_delta < 0.0:
        raise ValueError("min_delta cannot be negative")

    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device_obj)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = RegimeWeightedHuberLoss(delta=1.5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_state: Optional[dict[str, torch.Tensor]] = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        sample_count = 0

        for batch_x, batch_y, batch_weights in train_loader:
            batch_x = batch_x.to(device_obj)
            batch_y = batch_y.to(device_obj)
            batch_weights = batch_weights.to(device_obj)

            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y, batch_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            batch_size = int(batch_x.shape[0])
            train_loss += float(loss.item()) * batch_size
            sample_count += batch_size

        if sample_count == 0:
            raise ValueError("train_loader did not yield any samples")

        train_loss /= sample_count
        val_loss = evaluate_model(model, val_loader, criterion, device=device_obj)
        scheduler.step(val_loss)

        epoch_metrics = {
            "epoch": float(epoch + 1),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_metrics)

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            if restore_best_state:
                best_state = copy.deepcopy(model.state_dict())
        else:
            epochs_without_improvement += 1

        if verbose:
            print(
                f"Epoch {epoch + 1} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )

        if (
            early_stopping_patience is not None
            and epochs_without_improvement >= early_stopping_patience
        ):
            if verbose:
                print(
                    "Early stopping: "
                    f"validation loss did not improve for {early_stopping_patience} epochs"
                )
            break

    if restore_best_state and best_state is not None:
        model.load_state_dict(best_state)

    return history


def evaluate_model(
    model: nn.Module,
    data_loader: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    criterion: RegimeWeightedHuberLoss,
    *,
    device: Optional[torch.device | str] = None,
) -> float:
    """Evaluate weighted loss for a loader of (X, y, sample_weight) batches."""

    device_obj = torch.device(device or next(model.parameters()).device)
    model.eval()
    total_loss = 0.0
    sample_count = 0

    with torch.no_grad():
        for batch_x, batch_y, batch_weights in data_loader:
            batch_x = batch_x.to(device_obj)
            batch_y = batch_y.to(device_obj)
            batch_weights = batch_weights.to(device_obj)
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y, batch_weights)

            batch_size = int(batch_x.shape[0])
            total_loss += float(loss.item()) * batch_size
            sample_count += batch_size

    if sample_count == 0:
        raise ValueError("data_loader did not yield any samples")

    return total_loss / sample_count


def create_purged_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    embargo_days: int = 20,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create expanding walk-forward folds with an embargo before each test split."""

    if n_splits < 1:
        raise ValueError("n_splits must be at least 1")
    if embargo_days < 0:
        raise ValueError("embargo_days cannot be negative")

    n_rows = len(df)
    if n_rows <= n_splits:
        raise ValueError("df must contain more rows than n_splits")

    indices = np.arange(n_rows)
    test_size = n_rows // (n_splits + 1)
    if test_size <= 0:
        raise ValueError("not enough rows for the requested number of splits")

    folds: list[tuple[np.ndarray, np.ndarray]] = []

    for i in range(n_splits):
        test_start = (i + 1) * test_size
        test_end = test_start + test_size
        train_end = test_start - embargo_days

        if train_end <= 0:
            continue

        test_indices = indices[test_start:test_end]
        if len(test_indices) == 0:
            continue

        folds.append((indices[:train_end], test_indices))

    if not folds:
        raise ValueError("no valid folds were produced; reduce n_splits or embargo_days")

    return folds


def _build_hmm_model(
    *,
    n_components: int,
    covariance_type: str,
    n_iter: int,
    random_state: Optional[int],
) -> Any:
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError as exc:
        raise ImportError("hmmlearn is required to generate regime weights") from exc

    return GaussianHMM(
        n_components=n_components,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
    )


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"df is missing required columns: {', '.join(missing)}")
