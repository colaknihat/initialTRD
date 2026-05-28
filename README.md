# Initial TRD

Initial TRD is a research-oriented trading strategy prototype built from the
three pseudocode files in this workspace:

- `pseudocode.md`: macro regime classification, momentum prediction, and pair-trade execution logic.
- `trainModel_pseudocode.md`: Turkish macro feature engineering, regime-weighted LSTM training, and purged walk-forward validation.
- `test.md`: out-of-sample walk-forward testing and strategy performance metrics.

The implementation is intentionally modular. It does not place real broker
orders, fetch live data, or include production risk controls. Broker actions
are exposed as optional callbacks so the strategy can be tested without side
effects.

## Project Structure

`trading_strategy.py`

Core strategy logic:

- `classify_regime(...)`: fits an HMM on macro data and returns the latest regime.
- `make_lstm_features(...)`: combines price and sentiment tensors into LSTM input.
- `predict_momentum(...)`: predicts the next-period return. It accepts Keras/sklearn-style models with `predict()`, PyTorch `nn.Module` models, or plain callables.
- `generate_pairs_trade_signal(...)`: returns an explicit `TradeInstruction`.
- `execute_pairs_trade(...)`: returns the same instruction and optionally calls execution callbacks.

`train_model.py`

Training utilities:

- `engineer_turkish_features(...)`: builds BIST, FX, real-rate, CDS velocity, FX volatility, and market breadth features.
- `generate_regime_weights(...)`: assigns inverse-frequency weights from HMM regimes.
- `BISTResilientLSTM`: attention-based PyTorch LSTM.
- `RegimeWeightedHuberLoss`: Huber loss multiplied by per-sample regime weights.
- `train_bist_model(...)`: PyTorch training loop with AdamW, validation, scheduler, and gradient clipping.
- `evaluate_model(...)`: validation helper for PyTorch loaders.
- `create_purged_folds(...)`: expanding walk-forward folds with an embargo period.

`model_testing.py`

Out-of-sample evaluation utilities:

- `PurgedWalkForward`: splitter matching `create_purged_folds(...)`.
- `run_walk_forward_test(...)`: trains a fresh model per fold and reports performance.
- `calculate_strategy_returns(...)`: converts prediction direction into FX-adjusted strategy returns.
- `calculate_sharpe(...)`, `calculate_max_drawdown(...)`, `calculate_rmse(...)`, and `calculate_directional_accuracy(...)`.

Unit tests:

- `test_trading_strategy.py`
- `test_train_model.py`
- `test_model_testing.py`

Execution scripts:

- `scripts/run_feature_engineering.py`: reads a raw market CSV and writes engineered features.
- `scripts/run_lstm_training.py`: trains and saves the PyTorch LSTM model.
- `scripts/run_walk_forward_test.py`: runs purged walk-forward validation.
- `scripts/run_strategy_signal.py`: generates a pair-trade instruction from stock close prices and either a supplied prediction or a saved LSTM model.

## Data Contracts

Raw macro/market dataframe for `engineer_turkish_features(...)` must include:

```text
BIST100
USD_TRY
CBRT_Rate
CPI
5Y_CDS_Spread
advancing_stocks
declining_stocks
```

The engineered dataframe includes:

```text
bist_ret
fx_ret
target
real_rate
cds_velocity
fx_volatility
market_breadth
```

`target` is BIST return minus USD/TRY return, so model performance is measured
against holding dollars rather than nominal Turkish lira gains.

PyTorch model training expects batches shaped as:

```text
batch_X:       (batch, timesteps, features)
batch_y:       (batch, 1) or (batch,)
batch_weights: (batch,)
```

Pair trading inputs can be either pandas Series of close prices or dataframes
with a `close` column.

## Compatibility Notes

- `trading_strategy.predict_momentum(...)` now works with the PyTorch model type created in `train_model.py`, as well as Keras/sklearn-style models.
- `train_model.create_purged_folds(...)` and `model_testing.PurgedWalkForward` now use the same expanding split logic.
- HMM functions require `hmmlearn` unless you inject a compatible model with `fit(...)` and `predict(...)`.
- `build_lstm_model(...)` in `trading_strategy.py` uses TensorFlow only when no prediction model is supplied.
- The strategy regime enum has three macro regimes, while training sample weights use four market regimes. These are separate modeling layers.

## How To Run

From PowerShell:

```powershell
cd C:\Users\colakos\Desktop\initialTRD
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy pandas torch
```

Optional dependencies:

```powershell
python -m pip install hmmlearn tensorflow scikit-learn
```

Use `hmmlearn` for real HMM regime classification and sample weighting.
Use `tensorflow` only if you want `trading_strategy.build_lstm_model(...)`.
Use `scikit-learn` only if you choose sklearn models for walk-forward testing.

Run the full test suite:

```powershell
python -m unittest -v
```

Run a syntax/bytecode check:

```powershell
python -m compileall .
```

## Execution Scripts

The scripts are intended to be run from the project root. Relative paths are
resolved from this directory, and each script has defaults that match the
previous script's output:

```powershell
cd C:\Users\colakos\Desktop\initialTRD
```

Fetch and align raw market data:

```powershell
python fetch_and_align_data.py
```

Create engineered features:

```powershell
python scripts/run_feature_engineering.py
```

Create engineered features with HMM sample weights:

```powershell
python scripts/run_feature_engineering.py --with-regime-weights --random-state 7
```

Train the PyTorch LSTM:

```powershell
python scripts/run_lstm_training.py `
  --epochs 20 `
  --device cpu
```

Run dependency-free walk-forward validation:

```powershell
python scripts/run_walk_forward_test.py
```

Run walk-forward validation with a ridge model:

```powershell
python scripts/run_walk_forward_test.py --model ridge
```

Generate a pair-trade instruction using a manual prediction:

```powershell
python scripts/run_strategy_signal.py --prediction 0.015
```

Generate a pair-trade instruction using the saved LSTM:

```powershell
python scripts/run_strategy_signal.py --device cpu
```

`stock_a.csv` and `stock_b.csv` must include a `close` column. The LSTM signal
script uses the latest `sequence_length` rows from the engineered feature CSV
and the feature list saved in the model checkpoint.

## Basic Usage

Feature engineering:

```python
import pandas as pd
from train_model import engineer_turkish_features

raw_df = pd.read_csv("market_data.csv")
features_df = engineer_turkish_features(raw_df)
```

Regime weighting:

```python
from train_model import generate_regime_weights

weighted_df = generate_regime_weights(features_df)
```

Training a PyTorch model:

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from train_model import BISTResilientLSTM, train_bist_model

feature_count = 6
model = BISTResilientLSTM(input_dim=feature_count)

dataset = TensorDataset(batch_X, batch_y, batch_weights)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

history = train_bist_model(
    model,
    train_loader=loader,
    val_loader=loader,
    epochs=20,
    device="cpu",
)
```

Walk-forward evaluation with a model that provides `fit()` and `predict()`:

```python
from model_testing import run_walk_forward_test

results = run_walk_forward_test(
    features_df,
    model_factory=MyModel,
    features=["real_rate", "cds_velocity", "fx_volatility", "market_breadth"],
    target="target",
    n_splits=5,
    embargo_days=15,
)
print(results)
```

Generating a pair-trade instruction:

```python
from trading_strategy import execute_pairs_trade

instruction = execute_pairs_trade(
    stock_a,
    stock_b,
    regime=1,
    lstm_prediction=0.015,
)
print(instruction)
```

To place orders, pass callbacks:

```python
instruction = execute_pairs_trade(
    stock_a,
    stock_b,
    regime=1,
    lstm_prediction=0.015,
    order_executor=my_order_executor,
    position_closer=my_position_closer,
)
```

## Verification Status

Current local verification:

```text
python -m unittest -v
15 tests passed

python -m compileall .
passed
```
