# Initial TRD

Initial TRD is a research-oriented Turkish market trading strategy prototype.
It provides feature engineering, regime-weighted LSTM training, purged
walk-forward evaluation, and pair-trade signal generation.

The project is source-only. Runtime data, trained models, and result files are
generated locally under `data/` and `artifacts/` when you run the CLI.

## Project Layout

```text
src/initial_trd/
  strategy.py       Pair-trade and prediction helpers
  training.py       Feature engineering, model, loss, and fold utilities
  evaluation.py     Walk-forward evaluation and performance metrics
  data_fetch.py     Market-data fetch and alignment workflow
  paths.py          Working-directory-relative data/artifact paths
  cli/              Console command implementations
tests/              Unit tests
docs/design_notes.md
```

## Setup

From PowerShell:

```powershell
cd C:\Users\colakos\Desktop\initialTRD
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Optional dependencies:

```powershell
python -m pip install -e ".[hmm]"
python -m pip install -e ".[sklearn]"
python -m pip install -e ".[tensorflow]"
```

Use `hmmlearn` for real HMM regime classification and sample weighting.
Use `scikit-learn` for the optional linear, ridge, and random-forest
walk-forward models. Use `tensorflow` only for `build_lstm_model(...)` in
`initial_trd.strategy`.

## CLI Usage

All relative paths are resolved from the current working directory.

Fetch and align raw market data:

```powershell
trd-fetch-data
```

Create engineered features:

```powershell
trd-engineer-features
```

Create engineered features with HMM sample weights:

```powershell
trd-engineer-features --with-regime-weights --random-state 7
```

Train the PyTorch LSTM:

```powershell
trd-train-lstm --epochs 20 --device cpu
OR
trd-train-lstm --epochs 20 --device cuda
```

Find the best epoch count by validation loss:

```powershell
trd-tune-epochs --max-epochs 100 --patience 10 --device cuda
```

This writes `artifacts/epoch_tuning.csv` and `artifacts/epoch_tuning.json`.
Use the reported best epoch for the next `trd-train-lstm --epochs ...` run.

Run the full default pipeline:

```powershell
trd-run-pipeline --epochs 66 --device cuda
```

The pipeline runs fetch, weighted feature engineering, LSTM training,
ridge walk-forward validation, and model-based signal generation. It writes
`artifacts/pipeline_summary.json` with the pair, output paths, average
walk-forward metrics, prediction, and final signal action.

The default pair is `THYAO.IS` vs `PGSUS.IS`. Change it like this:

```powershell
trd-run-pipeline `
  --stock-a-ticker ASELS.IS `
  --stock-b-ticker THYAO.IS `
  --stock-a-name ASELS `
  --stock-b-name THYAO `
  --epochs 66 `
  --device cuda
```

`--hmm-random-state 7` and `--fetch-random-state 7` are reproducibility seeds.
They make repeated runs comparable; they are not known-optimal values.

Run dependency-free walk-forward validation:

```powershell
trd-walk-forward
```

Run walk-forward validation with a ridge model:

```powershell
trd-walk-forward --model ridge
```

Generate a pair-trade instruction from a manual prediction:

```powershell
trd-signal --prediction 0.015
```

Generate a pair-trade instruction from the saved LSTM:

```powershell
trd-signal --device cpu
```

## Public Imports

```python
from initial_trd.strategy import execute_pairs_trade
from initial_trd.training import BISTResilientLSTM, engineer_turkish_features
from initial_trd.evaluation import run_walk_forward_test
```

Backward-compatible root imports such as `from train_model import ...` are not
preserved.

## Data Contracts

Raw macro/market input for `engineer_turkish_features(...)` must include:

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

Pair-trade inputs can be pandas Series of close prices or dataframes with a
`close` column.

## Verification

```powershell
python -m unittest -v
python -m compileall src tests
trd-fetch-data --help
trd-engineer-features --help
trd-train-lstm --help
trd-tune-epochs --help
trd-walk-forward --help
trd-signal --help
trd-run-pipeline --help
```
