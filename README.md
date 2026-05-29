# Initial TRD

Initial TRD is a research-oriented Turkish market trading strategy prototype.
It provides feature engineering, regime-weighted LSTM training, benchmark
purged walk-forward evaluation, and pair-trade signal generation.

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

Fetch and align raw market data. A real Turkey 5Y CDS CSV is required; the
default path is `data/turkey_5y_cds.csv`.

```powershell
trd-fetch-data --cds-csv data\turkey_5y_cds.csv
```

The fetcher reads CPI from the TCMB page that republishes TURKSTAT annual CPI
inflation, reads the CBRT one-week repo policy-rate history from the TCMB page
configured by `--cbrt-rate-url`, and reads 5Y CDS levels from `--cds-csv`. It
does not create synthetic macro data.

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
trd-run-pipeline --cds-csv data\turkey_5y_cds.csv --epochs 66 --device cuda
```

The pipeline runs fetch, weighted feature engineering, LSTM training,
benchmark ridge walk-forward validation, and LSTM-based signal generation. The
walk-forward metrics evaluate the configured benchmark model, not the saved
LSTM that generates the signal. It writes `artifacts/pipeline_summary.json`
with the pair, output paths, benchmark walk-forward metrics, prediction, and
final signal action.

The default pair is `THYAO.IS` vs `PGSUS.IS`. Change it like this:

```powershell
trd-run-pipeline `
  --stock-a-ticker ASELS.IS `
  --stock-b-ticker THYAO.IS `
  --stock-a-name ASELS `
  --stock-b-name THYAO `
  --cds-csv data\turkey_5y_cds.csv `
  --epochs 66 `
  --device cuda
```

`--hmm-random-state 7` is a model reproducibility seed. It makes repeated HMM
weighting runs comparable; it is not a known-optimal value.

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

The CDS CSV may be an Investing.com export with `Date` and `Price` columns, or
a Bloomberg/Refinitiv export with `Date` plus `PX_LAST`, `Close`, or
`5Y_CDS_Spread`.

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
