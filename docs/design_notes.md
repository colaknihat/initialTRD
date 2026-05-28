# Design Notes

Initial TRD started from three research sketches: strategy logic, LSTM training,
and walk-forward testing. This note keeps the useful intent without preserving
duplicate pseudocode beside the implementation.

## Strategy Intent

- Classify the current macro regime from inflation, rates, and USD/TRY with a
  Gaussian HMM.
- Predict next-period momentum from market price features plus optional
  sentiment features.
- Use the momentum prediction as a filter for pair-trade entry.
- Open a pair trade only when the current spread z-score and macro regime agree
  with the prediction.
- Close positions when the spread mean-reverts toward the exit threshold.

## Training Intent

- Engineer Turkey-specific macro features from BIST 100, USD/TRY, CBRT rate,
  CPI, CDS spread, and market breadth inputs.
- Use FX-adjusted return as the target so the model is measured against holding
  USD/TRY, not only nominal lira gains.
- Use an HMM over return and volatility features to assign inverse-frequency
  regime weights.
- Train an attention-based PyTorch LSTM with Huber loss, sample weights,
  AdamW, validation, learning-rate scheduling, and gradient clipping.

## Evaluation Intent

- Use purged walk-forward validation instead of random K-fold splits for
  financial time series.
- Insert an embargo between train and test windows to reduce leakage from
  slow-moving macro releases.
- Train a fresh model for each fold and report out-of-sample Sharpe, maximum
  drawdown, RMSE, directional accuracy, and mean strategy return.
- Calculate strategy returns from prediction direction times FX-adjusted
  realized returns.
