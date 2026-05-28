"""Shared file locations for command-line workflows.

Relative paths are resolved from the process working directory so an installed
CLI writes data and artifacts into the active workspace, not the package.
"""

from __future__ import annotations

from pathlib import Path


DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("artifacts")

RAW_MARKET_PATH = DATA_DIR / "raw_market.csv"
STOCK_A_PATH = DATA_DIR / "stock_a.csv"
STOCK_B_PATH = DATA_DIR / "stock_b.csv"

FEATURES_PATH = ARTIFACTS_DIR / "features.csv"
WEIGHTED_FEATURES_PATH = ARTIFACTS_DIR / "features_weighted.csv"
MODEL_PATH = ARTIFACTS_DIR / "bist_lstm.pt"
WALK_FORWARD_RESULTS_PATH = ARTIFACTS_DIR / "walk_forward_results.csv"
STRATEGY_SIGNAL_PATH = ARTIFACTS_DIR / "strategy_signal.json"
EPOCH_TUNING_PATH = ARTIFACTS_DIR / "epoch_tuning.csv"
EPOCH_TUNING_SUMMARY_PATH = ARTIFACTS_DIR / "epoch_tuning.json"
PIPELINE_SUMMARY_PATH = ARTIFACTS_DIR / "pipeline_summary.json"


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def first_existing_path(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]
