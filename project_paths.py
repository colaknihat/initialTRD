"""Shared file locations for the local research workflow."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

RAW_MARKET_PATH = DATA_DIR / "raw_market.csv"
STOCK_A_PATH = DATA_DIR / "stock_a.csv"
STOCK_B_PATH = DATA_DIR / "stock_b.csv"

FEATURES_PATH = ARTIFACTS_DIR / "features.csv"
WEIGHTED_FEATURES_PATH = ARTIFACTS_DIR / "features_weighted.csv"
MODEL_PATH = ARTIFACTS_DIR / "bist_lstm.pt"
WALK_FORWARD_RESULTS_PATH = ARTIFACTS_DIR / "walk_forward_results.csv"
STRATEGY_SIGNAL_PATH = ARTIFACTS_DIR / "strategy_signal.json"


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def first_existing_path(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]
