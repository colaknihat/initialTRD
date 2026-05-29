"""Semantic regime labels for HMM state outputs."""

from __future__ import annotations

from enum import IntEnum
from typing import Any

import numpy as np


class Regime(IntEnum):
    HIGH_INFLATION = 0
    DISINFLATION = 1
    CRISIS = 2


def map_hmm_states_to_regimes(values: Any, states: Any) -> np.ndarray:
    """Map arbitrary HMM state ids onto stable semantic regime values.

    The first feature is treated as returns and the second as volatility.
    Extra HMM components are folded into the nearest available semantic class.
    """

    features = np.asarray(values, dtype=float)
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("values must be a non-empty 2D array")
    if features.shape[1] < 2:
        raise ValueError("values must include return and volatility columns")
    if not np.isfinite(features[:, :2]).all():
        raise ValueError("return and volatility columns must contain only finite values")

    raw_states = np.asarray(states, dtype=int).reshape(-1)
    if len(raw_states) != len(features):
        raise ValueError("HMM must return one state per input row")

    stats = _summarize_states(features[:, 0], features[:, 1], raw_states)
    mapping = _semantic_state_mapping(stats)
    return np.asarray([int(mapping[int(state)]) for state in raw_states], dtype=int)


def _summarize_states(
    returns: np.ndarray,
    volatility: np.ndarray,
    states: np.ndarray,
) -> list[tuple[int, float, float]]:
    stats: list[tuple[int, float, float]] = []
    for state in sorted(np.unique(states)):
        mask = states == state
        stats.append(
            (
                int(state),
                float(np.mean(returns[mask])),
                float(np.mean(volatility[mask])),
            )
        )
    return stats


def _semantic_state_mapping(
    stats: list[tuple[int, float, float]]
) -> dict[int, Regime]:
    mapping = {state: Regime.HIGH_INFLATION for state, _, _ in stats}
    disinflation_state = _pick_disinflation_state(stats)
    mapping[disinflation_state] = Regime.DISINFLATION

    remaining = [stat for stat in stats if stat[0] != disinflation_state]
    if remaining:
        crisis_state = _pick_crisis_state(remaining)
        mapping[crisis_state] = Regime.CRISIS

    return mapping


def _pick_disinflation_state(stats: list[tuple[int, float, float]]) -> int:
    positive_return = [stat for stat in stats if stat[1] > 0.0]
    if positive_return:
        return min(positive_return, key=lambda stat: (stat[2], -stat[1], stat[0]))[0]

    return max(stats, key=lambda stat: (stat[1], -stat[2], -stat[0]))[0]


def _pick_crisis_state(stats: list[tuple[int, float, float]]) -> int:
    negative_return = [stat for stat in stats if stat[1] < 0.0]
    if negative_return:
        return max(negative_return, key=lambda stat: (stat[2], -stat[1], -stat[0]))[0]

    return min(stats, key=lambda stat: (stat[1], -stat[2], stat[0]))[0]
