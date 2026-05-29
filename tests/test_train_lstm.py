import unittest

import numpy as np
import pandas as pd

from initial_trd.cli.train_lstm import build_loaders, build_sequence_arrays


class FitTrackingHMM:
    def __init__(self):
        self.fit_data = None

    def fit(self, data):
        self.fit_data = np.asarray(data, dtype=float)
        return self

    def predict(self, data):
        values = np.asarray(data, dtype=float)
        return (values[:, 0] > 0.0).astype(int)


class BuildSequenceArraysTests(unittest.TestCase):
    def test_build_sequence_arrays_does_not_cross_date_gaps_after_dropna(self):
        dates = pd.date_range("2026-01-01", "2026-01-23", freq="D")
        values = [
            1.0,
            2.0,
            3.0,
            4.0,
            *([np.nan] * 15),
            20.0,
            21.0,
            22.0,
            23.0,
        ]
        df = pd.DataFrame(
            {
                "date": dates,
                "feature": values,
                "target": values,
            }
        )

        x, y, weights = build_sequence_arrays(
            df,
            feature_columns=["feature"],
            target_column="target",
            weight_column="sample_weight",
            sequence_length=2,
        )

        np.testing.assert_array_equal(
            x.reshape(-1, 2),
            np.asarray(
                [
                    [1.0, 2.0],
                    [2.0, 3.0],
                    [3.0, 4.0],
                    [20.0, 21.0],
                    [21.0, 22.0],
                    [22.0, 23.0],
                ],
                dtype=np.float32,
            ),
        )
        np.testing.assert_array_equal(
            y.reshape(-1),
            np.asarray([2.0, 3.0, 4.0, 21.0, 22.0, 23.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(weights, np.ones(6, dtype=np.float32))

    def test_split_safe_regime_weights_fit_only_training_rows(self):
        current_rows = np.asarray(
            [
                [-0.05, 0.8],
                [-0.03, 0.7],
                [0.04, 0.1],
                [0.06, 0.2],
                [-0.02, 0.9],
                [0.01, 0.4],
            ],
            dtype=np.float32,
        )
        x = np.zeros((6, 2, 2), dtype=np.float32)
        x[:, -1, :] = current_rows
        y = np.zeros((6, 1), dtype=np.float32)
        weights = np.ones(6, dtype=np.float32)
        model = FitTrackingHMM()

        build_loaders(
            x,
            y,
            weights,
            validation_size=0.5,
            batch_size=2,
            feature_columns=["bist_ret", "fx_volatility"],
            use_regime_weights=True,
            regime_model=model,
        )

        np.testing.assert_allclose(model.fit_data, current_rows[:3])


if __name__ == "__main__":
    unittest.main()
