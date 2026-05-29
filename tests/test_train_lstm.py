import unittest

import numpy as np
import pandas as pd

from initial_trd.cli.train_lstm import build_sequence_arrays


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
                    [20.0, 21.0],
                    [21.0, 22.0],
                ],
                dtype=np.float32,
            ),
        )
        np.testing.assert_array_equal(
            y.reshape(-1),
            np.asarray([3.0, 4.0, 22.0, 23.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(weights, np.ones(4, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
