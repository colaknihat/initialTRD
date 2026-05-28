import unittest

import numpy as np
import pandas as pd

from initial_trd.evaluation import (
    PurgedWalkForward,
    calculate_directional_accuracy,
    calculate_max_drawdown,
    calculate_rmse,
    calculate_sharpe,
    calculate_strategy_returns,
    run_walk_forward_test,
)


class MeanReturnModel:
    def fit(self, x, y):
        self.prediction = float(np.mean(y))
        return self

    def predict(self, x):
        return np.full(len(x), self.prediction)


class ModelTestingTests(unittest.TestCase):
    def test_purged_walk_forward_split_skips_embargo_period(self):
        splitter = PurgedWalkForward(n_splits=3, embargo_days=2)

        folds = list(splitter.split(np.arange(30)))

        self.assertEqual(len(folds), 3)
        np.testing.assert_array_equal(folds[0][0], np.arange(0, 5))
        np.testing.assert_array_equal(folds[0][1], np.arange(7, 14))
        for train_idx, test_idx in folds:
            self.assertLess(train_idx[-1], test_idx[0] - 1)

    def test_calculate_strategy_returns_uses_prediction_direction(self):
        predictions = np.array([0.2, -0.1, 0.0])
        actuals = np.array([0.03, -0.04, 0.08])

        returns = calculate_strategy_returns(predictions, actuals)

        np.testing.assert_allclose(returns, np.array([0.03, 0.04, 0.0]))

    def test_directional_accuracy_counts_matching_signs(self):
        predictions = np.array([0.2, -0.1, 0.0, 0.3])
        actuals = np.array([0.03, 0.04, 0.0, -0.01])

        accuracy = calculate_directional_accuracy(predictions, actuals)

        self.assertEqual(accuracy, 0.5)

    def test_sharpe_rmse_and_max_drawdown_metrics(self):
        returns = np.array([0.10, -0.20, 0.05])
        expected_sharpe = np.sqrt(252) * np.mean(returns) / np.std(returns, ddof=1)

        self.assertAlmostEqual(calculate_sharpe(returns), expected_sharpe)
        self.assertAlmostEqual(calculate_max_drawdown(returns), -0.20)
        self.assertAlmostEqual(
            calculate_rmse([0.1, 0.2, 0.3], [0.1, 0.0, 0.5]),
            np.sqrt((0.0**2 + 0.2**2 + (-0.2) ** 2) / 3.0),
        )

    def test_run_walk_forward_test_returns_fold_metrics(self):
        df = pd.DataFrame(
            {
                "feature_a": np.linspace(0.0, 1.0, 40),
                "feature_b": np.linspace(1.0, 0.0, 40),
                "target": np.sin(np.linspace(0.0, 3.0, 40)) / 100.0,
            }
        )

        result = run_walk_forward_test(
            df,
            MeanReturnModel,
            features=["feature_a", "feature_b"],
            target="target",
            n_splits=3,
            embargo_days=2,
            verbose=False,
        )

        self.assertEqual(len(result), 3)
        self.assertEqual(
            list(result.columns),
            [
                "fold",
                "train_size",
                "test_size",
                "test_start",
                "test_end",
                "test_sharpe",
                "test_max_dd",
                "rmse",
                "directional_accuracy",
                "mean_strategy_return",
            ],
        )
        self.assertTrue(np.isfinite(result["rmse"]).all())
        self.assertTrue(
            (
                (0.0 <= result["directional_accuracy"])
                & (result["directional_accuracy"] <= 1.0)
            ).all()
        )


if __name__ == "__main__":
    unittest.main()
