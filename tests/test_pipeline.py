import unittest

import pandas as pd

from initial_trd.cli.pipeline import (
    annotate_walk_forward_results,
    build_walk_forward_evaluation_summary,
)


class PipelineSummaryTests(unittest.TestCase):
    def test_walk_forward_artifacts_mark_benchmark_not_saved_lstm(self):
        results = pd.DataFrame(
            {
                "fold": [0.0, 1.0],
                "train_size": [20.0, 30.0],
                "test_size": [10.0, 10.0],
                "test_start": [20.0, 30.0],
                "test_end": [29.0, 39.0],
                "test_sharpe": [1.0, 3.0],
                "test_max_dd": [-0.2, -0.1],
                "rmse": [0.1, 0.2],
                "directional_accuracy": [0.4, 0.6],
                "mean_strategy_return": [0.01, 0.03],
            }
        )

        annotated = annotate_walk_forward_results(results, model_name="ridge")
        summary = build_walk_forward_evaluation_summary(
            annotated,
            model_name="ridge",
        )

        self.assertEqual(annotated["evaluation_target"].tolist(), ["benchmark_model"] * 2)
        self.assertEqual(annotated["model_name"].tolist(), ["ridge"] * 2)
        self.assertFalse(bool(annotated["evaluates_saved_lstm"].any()))
        self.assertEqual(summary["evaluation_target"], "benchmark_model")
        self.assertEqual(summary["model_name"], "ridge")
        self.assertFalse(summary["evaluates_saved_lstm"])
        self.assertIn("not the saved LSTM", summary["note"])
        self.assertAlmostEqual(summary["average_metrics"]["rmse"], 0.15)
        self.assertAlmostEqual(summary["average_metrics"]["test_sharpe"], 2.0)


if __name__ == "__main__":
    unittest.main()
