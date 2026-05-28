import unittest

from initial_trd.cli.tune_epochs import summarize_history


class TuneEpochsTests(unittest.TestCase):
    def test_summarize_history_selects_lowest_validation_loss(self):
        history = [
            {"epoch": 1.0, "train_loss": 0.30, "val_loss": 0.40},
            {"epoch": 2.0, "train_loss": 0.20, "val_loss": 0.25},
            {"epoch": 3.0, "train_loss": 0.10, "val_loss": 0.35},
        ]

        summary = summarize_history(history, requested_max_epochs=5)

        self.assertEqual(summary["best_epoch"], 2)
        self.assertEqual(summary["final_epoch"], 3)
        self.assertEqual(summary["epochs_after_best"], 1)
        self.assertTrue(summary["stopped_early"])

    def test_summarize_history_rejects_empty_history(self):
        with self.assertRaises(ValueError):
            summarize_history([], requested_max_epochs=5)


if __name__ == "__main__":
    unittest.main()
