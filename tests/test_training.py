import unittest

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from initial_trd.training import (
    BISTResilientLSTM,
    RegimeWeightedHuberLoss,
    create_purged_folds,
    engineer_turkish_features,
    evaluate_model,
    generate_regime_weights,
    train_bist_model,
)


class FakeHMM:
    def __init__(self, regimes):
        self.regimes = np.asarray(regimes)
        self.fit_data = None

    def fit(self, data):
        self.fit_data = data
        return self

    def predict(self, data):
        return self.regimes


class TrainModelTests(unittest.TestCase):
    def test_engineer_turkish_features_builds_expected_columns(self):
        source = pd.DataFrame(
            {
                "BIST100": np.linspace(1000.0, 1200.0, 20),
                "USD_TRY": np.linspace(30.0, 34.0, 20),
                "CBRT_Rate": [45.0] * 20,
                "CPI": np.linspace(65.0, 50.0, 20),
                "5Y_CDS_Spread": np.linspace(320.0, 280.0, 20),
                "advancing_stocks": np.arange(20, 40),
                "declining_stocks": np.arange(40, 60),
            }
        )

        result = engineer_turkish_features(source)

        self.assertIn("target", result.columns)
        self.assertIn("real_rate", result.columns)
        self.assertIn("cds_velocity", result.columns)
        self.assertIn("fx_volatility", result.columns)
        self.assertIn("market_breadth", result.columns)
        self.assertEqual(len(result), 7)
        self.assertNotIn("target", source.columns)
        row_index = result.index[0]
        expected_target = (
            source.loc[row_index, "BIST100"] / source.loc[row_index - 1, "BIST100"]
            - 1.0
            - (source.loc[row_index, "USD_TRY"] / source.loc[row_index - 1, "USD_TRY"] - 1.0)
        )
        self.assertAlmostEqual(result.loc[row_index, "target"], expected_target)
        self.assertAlmostEqual(
            result.loc[row_index, "real_rate"],
            source.loc[row_index, "CBRT_Rate"] - source.loc[row_index, "CPI"],
        )

    def test_generate_regime_weights_uses_inverse_frequency(self):
        df = pd.DataFrame(
            {
                "bist_ret": [-0.04, -0.03, 0.04, 0.06, 0.01],
                "fx_volatility": [0.8, 0.7, 0.1, 0.6, 0.4],
            }
        )
        model = FakeHMM([7, 7, 3, 5, 9])

        result = generate_regime_weights(df, model=model)

        np.testing.assert_allclose(model.fit_data, df.to_numpy(dtype=float))
        self.assertEqual(result["regime"].tolist(), [2, 2, 1, 0, 0])
        self.assertAlmostEqual(result.loc[0, "sample_weight"], 0.5)
        self.assertAlmostEqual(result.loc[2, "sample_weight"], 1.0)
        self.assertAlmostEqual(result.loc[3, "sample_weight"], 0.5)

    def test_create_purged_folds_creates_expanding_embargoed_splits(self):
        df = pd.DataFrame({"value": range(30)})

        folds = create_purged_folds(df, n_splits=3, embargo_days=2)

        self.assertEqual(len(folds), 3)
        np.testing.assert_array_equal(folds[0][0], np.arange(0, 5))
        np.testing.assert_array_equal(folds[0][1], np.arange(7, 14))
        for train_idx, test_idx in folds:
            self.assertLess(train_idx[-1], test_idx[0] - 1)

    def test_weighted_huber_loss_applies_sample_weights(self):
        criterion = RegimeWeightedHuberLoss(delta=1.0)
        predictions = torch.tensor([[0.0], [2.0]])
        targets = torch.tensor([[0.0], [0.0]])
        weights = torch.tensor([1.0, 0.5])

        loss = criterion(predictions, targets, weights)

        self.assertAlmostEqual(float(loss), 0.375)

    def test_model_training_and_evaluation_return_finite_losses(self):
        torch.manual_seed(7)
        model = BISTResilientLSTM(input_dim=2, hidden_dim=4, num_layers=1)
        x = torch.randn(6, 3, 2)
        y = torch.randn(6, 1)
        weights = torch.ones(6)
        loader = DataLoader(TensorDataset(x, y, weights), batch_size=2)
        criterion = RegimeWeightedHuberLoss(delta=1.5)

        initial_val_loss = evaluate_model(model, loader, criterion, device="cpu")
        history = train_bist_model(
            model,
            loader,
            loader,
            epochs=1,
            device="cpu",
            verbose=False,
        )

        self.assertTrue(np.isfinite(initial_val_loss))
        self.assertEqual(len(history), 1)
        self.assertTrue(np.isfinite(history[0]["train_loss"]))
        self.assertTrue(np.isfinite(history[0]["val_loss"]))


if __name__ == "__main__":
    unittest.main()
