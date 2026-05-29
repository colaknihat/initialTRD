import unittest

import numpy as np
import pandas as pd
import torch

from initial_trd.strategy import (
    Regime,
    classify_regime,
    execute_pairs_trade,
    generate_pairs_trade_signal,
    predict_momentum,
)


class FakeHMM:
    def __init__(self, states):
        self.states = np.asarray(states)
        self.fit_data = None
        self.predict_data = None

    def fit(self, data):
        self.fit_data = data
        return self

    def predict(self, data):
        self.predict_data = data
        return self.states


class FakeMomentumModel:
    def __init__(self, prediction):
        self.prediction = prediction
        self.seen_features = None

    def predict(self, features, **kwargs):
        self.seen_features = features
        return np.array([[self.prediction]])


class TorchMomentumModel(torch.nn.Module):
    def forward(self, features):
        return torch.full((features.shape[0], 1), 0.025, device=features.device)


class TradingStrategyTests(unittest.TestCase):
    def test_classify_regime_fits_history_and_predicts_latest_row(self):
        macro_data = pd.DataFrame(
            {
                "market_return": [-0.05, 0.02, 0.03],
                "volatility": [0.9, 0.1, 0.2],
                "inflation": [52.0, 48.0, 44.0],
            }
        )
        model = FakeHMM([9, 4, 4])

        regime = classify_regime(macro_data, model=model)

        self.assertEqual(regime, Regime.DISINFLATION)
        np.testing.assert_allclose(model.fit_data, macro_data.to_numpy())
        np.testing.assert_allclose(model.predict_data, macro_data.to_numpy())

    def test_predict_momentum_combines_price_and_sentiment_features(self):
        price_data = np.arange(12, dtype=float).reshape(1, 3, 4)
        sentiment_scores = np.ones((1, 3, 2), dtype=float)
        model = FakeMomentumModel(prediction=0.0175)

        prediction = predict_momentum(price_data, sentiment_scores, model=model)

        self.assertEqual(prediction, 0.0175)
        self.assertEqual(model.seen_features.shape, (1, 3, 6))
        np.testing.assert_allclose(model.seen_features[:, :, :4], price_data)
        np.testing.assert_allclose(model.seen_features[:, :, 4:], sentiment_scores)

    def test_predict_momentum_accepts_pytorch_models_from_training_module(self):
        price_data = np.arange(12, dtype=float).reshape(1, 3, 4)
        sentiment_scores = np.ones((1, 3, 2), dtype=float)

        prediction = predict_momentum(
            price_data,
            sentiment_scores,
            model=TorchMomentumModel(),
        )

        self.assertAlmostEqual(prediction, 0.025)

    def test_execute_pairs_trade_opens_pair_when_conditions_match(self):
        stock_a = pd.DataFrame({"close": [101.0] * 29 + [90.0]})
        stock_b = pd.DataFrame({"close": [100.0] * 30})
        calls = []

        def order_executor(side, stock, **kwargs):
            calls.append((side, stock, kwargs))

        instruction = execute_pairs_trade(
            stock_a,
            stock_b,
            Regime.DISINFLATION,
            0.02,
            order_executor=order_executor,
        )

        self.assertEqual(instruction.action, "OPEN_PAIR")
        self.assertEqual(instruction.long_leg, "stock_A")
        self.assertEqual(instruction.short_leg, "stock_B")
        self.assertEqual(len(calls), 1)
        side, stock, kwargs = calls[0]
        self.assertEqual(side, "BUY")
        self.assertIs(stock, stock_a)
        self.assertEqual(kwargs["hedge"], "SHORT")
        self.assertIs(kwargs["hedge_asset"], stock_b)

    def test_execute_pairs_trade_opens_reverse_pair_when_spread_is_high(self):
        stock_a = pd.DataFrame({"close": [99.0] * 29 + [110.0]})
        stock_b = pd.DataFrame({"close": [100.0] * 30})
        calls = []

        def order_executor(side, stock, **kwargs):
            calls.append((side, stock, kwargs))

        instruction = execute_pairs_trade(
            stock_a,
            stock_b,
            Regime.DISINFLATION,
            0.02,
            order_executor=order_executor,
            entry_z=-2.0,
            stock_a_name="A",
            stock_b_name="B",
        )

        self.assertEqual(instruction.action, "OPEN_PAIR")
        self.assertEqual(instruction.long_leg, "B")
        self.assertEqual(instruction.short_leg, "A")
        self.assertEqual(len(calls), 1)
        side, stock, kwargs = calls[0]
        self.assertEqual(side, "BUY")
        self.assertIs(stock, stock_b)
        self.assertEqual(kwargs["hedge"], "SHORT")
        self.assertIs(kwargs["hedge_asset"], stock_a)

    def test_execute_pairs_trade_closes_when_spread_reverts(self):
        stock_a = pd.DataFrame({"close": 100.0 + np.array(list(range(29)) + [14])})
        stock_b = pd.DataFrame({"close": [100.0] * 30})
        closed = []

        instruction = execute_pairs_trade(
            stock_a,
            stock_b,
            Regime.CRISIS,
            -0.01,
            position_closer=lambda: closed.append(True),
        )

        self.assertEqual(instruction.action, "CLOSE")
        self.assertEqual(closed, [True])

    def test_pair_signal_aligns_dataframe_closes_by_date(self):
        stock_a = pd.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
                "close": [100.0, 200.0, 300.0],
            }
        )
        stock_b = pd.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-03"],
                "close": [90.0, 290.0],
            }
        )

        instruction = generate_pairs_trade_signal(
            stock_a,
            stock_b,
            Regime.DISINFLATION,
            0.01,
            window=2,
        )

        self.assertEqual(instruction.action, "HOLD")
        self.assertEqual(
            instruction.reason,
            "z-score is unavailable for the latest window",
        )


if __name__ == "__main__":
    unittest.main()
