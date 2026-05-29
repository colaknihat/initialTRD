import unittest

import pandas as pd

from initial_trd.backtesting import (
    calculate_backtest_account_values,
    classify_backtest_regimes_by_date,
    generate_symmetric_pair_signal,
    simulate_pair_backtest,
)
from initial_trd.strategy import Regime


class DateAwareFakeHMM:
    def __init__(self):
        self.fit_lengths = []
        self.predict_lengths = []

    def fit(self, data):
        self.fit_lengths.append(len(data))
        return self

    def predict(self, data):
        self.predict_lengths.append(len(data))
        if len(data) == 2:
            return [0, 1]
        return [0, 1, 2]


class PairBacktestingTests(unittest.TestCase):
    def test_symmetric_signal_opens_long_a_short_b_for_negative_z_score(self):
        stock_a = pd.Series([101.0] * 29 + [90.0])
        stock_b = pd.Series([100.0] * 30)

        instruction = generate_symmetric_pair_signal(
            stock_a,
            stock_b,
            Regime.DISINFLATION,
            0.01,
            stock_a_name="A",
            stock_b_name="B",
        )

        self.assertEqual(instruction.action, "OPEN_PAIR")
        self.assertEqual(instruction.long_leg, "A")
        self.assertEqual(instruction.short_leg, "B")

    def test_symmetric_signal_opens_long_b_short_a_for_positive_z_score(self):
        stock_a = pd.Series([99.0] * 29 + [110.0])
        stock_b = pd.Series([100.0] * 30)

        instruction = generate_symmetric_pair_signal(
            stock_a,
            stock_b,
            Regime.DISINFLATION,
            0.01,
            stock_a_name="A",
            stock_b_name="B",
        )

        self.assertEqual(instruction.action, "OPEN_PAIR")
        self.assertEqual(instruction.long_leg, "B")
        self.assertEqual(instruction.short_leg, "A")

    def test_close_while_flat_has_no_trade_or_pnl(self):
        dates = pd.date_range("2026-01-01", periods=5, freq="D")
        closes = pd.DataFrame(
            {
                "A": [100.0, 100.0, 102.0, 101.0, 102.0],
                "B": [100.0, 100.0, 100.0, 100.0, 100.0],
            },
            index=dates,
        )

        result = simulate_pair_backtest(
            closes,
            {dates[-2]: 0.01},
            tickers=["A", "B"],
            days=1,
            window=3,
        )

        row = result.rows.iloc[0]
        self.assertEqual(row["signal_action"], "CLOSE")
        self.assertEqual(row["executed_action"], "HOLD")
        self.assertFalse(row["active_position"])
        self.assertEqual(row["pair_return"], 0.0)
        self.assertEqual(
            result.summary["account_values"]["portfolio_100"]["ending_value"],
            100.0,
        )

    def test_backtest_uses_regime_for_each_signal_date(self):
        dates = pd.date_range("2026-01-01", periods=31, freq="D")
        closes = pd.DataFrame(
            {
                "A": [101.0] * 29 + [90.0, 91.0],
                "B": [100.0] * 31,
            },
            index=dates,
        )

        result = simulate_pair_backtest(
            closes,
            {dates[-2]: 0.01},
            tickers=["A", "B"],
            days=1,
            regime={dates[-2]: Regime.CRISIS},
        )

        row = result.rows.iloc[0]
        self.assertEqual(row["signal_action"], "HOLD")
        self.assertEqual(row["regime"], Regime.CRISIS)
        self.assertEqual(result.summary["trades_opened"], 0)

    def test_classify_backtest_regimes_by_date_uses_history_only(self):
        features = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=3, freq="D"),
                "bist_ret": [-0.05, 0.03, -0.04],
                "fx_volatility": [0.8, 0.1, 0.9],
            }
        )
        models = []

        def model_factory():
            model = DateAwareFakeHMM()
            models.append(model)
            return model

        regimes = classify_backtest_regimes_by_date(
            features,
            features["date"].iloc[1:],
            model_factory=model_factory,
            n_components=2,
        )

        self.assertEqual(regimes[pd.Timestamp("2026-01-02")], Regime.DISINFLATION)
        self.assertEqual(regimes[pd.Timestamp("2026-01-03")], Regime.CRISIS)
        self.assertEqual([model.fit_lengths for model in models], [[2], [3]])
        self.assertEqual([model.predict_lengths for model in models], [[2], [3]])

    def test_account_values_cover_portfolio_gross_and_notional_views(self):
        rows = pd.DataFrame(
            {
                "signal_date": ["2026-01-01", "2026-01-01", "2026-01-02"],
                "active_position": [True, True, False],
                "pair_return": [0.10, -0.05, 0.20],
                "gross_100_pnl": [10.0, -5.0, 0.0],
                "notional_200_pnl": [20.0, -10.0, 0.0],
            }
        )

        values = calculate_backtest_account_values(rows, initial_capital=100.0)

        self.assertAlmostEqual(values["portfolio_100"]["ending_value"], 102.5)
        self.assertAlmostEqual(values["gross_100_per_signal"]["ending_value"], 105.0)
        self.assertAlmostEqual(values["notional_200_per_signal"]["ending_value"], 110.0)


if __name__ == "__main__":
    unittest.main()
