import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from initial_trd.backtesting import (
    calculate_backtest_account_values,
    calculate_pair_return,
    classify_backtest_regimes_by_date,
    estimate_pair_relationship,
    generate_symmetric_pair_signal,
    simulate_pair_backtest,
    train_lstm_prediction_for_date,
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
    def test_pair_return_uses_hedge_ratio_and_entry_costs(self):
        dates = pd.date_range("2026-01-01", periods=2, freq="D")
        closes = pd.DataFrame(
            {
                "A": [100.0, 112.0],
                "B": [100.0, 97.0],
            },
            index=dates,
        )

        long_return, short_return, pair_return = calculate_pair_return(
            closes,
            signal_date=dates[0],
            next_date=dates[1],
            long_leg="A",
            short_leg="B",
            stock_a="A",
            stock_b="B",
            hedge_ratio=2.0,
            transaction_cost_per_leg=0.0010,
            slippage_per_leg=0.0005,
            charge_entry_cost=True,
        )

        self.assertAlmostEqual(long_return, 0.12)
        self.assertAlmostEqual(short_return, 0.03)
        self.assertAlmostEqual(pair_return, 0.0585)

    def test_estimate_pair_relationship_accepts_cointegrated_log_prices(self):
        rng = np.arange(180, dtype=float)
        log_b = 4.0 + np.cumsum(rng) * 0.0005
        stationary_residual = 0.01 * ((rng % 5) - 2.0)
        log_a = 0.4 + 1.5 * log_b + stationary_residual
        dates = pd.date_range("2025-01-01", periods=len(rng), freq="D")
        closes = pd.DataFrame(
            {
                "A": np.exp(log_a),
                "B": np.exp(log_b),
            },
            index=dates,
        )

        relationship = estimate_pair_relationship(
            closes,
            "A",
            "B",
            min_observations=60,
        )

        self.assertTrue(relationship.is_cointegrated)
        self.assertAlmostEqual(relationship.hedge_ratio, 1.5, places=1)

    def test_estimate_pair_relationship_rejects_independent_log_trends(self):
        random = np.random.default_rng(7)
        log_a = 4.0 + np.cumsum(random.normal(0.0, 0.01, size=240))
        log_b = 4.0 + np.cumsum(random.normal(0.0, 0.01, size=240))
        dates = pd.date_range("2025-01-01", periods=len(log_a), freq="D")
        closes = pd.DataFrame(
            {
                "A": np.exp(log_a),
                "B": np.exp(log_b),
            },
            index=dates,
        )

        relationship = estimate_pair_relationship(
            closes,
            "A",
            "B",
            min_observations=60,
        )

        self.assertFalse(relationship.is_cointegrated)

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
        dates = pd.date_range("2026-01-01", periods=6, freq="D")
        closes = pd.DataFrame(
            {
                "A": [100.0, 100.0, 101.0, 100.6, 100.8, 101.0],
                "B": [100.0] * 6,
            },
            index=dates,
        )

        result = simulate_pair_backtest(
            closes,
            {dates[-3]: 0.01},
            tickers=["A", "B"],
            days=1,
            window=3,
        )

        row = result.rows.iloc[0]
        self.assertEqual(row["signal_action"], "CLOSE")
        self.assertEqual(row["executed_action"], "HOLD")
        self.assertFalse(row["active_position"])
        self.assertEqual(row["pair_return"], 0.0)
        self.assertEqual(row["execution_date"], dates[-2].strftime("%Y-%m-%d"))
        self.assertEqual(
            result.summary["account_values"]["portfolio_100"]["ending_value"],
            100.0,
        )

    def test_open_is_blocked_when_short_leg_is_not_available(self):
        dates = pd.date_range("2026-01-01", periods=32, freq="D")
        closes = pd.DataFrame(
            {
                "A": [101.0] * 29 + [90.0, 91.0, 92.0],
                "B": [100.0] * 32,
            },
            index=dates,
        )

        result = simulate_pair_backtest(
            closes,
            {dates[-3]: 0.01},
            tickers=["A", "B"],
            days=1,
            shortable_tickers=[],
            require_cointegration=False,
            transaction_cost_per_leg=0.0,
            slippage_per_leg=0.0,
        )

        row = result.rows.iloc[0]
        self.assertEqual(row["signal_action"], "OPEN_PAIR")
        self.assertEqual(row["executed_action"], "BLOCKED_SHORT_UNAVAILABLE")
        self.assertFalse(row["active_position"])
        self.assertEqual(result.summary["trades_blocked_short"], 1)

    def test_open_is_blocked_when_pair_fails_cointegration_filter(self):
        dates = pd.date_range("2026-01-01", periods=32, freq="D")
        closes = pd.DataFrame(
            {
                "A": [101.0] * 29 + [90.0, 91.0, 92.0],
                "B": [100.0] * 32,
            },
            index=dates,
        )

        result = simulate_pair_backtest(
            closes,
            {dates[-3]: 0.01},
            tickers=["A", "B"],
            days=1,
            shortable_tickers=["B"],
            min_cointegration_observations=4,
            transaction_cost_per_leg=0.0,
            slippage_per_leg=0.0,
        )

        row = result.rows.iloc[0]
        self.assertEqual(row["signal_action"], "OPEN_PAIR")
        self.assertEqual(row["executed_action"], "BLOCKED_COINTEGRATION")
        self.assertFalse(row["cointegrated"])
        self.assertEqual(result.summary["trades_blocked_cointegration"], 1)

    def test_open_position_pays_exit_cost_on_final_interval(self):
        dates = pd.date_range("2026-01-01", periods=32, freq="D")
        closes = pd.DataFrame(
            {
                "A": [101.0] * 29 + [90.0, 91.0, 92.0],
                "B": [100.0] * 32,
            },
            index=dates,
        )

        result = simulate_pair_backtest(
            closes,
            {dates[-3]: 0.01},
            tickers=["A", "B"],
            days=1,
            require_short_availability=False,
            require_cointegration=False,
            transaction_cost_per_leg=0.0010,
            slippage_per_leg=0.0005,
        )

        row = result.rows.iloc[0]
        gross_return = 0.5 * (92.0 / 91.0 - 1.0)
        self.assertEqual(row["executed_action"], "OPEN_PAIR")
        self.assertEqual(row["execution_date"], dates[-2].strftime("%Y-%m-%d"))
        self.assertTrue(row["forced_close_end"])
        self.assertAlmostEqual(row["trade_cost_return"], 0.0030)
        self.assertAlmostEqual(row["pair_return"], gross_return - 0.0030)
        self.assertEqual(result.summary["positions_forced_closed_end"], 1)

    def test_backtest_uses_regime_for_each_signal_date(self):
        dates = pd.date_range("2026-01-01", periods=32, freq="D")
        closes = pd.DataFrame(
            {
                "A": [101.0] * 29 + [90.0, 91.0, 92.0],
                "B": [100.0] * 32,
            },
            index=dates,
        )

        result = simulate_pair_backtest(
            closes,
            {dates[-3]: 0.01},
            tickers=["A", "B"],
            days=1,
            regime={dates[-3]: Regime.CRISIS},
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

    def test_online_lstm_training_excludes_unrevealed_targets(self):
        dates = pd.date_range("2026-01-01", periods=5, freq="D")
        features = pd.DataFrame(
            {
                "date": dates,
                "bist_ret": [-0.02, -0.01, 0.01, 0.02, 0.03],
                "fx_ret": [0.0] * 5,
                "real_rate": [1.0] * 5,
                "cds_velocity": [0.0] * 5,
                "fx_volatility": [0.3, 0.4, 0.2, 0.5, 0.6],
                "market_breadth": [1.0] * 5,
                "target": [0.1, 0.2, 0.3, 0.4, 0.5],
            }
        )
        captured_dates = []

        def fake_build_sequence_arrays(df, **kwargs):
            del kwargs
            captured_dates.extend(pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d"))
            raise RuntimeError("stop after capture")

        with patch(
            "initial_trd.backtesting.build_sequence_arrays",
            side_effect=fake_build_sequence_arrays,
        ):
            with self.assertRaisesRegex(RuntimeError, "stop after capture"):
                train_lstm_prediction_for_date(
                    features,
                    dates[-1],
                    feature_columns=[
                        "bist_ret",
                        "fx_ret",
                        "real_rate",
                        "cds_velocity",
                        "fx_volatility",
                        "market_breadth",
                    ],
                    target_column="target",
                    weight_column="sample_weight",
                    sequence_length=2,
                    validation_size=0.5,
                    batch_size=1,
                    epochs=1,
                    hidden_dim=4,
                    num_layers=1,
                    dropout=0.0,
                    lr=0.001,
                    weight_decay=0.0,
                    seed=7,
                    device="cpu",
                    use_regime_weights=False,
                    hmm_random_state=None,
                    target_horizon=2,
                )

        self.assertEqual(captured_dates, ["2026-01-01", "2026-01-02", "2026-01-03"])

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
