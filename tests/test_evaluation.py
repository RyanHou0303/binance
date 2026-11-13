import unittest

import numpy as np
import pandas as pd

from backtesting import run_backtest
from evaluation import evaluate_backtest


def make_result(values, start="2023-01-01", freq="1D", trades=None):
    equity = pd.Series(
        values, index=pd.date_range(start, periods=len(values), freq=freq, tz="UTC"),
        name="equity", dtype=float,
    )
    if trades is None:
        trades = pd.DataFrame(columns=["time", "notional", "fee", "slippage"])
    return {"equity": equity, "trades": trades}


class EvaluationTests(unittest.TestCase):
    def test_hand_calculated_returns_and_risk(self):
        result = make_result([100, 110, 99, 118.8])
        # Ignore a stale cached returns field; equity is the source of truth.
        result["returns"] = pd.Series([999])
        report = evaluate_backtest(result, risk_free_rate=0.05)
        metrics = report["summary"]
        returns = np.array([0.1, -0.1, 0.2])
        std = returns.std(ddof=1)
        self.assertAlmostEqual(metrics["total_return"], 0.188)
        self.assertAlmostEqual(metrics["annualized_return"], 1.188 ** (365 / 3) - 1, places=2)
        self.assertAlmostEqual(metrics["annualized_volatility"], std * np.sqrt(365))
        expected_sharpe = (returns.mean() - (1.05 ** (1 / 365) - 1)) / std * np.sqrt(365)
        self.assertAlmostEqual(metrics["sharpe"], expected_sharpe)
        self.assertAlmostEqual(metrics["max_drawdown"], 0.1)
        self.assertAlmostEqual(metrics["period_win_rate"], 2 / 3)
        self.assertEqual(metrics["worst_period_end"], result["equity"].index[2])

    def test_hourly_and_daily_annualization(self):
        for freq, count in (("1D", 365), ("1h", 8760)):
            with self.subTest(freq=freq):
                result = make_result(100 * np.exp(np.linspace(0, np.log(1.1), count + 1)), freq=freq)
                metrics = evaluate_backtest(result)["summary"]
                self.assertEqual(metrics["periods_per_year"], count)
                self.assertAlmostEqual(metrics["years"], 1)
                self.assertAlmostEqual(metrics["annualized_return"], 0.1)
                self.assertTrue(np.isnan(metrics["sharpe"]))

    def test_flat_and_single_period(self):
        for values in ([100, 100], [100, 100, 100]):
            with self.subTest(values=values):
                metrics = evaluate_backtest(make_result(values))["summary"]
                self.assertEqual(metrics["total_return"], 0)
                self.assertEqual(metrics["max_drawdown"], 0)
                self.assertEqual(metrics["period_win_rate"], 0)
                self.assertTrue(np.isnan(metrics["sharpe"]))
                self.assertTrue(np.isnan(metrics["calmar"]))
                self.assertEqual(metrics["trade_count"], 0)
                self.assertEqual(metrics["total_turnover"], 0)

    def test_month_and_year_boundaries_include_first_return(self):
        report = evaluate_backtest(make_result([100, 110, 121, 108.9], start="2023-12-30"))
        self.assertAlmostEqual(report["monthly_returns"].loc["2023-12"], 0.21)
        self.assertAlmostEqual(report["monthly_returns"].loc["2024-01"], -0.1)
        self.assertAlmostEqual(report["yearly_returns"].loc["2023"], 0.21)
        self.assertAlmostEqual(report["yearly_returns"].loc["2024"], -0.1)
        self.assertAlmostEqual((1 + report["monthly_returns"]).prod() - 1, 0.089)

    def test_turnover_costs_and_no_input_mutation(self):
        result = make_result([100, 110, 99])
        times = result["equity"].index
        result["trades"] = pd.DataFrame({
            "time": [times[0], times[0], times[1]],
            "notional": [25., 25., 110.],
            "fee": [0.025, 0.025, 0.11],
            "slippage": [0.01, 0.01, 0.02],
        })
        original_equity = result["equity"].copy()
        original_trades = result["trades"].copy()
        report = evaluate_backtest(result)
        metrics = report["summary"]
        np.testing.assert_allclose(report["turnover"], [0.5, 1.0])
        self.assertEqual(metrics["trade_count"], 3)
        self.assertEqual(metrics["traded_notional"], 160)
        self.assertAlmostEqual(metrics["total_turnover"], 1.5)
        self.assertAlmostEqual(metrics["annualized_turnover"], 1.5 * 365 / 2)
        self.assertAlmostEqual(metrics["total_cost"], 0.2)
        self.assertAlmostEqual(metrics["cost_to_initial_equity"], 0.002)
        pd.testing.assert_series_equal(result["equity"], original_equity)
        pd.testing.assert_frame_equal(result["trades"], original_trades)

    def test_benchmark_comparison_requires_same_times(self):
        result = make_result([100, 110, 99])
        benchmark = result["equity"] * 10
        report = evaluate_backtest(result, benchmarks={"BTC": benchmark})
        np.testing.assert_allclose(report["comparison"]["strategy"], report["comparison"]["BTC"], equal_nan=True)
        with self.assertRaises(ValueError):
            evaluate_backtest(result, benchmarks={"BTC": benchmark.iloc[1:]})

    def test_invalid_equity_and_trades(self):
        for values in ([100], [100, np.nan], [100, 0], [100, -1], [100, np.inf]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                evaluate_backtest(make_result(values))
        for offsets in ([0, 1, 3], [0, 1, 1], [0, 2, 1]):
            result = make_result([100, 110, 99])
            result["equity"].index = pd.Timestamp("2023-01-01", tz="UTC") + pd.to_timedelta(offsets, unit="D")
            with self.subTest(offsets=offsets), self.assertRaises(ValueError):
                evaluate_backtest(result)
        result = make_result([100, 110])
        result["trades"] = pd.DataFrame({
            "time": [result["equity"].index[-1]], "notional": [10], "fee": [0], "slippage": [0],
        })
        with self.assertRaises(ValueError):
            evaluate_backtest(result)
        with self.assertRaises(ValueError):
            evaluate_backtest(make_result([100, 110]), risk_free_rate=-1)

    def test_real_backtest_long_short_result(self):
        times = pd.date_range("2023-01-01", periods=2, tz="UTC")
        data = {
            "open": pd.DataFrame([[100, 100], [110, 90]], index=times, columns=["A", "B"]),
            "close": pd.DataFrame([[110, 90], [121, 81]], index=times, columns=["A", "B"]),
            "start": times[0], "step": pd.Timedelta(days=1),
        }
        config = {
            "universe": ["A", "B"], "interval": "1d", "initial_cash": 100,
            "weights": pd.DataFrame([[.5, -.5], [-.5, .5]], index=times, columns=["A", "B"]),
        }
        result = run_backtest(config, data=data)
        report = evaluate_backtest(result)
        self.assertAlmostEqual(report["summary"]["total_return"], -.01)
        self.assertAlmostEqual(report["summary"]["max_drawdown"], .1)
        self.assertAlmostEqual(report["summary"]["total_fees"], result["summary"]["total_fees"])
        self.assertEqual(report["summary"]["trade_count"], 4)


if __name__ == "__main__":
    unittest.main()
