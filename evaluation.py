"""回测绩效评估：收益、风险、交易成本、换手率及可选基准比较。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_equity(equity: pd.Series) -> pd.Series:
    if not isinstance(equity, pd.Series) or len(equity) < 2:
        raise ValueError("equity must be a Series with at least two observations.")
    if not isinstance(equity.index, pd.DatetimeIndex):
        raise ValueError("equity must have a DatetimeIndex.")

    equity = equity.astype(float).copy()
    equity.index = pd.to_datetime(equity.index, utc=True)
    if (
        equity.index.hasnans
        or not equity.index.is_unique
        or not equity.index.is_monotonic_increasing
        or not np.isfinite(equity.to_numpy()).all()
        or (equity <= 0).any()
    ):
        raise ValueError("equity needs increasing unique times and finite positive values.")

    intervals = equity.index[1:] - equity.index[:-1]
    if not (intervals == intervals[0]).all():
        raise ValueError("equity must be sampled at regular intervals for annualization.")
    return equity


def _performance(equity: pd.Series, risk_free_rate: float) -> tuple:
    returns = equity.pct_change(fill_method=None).iloc[1:]
    drawdown = equity / equity.cummax() - 1
    step = equity.index[1] - equity.index[0]
    periods_per_year = float(pd.Timedelta(days=365) / step)
    years = float((equity.index[-1] - equity.index[0]) / pd.Timedelta(days=365))

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    with np.errstate(over="ignore"):
        log_growth = np.log(equity.iloc[-1]) - np.log(equity.iloc[0])
        annualized_return = float(np.expm1(log_growth / years))
    std = float(returns.std(ddof=1)) if len(returns) > 1 else np.nan
    # Constant returns may still have tiny floating point rounding differences.
    if np.isfinite(std) and std < 1e-12:
        std = 0.0
    volatility = std * np.sqrt(periods_per_year)
    period_rf = np.expm1(np.log1p(risk_free_rate) / periods_per_year)
    sharpe = (
        float((returns.mean() - period_rf) / std * np.sqrt(periods_per_year))
        if std > 0 else np.nan
    )
    max_drawdown = float(-drawdown.min())

    summary = pd.Series({
        "start": equity.index[0],
        "end": equity.index[-1],
        "periods": len(returns),
        "years": years,
        "periods_per_year": periods_per_year,
        "initial_equity": float(equity.iloc[0]),
        "final_equity": float(equity.iloc[-1]),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": float(volatility),
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": annualized_return / max_drawdown if max_drawdown > 0 else np.nan,
        "period_win_rate": float((returns > 0).mean()),
        "best_period_return": float(returns.max()),
        "best_period_end": returns.idxmax(),
        "worst_period_return": float(returns.min()),
        "worst_period_end": returns.idxmin(),
    }, name="strategy")
    return summary, returns, drawdown


def evaluate_backtest(result, risk_free_rate=0.0, benchmarks=None):
    """评估 run_backtest() 的结果，返回 summary 和明细时间序列。

    risk_free_rate 是年化无风险利率，默认 0。按一年 365 天年化；
    Sharpe 使用每根 K 线的算术超额收益、样本标准差和平方根年化。
    零波动率或不足两个收益观察值时 Sharpe 为 NaN；零回撤时 Calmar 为 NaN。

    turnover = 每根 K 线买卖成交额之和 / 上一期末净资产，首根用初始资金。
    包含首次建仓；这是双边换手率。trade_count 是成交记录数，不是平仓次数。
    cost_to_initial_equity 是成本金额 / 初始资金，不代表剔除成本后的收益差。

    benchmarks 可传 {"BTC": btc_result["equity"], "equal_weight": eq_result["equity"]}。
    基准必须覆盖与策略完全一致的净值时点；传入基准自身已经扣费的净值即可。
    comparison 比较收益、风险指标，不重复计算基准的交易成本和换手率。

    月度、年度收益按 K 线结束前的 UTC 日期归属：
    例如 1 月 1 日 00:00 结算的日线收益归入前一年的 12 月。
    """
    risk_free_rate = float(risk_free_rate)
    if not np.isfinite(risk_free_rate) or risk_free_rate <= -1:
        raise ValueError("risk_free_rate must be finite and greater than -1.")

    equity = _validate_equity(result.get("equity"))
    summary, returns, drawdown = _performance(equity, risk_free_rate)

    trades = result.get("trades")
    required = {"time", "notional", "fee", "slippage"}
    if not isinstance(trades, pd.DataFrame) or not required.issubset(trades.columns):
        raise ValueError("result must include a trade log with time, notional, fee, slippage.")
    trades = trades.copy()
    trades["time"] = pd.to_datetime(trades["time"], utc=True)
    for column in ("notional", "fee", "slippage"):
        trades[column] = pd.to_numeric(trades[column], errors="raise").astype(float)
    amounts = trades[["notional", "fee", "slippage"]].to_numpy()
    if not np.isfinite(amounts).all() or (amounts < 0).any():
        raise ValueError("trade notionals and costs must be finite and nonnegative.")

    execution_times = equity.index[:-1]
    if trades["time"].isna().any() or not trades["time"].isin(execution_times).all():
        raise ValueError("trade times must match bar opening times in the equity curve.")
    traded = trades.groupby("time")["notional"].sum().reindex(execution_times, fill_value=0.0)
    turnover = (traded / equity.iloc[:-1]).rename("turnover")
    fees = float(trades["fee"].sum())
    slippage = float(trades["slippage"].sum())
    costs = pd.Series({
        "trade_count": len(trades),
        "traded_notional": float(traded.sum()),
        "total_turnover": float(turnover.sum()),
        "average_period_turnover": float(turnover.mean()),
        "annualized_turnover": float(turnover.sum() / summary["years"]),
        "total_fees": fees,
        "total_slippage": slippage,
        "total_cost": fees + slippage,
        "cost_to_initial_equity": (fees + slippage) / equity.iloc[0],
    }, dtype=object)
    summary = pd.concat([summary, costs]).rename("strategy")

    # Attribute a bar ending at midnight to the calendar period it just completed.
    dates = (returns.index - pd.Timedelta(nanoseconds=1)).tz_localize(None)
    monthly_returns = ((1 + returns).groupby(dates.to_period("M")).prod() - 1).rename("return")
    yearly_returns = ((1 + returns).groupby(dates.to_period("Y")).prod() - 1).rename("return")

    comparison_keys = [
        "total_return", "annualized_return", "annualized_volatility",
        "sharpe", "max_drawdown", "calmar", "period_win_rate",
    ]
    comparison = {"strategy": summary.loc[comparison_keys].astype(float)}
    if benchmarks is not None:
        for name, benchmark in benchmarks.items():
            if name == "strategy":
                raise ValueError("The benchmark name 'strategy' is reserved.")
            benchmark = _validate_equity(benchmark)
            if not benchmark.index.equals(equity.index):
                raise ValueError(f"Benchmark {name!r} must have the same equity timestamps.")
            metrics, _, _ = _performance(benchmark, risk_free_rate)
            comparison[name] = metrics.loc[comparison_keys].astype(float)

    return {
        "summary": summary,
        "returns": returns,
        "drawdown": drawdown.rename("drawdown"),
        "turnover": turnover,
        "monthly_returns": monthly_returns,
        "yearly_returns": yearly_returns,
        "comparison": pd.DataFrame(comparison),
    }
