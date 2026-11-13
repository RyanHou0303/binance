from pathlib import Path
import warnings

import numpy as np
import pandas as pd


def load_data(config):
    universe = list(config["universe"])
    if not universe or len(set(universe)) != len(universe):
        raise ValueError(
            "universe must be nonempty and contain unique symbols."
        )

    steps = {
        "1h": pd.Timedelta(hours=1),
        "1d": pd.Timedelta(days=1),
    }
    if config["interval"] not in steps:
        raise ValueError("interval must be '1h' or '1d'.")

    step = steps[config["interval"]]
    start = pd.to_datetime(config["start"], utc=True)
    latest = pd.Timestamp.now(tz="UTC").floor(step)
    end = (
        pd.to_datetime(config["end"], utc=True)
        if config.get("end")
        else latest
    )

    if (
        pd.isna(start)
        or pd.isna(end)
        or start >= end
        or end > latest
    ):
        raise ValueError(
            "Require start < end <= the latest completed bar boundary."
        )

    if start != start.floor(step) or end != end.floor(step):
        raise ValueError(
            "start and end must align with the bar interval."
        )

    warmup = config.get("warmup_bars", 0)
    if (
        not isinstance(warmup, int)
        or isinstance(warmup, bool)
        or warmup < 0
    ):
        raise ValueError(
            "warmup_bars must be a nonnegative integer."
        )

    missing_policy = config.get("missing_data", "warn")
    if missing_policy not in ("warn", "raise"):
        raise ValueError(
            "missing_data must be 'warn' or 'raise'."
        )

    index = pd.date_range(
        start - warmup * step,
        end,
        freq=step,
        inclusive="left",
    )
    frames = {}

    for symbol in universe:
        path = (
            Path(config["data_dir"])
            / f"{symbol}_{config['interval']}.csv"
        )
        df = pd.read_csv(
            path, usecols=["open_time", "open", "close"]
        )
        df["open_time"] = pd.to_datetime(
            df["open_time"], utc=True
        )

        if (
            df["open_time"].isna().any()
            or df["open_time"].duplicated().any()
        ):
            raise ValueError(
                f"{symbol}: missing or duplicate timestamps."
            )

        df = df.set_index("open_time").reindex(index)
        df = df[["open", "close"]].apply(
            pd.to_numeric, errors="coerce"
        )

        bad_cells = ~np.isfinite(df) | (df <= 0)
        bad_rows = bad_cells.any(axis=1)

        if bad_rows.any():
            message = (
                f"{symbol}: {int(bad_rows.sum())} rows have "
                f"missing or invalid prices; first occurrence: "
                f"{df.index[bad_rows][0]}."
            )

            if missing_policy == "raise":
                raise ValueError(message)

            warnings.warn(
                message + " Values are kept as NaN.",
                RuntimeWarning,
                stacklevel=2,
            )

        frames[symbol] = df.mask(bad_cells, np.nan)

    return {
        "open": pd.DataFrame({
            s: frames[s]["open"] for s in universe
        }),
        "close": pd.DataFrame({
            s: frames[s]["close"] for s in universe
        }),
        "step": step,
        "start": start,
        "end": end,
    }


def rebalance_key(timestamp, rule):
    if rule == "1h":
        return timestamp.floor("h")
    if rule == "1d":
        return timestamp.normalize()
    if rule == "1w":
        return timestamp.normalize() - pd.Timedelta(
            days=timestamp.weekday()
        )
    if rule == "1mo":
        return timestamp.year, timestamp.month
    return None


def run_backtest(config, data=None):
    """按目标权重回测，正权重做多、负权重做空，绝对权重合计不超过 1。

    做空使用简化现金与持仓记账，不模拟借币、资金费率或保证金强平。
    若净资产不再为正则停止并报错。
    """
    # Reuse preloaded data when provided.
    if data is None:
        data = load_data(config)

    if config.get("weights") is None:
        return {"data": data, "summary": None}

    universe = list(config["universe"])
    opens = data["open"].loc[data["start"]:, universe]
    closes = data["close"].reindex(
        index=opens.index, columns=universe
    )

    if opens.empty:
        raise ValueError(
            "No bars are available in the backtest period."
        )

    # Rows are execution times; columns are symbols.
    weights = config["weights"]
    if not isinstance(weights, pd.DataFrame):
        raise TypeError(
            "weights must be a time-by-symbol pandas DataFrame."
        )

    weights = weights.copy()
    weights.index = pd.to_datetime(weights.index, utc=True)

    if (
        weights.index.hasnans
        or not weights.index.is_unique
        or not weights.columns.is_unique
    ):
        raise ValueError(
            "weights must have valid, unique timestamps and columns."
        )

    weights = weights.reindex(
        index=opens.index, columns=universe
    )
    w = weights.to_numpy(dtype=float)

    if (
        not np.isfinite(w).all()
        or (np.abs(w).sum(axis=1) > 1 + 1e-12).any()
    ):
        raise ValueError(
            "weights must cover all bars and symbols, be finite, "
            "and have absolute values summing to at most 1 per row."
        )

    initial = float(config.get("initial_cash", 100_000))
    fee_rate = float(config.get("fee_bps", 0)) / 10_000
    slip_rate = float(config.get("slippage_bps", 0)) / 10_000
    cost_rate = fee_rate + slip_rate

    if (
        not np.isfinite([initial, fee_rate, slip_rate]).all()
        or initial <= 0
        or min(fee_rate, slip_rate) < 0
        or cost_rate >= 1
    ):
        raise ValueError(
            "Require positive capital and nonnegative total costs < 1."
        )

    rule = config.get("rebalance", "1d")
    if rule not in (None, "1h", "1d", "1w", "1mo"):
        raise ValueError("Unsupported rebalance frequency.")

    if config["interval"] == "1d" and rule == "1h":
        raise ValueError(
            "Hourly rebalancing requires hourly data."
        )

    cash = initial
    quantity = np.zeros(len(universe))
    last_price = np.full(len(universe), np.nan)

    equity_values, positions, cash_values, trades = [], [], [], []
    previous_key = None

    for i, timestamp in enumerate(opens.index):
        open_price = opens.iloc[i].to_numpy(dtype=float)
        valid = np.isfinite(open_price) & (open_price > 0)

        # Value existing positions at known opening prices.
        last_price[valid] = open_price[valid]
        mark = np.nan_to_num(last_price, nan=0.0)

        value = quantity * mark
        equity_before = cash + value.sum()
        if equity_before <= 0:
            raise ValueError(
                f"Equity is nonpositive at {timestamp}; "
                "margin liquidation is not modeled."
            )

        key = rebalance_key(timestamp, rule)

        if i == 0 or (rule is not None and key != previous_key):
            # Target weights must be known before this bar opens.
            target_value = equity_before * w[i]

            # Freeze positions without valid opening prices.
            sell = np.where(
                valid, np.maximum(value - target_value, 0), 0
            )
            buy = np.where(
                valid, np.maximum(target_value - value, 0), 0
            )

            # Sells may open shorts; buys may cover them. Short proceeds
            # increase cash, while negative quantities remain liabilities.
            # Sell first, then scale purchases to cash after costs.
            cash += sell.sum() * (1 - cost_rate)
            required = buy.sum() * (1 + cost_rate)

            if required > cash:
                buy *= max(cash, 0) / required

            quantity[valid] += (
                (buy[valid] - sell[valid]) / open_price[valid]
            )
            cash = max(
                0.0, cash - buy.sum() * (1 + cost_rate)
            )

            for side, amounts in (("SELL", sell), ("BUY", buy)):
                for j in np.flatnonzero(amounts > 0):
                    amount = amounts[j]
                    trades.append({
                        "time": timestamp,
                        "symbol": universe[j],
                        "side": side,
                        "notional": amount,
                        "fee": amount * fee_rate,
                        "slippage": amount * slip_rate,
                    })

        previous_key = key

        # Mark to closing prices; retain the last known price if missing.
        close_price = closes.iloc[i].to_numpy(dtype=float)
        valid_close = (
            np.isfinite(close_price) & (close_price > 0)
        )
        last_price[valid_close] = close_price[valid_close]

        equity_after = cash + quantity @ np.nan_to_num(last_price, nan=0.0)
        if equity_after <= 0:
            raise ValueError(
                f"Equity is nonpositive at {timestamp + data['step']}; "
                "margin liquidation is not modeled."
            )
        equity_values.append(equity_after)
        positions.append(quantity.copy())
        cash_values.append(cash)

    # Include the initial balance and label equity at each bar's end.
    times = opens.index + data["step"]
    equity = pd.Series(
        [initial] + equity_values,
        index=[data["start"]] + list(times),
        name="equity",
    )
    drawdown = equity / equity.cummax() - 1

    trade_log = pd.DataFrame(trades, columns=[
        "time", "symbol", "side",
        "notional", "fee", "slippage",
    ])

    return {
        "data": data,
        "equity": equity,
        "returns": equity.pct_change(fill_method=None).iloc[1:],
        "holdings": pd.DataFrame(
            positions, index=times, columns=universe
        ),
        "cash": pd.Series(cash_values, index=times, name="cash"),
        "trades": trade_log,
        "summary": {
            "final_equity": float(equity.iloc[-1]),
            "total_return": float(equity.iloc[-1] / initial - 1),
            "max_drawdown": float(-drawdown.min()),
            "total_fees": float(trade_log["fee"].sum()),
            "total_slippage": float(trade_log["slippage"].sum()),
        },
    }


def plot_backtest(result, save_path=None, show=True):
    """绘制归一化净值和回撤，支持保存图片，返回 (fig, axes)。

    例如：plot_backtest(result, save_path="outputs/backtest.png")。
    批量运行时设置 show=False，使用完后调用 plt.close(fig)。
    """
    equity = result.get("equity")
    if equity is None or equity.empty:
        raise ValueError("Run a backtest with weights before plotting.")

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    net_value = equity / equity.iloc[0]
    drawdown = equity / equity.cummax() - 1

    fig, axes = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        layout="constrained",
    )
    axes[0].plot(net_value.index, net_value, color="#2563eb", lw=1.5)
    axes[0].axhline(1, color="gray", lw=0.8, linestyle="--")
    axes[0].set_title("Backtest Performance")
    axes[0].set_ylabel("Net value (initial = 1)")

    axes[1].fill_between(
        drawdown.index, drawdown.to_numpy(), 0,
        color="#dc2626", alpha=0.35,
    )
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date (UTC)")
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1))

    locator = mdates.AutoDateLocator(tz=equity.index.tz)
    axes[1].xaxis.set_major_locator(locator)
    axes[1].xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(locator, tz=equity.index.tz)
    )
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.margins(x=0)

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)

    if show:
        plt.show()

    return fig, axes


if __name__ == "__main__":
    from evaluation import evaluate_backtest

    config = {
        "universe": [
            "BTCUSDT", "ETHUSDT", "BNBUSDT",
            "SOLUSDT", "XRPUSDT", "DOGEUSDT",
            "ADAUSDT", "TRXUSDT", "LINKUSDT",
        ],

        # UTC range: start inclusive, end exclusive.
        "start": "2021-01-01",
        "end": "2026-01-01",
        "data_dir": "/Users/bohanhou/binance/data",
        "interval": "1h",
        "warmup_bars": 0,
        "missing_data": "warn",

        # Example one-way costs: 10 bps = 0.1%.
        "initial_cash": 100_000,
        "fee_bps": 10,
        "slippage_bps": 0,
        "rebalance": "1d",
        "weights": None,
    }

    # Load once and construct an n-by-k target weight matrix.
    data = load_data(config)
    index = data["open"].loc[data["start"]:].index

    # Equal weights are used only as an example.
    config["weights"] = pd.DataFrame(
        1 / len(config["universe"]),
        index=index,
        columns=config["universe"],
    )

    result = run_backtest(config, data=data)
    report = evaluate_backtest(result)
    print(report["summary"].to_string())
    plot_backtest(
        result,
        save_path=Path(__file__).resolve().parent / "outputs" / "backtest.png",
    )
