from collections import deque

import numpy as np
import pandas as pd


class Signal:
    """Monthly ridge regression using causal, close-derived features."""

    def get_weight(self, close: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(close, pd.DataFrame):
            raise TypeError("close must be a pandas DataFrame.")
        if not isinstance(close.index, pd.DatetimeIndex):
            raise TypeError("A DatetimeIndex is required for monthly refitting.")
        if (
            close.index.hasnans
            or close.index.has_duplicates
            or not close.index.is_monotonic_increasing
        ):
            raise ValueError("Timestamps must be valid, unique, and increasing.")
        if not close.columns.is_unique:
            raise ValueError("Asset identifiers must be unique.")
        if any(pd.api.types.is_complex_dtype(dtype) for dtype in close.dtypes):
            raise TypeError("Closing prices must be real numbers.")

        try:
            prices = close.to_numpy(dtype=np.float64, na_value=np.nan)
        except (TypeError, ValueError) as exc:
            raise TypeError("Closing prices must be numeric.") from exc

        n_dates, n_assets = prices.shape
        weights = np.zeros((n_dates, n_assets), dtype=np.float64)

        if n_dates < 2 or n_assets < 20:
            return pd.DataFrame(weights, index=close.index, columns=close.columns)

        # Assumptions: comparable daily bars, adjusted positive closing prices,
        # a point-in-time universe, fractional positions, and permission to short.
        # Invalid prices represent missing observations; never fill them forward.
        valid = np.isfinite(prices) & (prices > 0.0)
        log_prices = np.full(prices.shape, np.nan, dtype=np.float64)
        np.log(prices, out=log_prices, where=valid)

        log_frame = pd.DataFrame(log_prices)
        daily_returns = log_frame.diff()

        # Volatility uses sample standard deviation; cross-sectional scaling
        # below uses population standard deviation.
        vol_21 = daily_returns.rolling(21, min_periods=21).std(ddof=1).to_numpy()
        vol_63 = daily_returns.rolling(63, min_periods=63).std(ddof=1).to_numpy()
        complete_history = (
            pd.DataFrame(valid)
            .rolling(253, min_periods=253)
            .sum()
            .eq(253)
            .to_numpy()
        )

        # A fixed ordering of asset identifiers resolves all score ties.
        identifier_order = sorted(
            range(n_assets),
            key=lambda j: (
                type(close.columns[j]).__module__,
                type(close.columns[j]).__qualname__,
                repr(close.columns[j]),
            ),
        )
        identifier_rank = np.empty(n_assets, dtype=np.int64)
        identifier_rank[identifier_order] = np.arange(n_assets)

        pair_i, pair_j = np.triu_indices(8, k=1)
        feature_count = 8 + 8 + len(pair_i)
        parameter_count = feature_count + 1

        def make_design(t):
            eligible = np.flatnonzero(complete_history[t])
            if len(eligible) < 20:
                return np.empty(0, dtype=np.int64), None

            # Cumulative returns are sums of log returns. Computing differences
            # of log prices also avoids overflowing price ratios.
            raw = np.column_stack(
                [
                    log_prices[t, eligible] - log_prices[t - 1, eligible],
                    log_prices[t, eligible] - log_prices[t - 5, eligible],
                    log_prices[t - 5, eligible] - log_prices[t - 21, eligible],
                    log_prices[t - 5, eligible] - log_prices[t - 63, eligible],
                    log_prices[t - 5, eligible] - log_prices[t - 126, eligible],
                    log_prices[t - 5, eligible] - log_prices[t - 252, eligible],
                    vol_21[t, eligible],
                    vol_63[t, eligible],
                ]
            )

            finite = np.isfinite(raw).all(axis=1)
            eligible = eligible[finite]
            raw = raw[finite]
            if len(eligible) < 20:
                return np.empty(0, dtype=np.int64), None

            bounds = np.quantile(raw, [0.025, 0.975], axis=0)
            clipped = np.clip(raw, bounds[0], bounds[1])
            means = clipped.mean(axis=0)
            scales = clipped.std(axis=0, ddof=0)
            varying = (np.ptp(clipped, axis=0) > 0.0) & (scales > 0.0)

            standardized = np.zeros_like(clipped)
            standardized[:, varying] = (
                clipped[:, varying] - means[varying]
            ) / scales[varying]

            phi = np.column_stack(
                [
                    standardized,
                    standardized**2,
                    standardized[:, pair_i] * standardized[:, pair_j],
                ]
            )
            design = np.column_stack([np.ones(len(eligible)), phi])
            return eligible, design

        # Each stored date contributes its own mean loss, so dates have equal
        # influence even when their eligible asset counts differ.
        training_dates = deque()
        gram_sum = np.zeros((parameter_count, parameter_count))
        cross_sum = np.zeros(parameter_count)
        pending = deque()
        coefficients = None

        years = close.index.year.to_numpy()
        months = close.index.month.to_numpy()

        # Process closes only through the bar preceding each execution row.
        for t in range(n_dates - 1):
            if t >= 252:
                eligible, design = make_design(t)
            else:
                eligible, design = np.empty(0, dtype=np.int64), None

            pending.append((eligible, design))

            # At cutoff t, observation s=t-2 first has an available label:
            # log(C[s+2] / C[s+1]). Feature scaling remains that observed at s.
            if len(pending) > 2:
                past_assets, past_design = pending.popleft()
                if past_design is not None:
                    labels = (
                        log_prices[t, past_assets]
                        - log_prices[t - 1, past_assets]
                    )
                    observed = np.isfinite(labels)

                    # An eligible training date requires at least 20 assets
                    # with both date-s features and an observed delayed label.
                    if np.count_nonzero(observed) >= 20:
                        x = past_design[observed]
                        y = labels[observed]

                        # A constant label cross-section carries zero relative
                        # return information and is standardized to zero.
                        scale = y.std(ddof=0)
                        if np.ptp(y) == 0.0 or scale == 0.0:
                            y = np.zeros_like(y)
                        else:
                            y = (y - y.mean()) / scale

                        date_gram = (x.T @ x) / len(y)
                        date_cross = (x.T @ y) / len(y)

                        training_dates.append((date_gram, date_cross))
                        gram_sum += date_gram
                        cross_sum += date_cross

                        if len(training_dates) > 756:
                            old_gram, old_cross = training_dates.popleft()
                            gram_sum -= old_gram
                            cross_sum -= old_cross

            # Assume the final observed bar before a calendar-month transition
            # is that month's final trading close. The execution bar's timestamp
            # identifies this transition without using its closing price.
            month_end = (
                years[t] != years[t + 1] or months[t] != months[t + 1]
            )
            if month_end:
                if len(training_dates) >= 504:
                    a = gram_sum / len(training_dates)
                    rhs = cross_sum / len(training_dates)
                    a = (a + a.T) * 0.5

                    # Lambda=1 on every nonlinear feature coefficient.
                    # The first coordinate is the unpenalized intercept.
                    a[np.diag_indices_from(a)] += 1.0
                    a[0, 0] -= 1.0
                    coefficients = np.linalg.solve(a, rhs)
                else:
                    coefficients = None

            if coefficients is None or design is None:
                continue

            scores = design @ coefficients
            if not np.isfinite(scores).all():
                continue

            # Ascending score, then ascending fixed identifier. Taking opposite
            # ends gives disjoint long and short selections even for tied scores.
            ranked = np.lexsort((identifier_rank[eligible], scores))
            n_selected = len(eligible) // 5
            target = weights[t + 1]
            target[eligible[ranked[:n_selected]]] = -0.5 / n_selected
            target[eligible[ranked[-n_selected:]]] = 0.5 / n_selected

            # Guard against floating-point summation marginally exceeding one.
            gross = np.abs(target).sum()
            while gross > 1.0:
                target *= np.nextafter(1.0 / gross, 0.0)
                gross = np.abs(target).sum()

        # Missing-data exits appear at the next open. Preserve every original
        # column and all historical targets, including subsequently missing assets.
        # Close-only inputs generate targets, but cannot validate actual open
        # execution, liquidity, borrowing, transaction costs, or tradable returns.
        return pd.DataFrame(weights, index=close.index, columns=close.columns)

if __name__ == "__main__":
    from pathlib import Path

    import pandas as pd

    from backtesting import load_data, plot_backtest, run_backtest
    from evaluation import evaluate_backtest

    config = {
        "universe": [
            "BTCUSDT", "ETHUSDT", "BNBUSDT",
            "SOLUSDT", "XRPUSDT", "DOGEUSDT",
            "ADAUSDT", "TRXUSDT", "LINKUSDT",
        ],
        "start": "2021-01-01",
        "end": "2026-01-01",
        "data_dir": Path(__file__).resolve().parent / "data",
        "interval": "1h",
        "warmup_bars": 2,  # 两根收盘价用于计算首日的涨跌幅因子。
        "missing_data": "warn",
        "initial_cash": 100_000,
        "fee_bps": 10,
        "slippage_bps": 0,
        "rebalance": "1d",
        "weights": None,
    }

    # 1. 加载行情，包含用于生成信号的预热数据。
    data = load_data(config)

    # 2. 生成权重矩阵，并传入回测配置。
    signal = Signal()
    weight = signal.get_weight(data["close"])
    config["weights"] = weight

    # 3. 复用已加载的行情运行回测。
    result = run_backtest(config, data=data)


    print("前 5 行交易权重：")
    print(weight.loc[data["start"]:].head().to_string())
    print("\n回测结果：")
    report = evaluate_backtest(result)
    print(report["summary"].to_string())
    print("\n年度收益：")
    print(report["yearly_returns"].map("{:.2%}".format).to_string())
    plot_backtest(
        result,
    )
