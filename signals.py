from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def __getattr__(name):
    # 当前文件与标准库 signal 同名，转发 Matplotlib 图窗需要的系统接口。
    import _signal

    return getattr(_signal, name)


class Signal:
    """生成回测使用的目标权重，当前示例为每个币固定 10%。

    用法：weight = Signal().get_weight(data["close"])
    返回的 DataFrame 行为开盘交易时间，列为币种。
    """

    def get_weight(self, close: pd.DataFrame) -> pd.DataFrame:
        """保留上一根 K 线涨跌幅的选币模板，当前返回固定权重。

        如恢复排名策略，可用 long/short 掩码赋予正负权重，
        并在有效币种不足四只时将整行设为 0。
        """
        import pandas as pd

        if close.shape[1] == 0:
            raise ValueError("close 至少需要包含一个币种")


        # 因子可在这里替换；shift(1) 保证开盘只使用之前已知的数据。
        factor = close.pct_change(fill_method=None).shift(1)
        rank = factor.rank(axis=1, ascending=False, method="first")
        count = factor.notna().sum(axis=1)

        long = rank.le(2)
        short = rank.gt(count - 2, axis=0)
        weight = pd.DataFrame(0.0,index=close.index, columns=close.columns)
        weight.loc[:, :] = 0.1

        return weight


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
