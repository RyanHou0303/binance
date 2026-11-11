from pathlib import Path
import numpy as np
import pandas as pd
import warnings
config = {
    "universe": [
        "BTCUSDT", "ETHUSDT", "BNBUSDT",
        "SOLUSDT", "XRPUSDT", "DOGEUSDT",
        "ADAUSDT", "TRXUSDT", "LINKUSDT",
    ],

    # 时间范围：UTC，包含 start，不包含 end
    "start": "2021-01-01",
    "end": "2026-01-01",

    # 数据
    "data_dir": "/Users/bohanhou/binance/data",
    "interval": "1d",            # K 线频率：1h / 1d
    "warmup_bars": 0,            # 开始前额外读取的数据，供指标预热
    "missing_data": "raise",     # 缺数据时报错

    # 账户与交易成本：以下为示例假设
    "initial_cash": 100_000,
    "fee_bps": 10,               # 单边手续费，10 bps = 0.1%
    "slippage_bps": 0,           # 单边滑点

    # 组合管理
    "rebalance": "1d",           # 调仓频率，与 K 线频率分开
    "weights": None,             # 具体形式之后再定
}


def load_data(config):
    universe = list(config["universe"])
    steps={
        "1h":pd.Timedelta(hours=1),
        "1d":pd.Timedelta(days=1),
    }

    if config["interval"] not in steps:
        raise ValueError("interval only support1h/1d")

    step = steps[config["interval"]]
    

    start = steps[config["interval"]]
    start = pd.to_datetime(config["start"], utc=True)
    latest = pd.Timestamp.now(tz="UTC").floor(step)
    end = (
        pd.to_datetime(config["end"],utc=True)
        if config.get("end")
        else latest
    )
    if start>=end or end>latest:
        raise ValueError("need start<end")
    warmup = config.get("warmup_bars",0)

    index = pd.date_range(
        start-warmup*step,
        end,
        freq=step,
        inclusive="left"
    )
    frame={}
    for symbol in universe:
        path = Path(config["data_dir"])/f"{symbol}_{config['interval']}.csv"
        df = pd.read_csv(path,usecols=["open_time","open","close"])
        df["open_time"]=pd.to_datetime(df["open_time"],utc=True)
        df=df.set_index("open_time").reindex(index)
        df=df[["open","close"]].apply(pd.to_numeric,errors="coerce")

        bad_cells=~np.isfinite(df)|(df<=0)
        bad_rows = bad_cells.any(axis=1)

        if bad_rows.any():
            warnings.warn(
                 f"{symbol}: {int(bad_rows.sum())} 行存在缺失或无效价格，"
                  f"首次出现在 {df.index[bad_rows][0]}，已设为 NaN，继续加载。",

            )
        df = df.mask(bad_cells,np.nan)
        frame[symbol] = df
        

    return{
        "open":pd.DataFrame({s:frame[s]["open"] for s in universe}),
        "close":pd.DataFrame({s:frame[s]["close"] for s in universe}),
        "step":step,
        "start":start,
        "end":end,

    }

def rebalance_key(timestamp,rule):
    if rule == "1h":
        return timestamp.floor("h")
    if rule == "1d":
        return timestamp.normalize()
    if rule =="1w":
        return timestamp.normalize()-pd.Timedelta(days=timestamp.weekday())
    if rule =="1mo":
        return timestamp.year,timestamp.month
    return None
data = load_data(config)
print(data)



def run_backtest(config):
    data =load_data(config)
    if config.get("weights") is None:
        return{"data":data,
               "summary":None}
    rule = config.get("rebalance", "1d")

    universe=list(config["universe"])
    targets=config["weights"]

    weights = np.array(
        [targets.get(s, 0.0) for s in universe],
        dtype=float,
    )
    if weights.sum() > 1:
        weights /= weights.sum()
    w = weights.to_numpy(dtype=float)

    initial = float(config.get("initial_cash", 100_000))
    fee_rate = float(config.get("fee_bps", 0)) / 10_000
    slip_rate = float(config.get("slippage_bps", 0)) / 10_000

    cash = initial
    quantity = np.zeros(len(universe))
    
    equity_values, positions, cash_values, trades = [], [], [], []
    previous_key =None

    opens=data["open"].loc[data["start"]:,universe]
    closes=data["close"].reindex(
        index=opens.index,columns=universe
    )
    weights = config["weights"].reindex(
        index=opens.index, columns=universe
    )
    last_price = np.full(len(universe), np.nan)

    for i,timestamp in enumerate(open.index):
        open_price = open.iloc[i].to_numpy(dtype=float)
        valid=np.isfinite(open_price)&(open_price>0)

        last_price[valid]=open_price[valid]
        mark = np.nan_to_num(last_price,nan=0.0)

        value= quantity*mark
        equity_before = cash+value.sum()#your account when open(equity+cash)

        key=rebalance_key(timestamp,rule)
        if i == 0 or (rule is not None and key != previous_key):
            target_value = equity_before*w[i] #target position
            sell = np.where(
                valid, np.maximum(value - target_value, 0), 0
            )

            buy = np.where(
                valid, np.maximum(target_value - value, 0), 0
            )

            cash += sell.sum() * (1-cost_rate)

            required = buy.sum()*(1+cost_rate)
            if required > cash:
                buy *= max(cash, 0) / required


            quantity[valid] += (buy[valid] - sell[valid]) / open_price[valid]
            cash = max(0.0, cash - buy.sum() * (1 + cost_rate))























