import time
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ===== 修改这里 =====
SYMBOLS = [
    "BTCUSDT",   # Bitcoin
    "ETHUSDT",   # Ethereum
    "BNBUSDT",   # BNB
    "SOLUSDT",   # Solana
    "XRPUSDT",   # XRP
    "DOGEUSDT",  # Dogecoin
    "ADAUSDT",   # Cardano
    "TRXUSDT",   # TRON
    "LINKUSDT",  # Chainlink
]

INTERVALS = ["1h", "1d"]
START = "2020-01-01"
END = None
OUTPUT_DIR = Path("/Users/bohanhou/binance")
END = None                    # None = 至今；也可填 "2026-01-01"，不含当天
OUTPUT_DIR = Path("/Users/bohanhou/binance/data")
# ====================

BASE_URL = "https://data-api.binance.vision"

COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trade_count",
    "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
]

session = requests.Session()
session.mount(
    "https://",
    HTTPAdapter(
        max_retries=Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            respect_retry_after_header=True,
        )
    ),
)


def get_json(path, params=None):
    response = session.get(
        BASE_URL + path,
        params=params,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def to_ms(value):
    return int(pd.to_datetime(value, utc=True).timestamp() * 1000)


def download_klines(symbol, interval, start, end=None):
    start_ms = to_ms(start)
    server_now = int(get_json("/api/v3/time")["serverTime"])
    end_ms = min(to_ms(end), server_now) if end else server_now

    if start_ms >= end_ms:
        raise ValueError("START 必须早于 END 或当前时间")

    rows = []
    cursor = start_ms

    while cursor < end_ms:
        batch = get_json(
            "/api/v3/klines",
            params={
                "symbol": symbol.upper(),
                "interval": interval,
                "startTime": cursor,
                "endTime": end_ms - 1,
                "limit": 1000,
                "timeZone": "0",  # UTC
            },
        )

        if not batch:
            break

        # 仅保留在结束时间之前已经收盘的 K 线
        rows.extend(row for row in batch if int(row[6]) < end_ms)

        next_cursor = int(batch[-1][0]) + 1
        if next_cursor <= cursor:
            raise RuntimeError("接口返回时间没有前进，停止下载")

        cursor = next_cursor
        print(f"{symbol} {interval}: 已获取 {len(rows)} 根")

        if len(batch) < 1000:
            break

        time.sleep(0.3)

    df = pd.DataFrame(rows, columns=COLUMNS)
    df = df.drop_duplicates("open_time").sort_values("open_time")
    df = df.drop(columns="ignore")

    for col in ("open_time", "close_time"):
        df[col] = pd.to_datetime(df[col], unit="ms", utc=True)

    for col in df.columns:
        if col not in ("open_time", "close_time"):
            df[col] = pd.to_numeric(df[col])

    return df.reset_index(drop=True)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with session:
        for symbol in SYMBOLS:
            for interval in INTERVALS:
                df = download_klines(symbol, interval, START, END)

                if df.empty:
                    print(f"{symbol} {interval}: 该时间范围没有完整 K 线")
                    continue

                output = OUTPUT_DIR / f"{symbol.upper()}_{interval}.csv"
                df.to_csv(output, index=False)
                print(f"已保存：{output}，共 {len(df)} 行")