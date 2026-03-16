"""
fetch_data.py
从 akshare 拉取沪深300指数价格及PE数据，保存到本地 CSV。
"""

import akshare as ak
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

PRICE_FILE = DATA_DIR / "csi300_price.csv"
PE_FILE    = DATA_DIR / "csi300_pe.csv"


def fetch_price(start: str = "20100101") -> pd.DataFrame:
    print("正在获取沪深300价格数据...")
    df = ak.stock_zh_index_daily(symbol="sh000300")
    df = df[["date", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= pd.to_datetime(start)]
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(PRICE_FILE, index=False)
    print(f"价格数据已保存：{PRICE_FILE}（共 {len(df)} 条）")
    return df


def fetch_pe(start: str = "20100101") -> pd.DataFrame:
    print("正在获取沪深300 PE数据...")
    df = ak.stock_index_pe_lg(symbol="沪深300")
    df = df[["日期", "滚动市盈率"]].rename(columns={"日期": "date", "滚动市盈率": "pe"})
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= pd.to_datetime(start)]
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(PE_FILE, index=False)
    print(f"PE数据已保存：{PE_FILE}（共 {len(df)} 条）")
    return df


def load_price() -> pd.DataFrame:
    if not PRICE_FILE.exists():
        return fetch_price()
    return pd.read_csv(PRICE_FILE, parse_dates=["date"])


def load_pe() -> pd.DataFrame:
    if not PE_FILE.exists():
        return fetch_pe()
    return pd.read_csv(PE_FILE, parse_dates=["date"])


def load_merged() -> pd.DataFrame:
    """合并价格和PE，返回对齐后的日频数据"""
    price = load_price()
    pe    = load_pe()
    df = pd.merge_asof(
        price.sort_values("date"),
        pe.sort_values("date"),
        on="date",
        direction="backward",
    )
    df = df.dropna(subset=["pe"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    fetch_price()
    fetch_pe()
    df = load_merged()
    print(f"\n合并后数据预览（最新5条）：")
    print(df.tail())
