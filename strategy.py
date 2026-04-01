"""
strategy.py
基于 PE+PB 双因子百分位生成沪深300仓位信号。

改动（v2）：
  1. 新增 PB 历史百分位，与 PE 百分位取均值作为合成估值分位
  2. 仓位由硬分档改为线性平滑（连续映射）
  3. 新增趋势过滤：收盘价低于250日均线时仓位强制归零

合成估值分位 → 仓位映射（线性平滑）：
  分位 ≤ LOW_PCT  → 满仓  MAX_POS
  分位 ≥ HIGH_PCT → 空仓  0.0
  中间             → 线性插值
"""

import pandas as pd
import numpy as np
from fetch_data import load_merged

# ── 默认参数 ──────────────────────────────────────────────
DEFAULT_LOOKBACK  = 1200      # 历史回看窗口（交易日），约5年
DEFAULT_MA_WINDOW = 250       # 趋势均线窗口
DEFAULT_LOW_PCT   = 0.20      # 低于此分位 → 满仓
DEFAULT_HIGH_PCT  = 0.80      # 高于此分位 → 空仓
DEFAULT_MAX_POS   = 1.00      # 满仓上限
# ─────────────────────────────────────────────────────────


def calc_percentile(series: pd.Series, lookback: int) -> pd.Series:
    """计算滚动历史百分位（当前值在过去 lookback 期内的排名）"""
    return series.rolling(window=lookback, min_periods=lookback // 2).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )


def linear_position(
    composite_pct: float,
    low_pct: float  = DEFAULT_LOW_PCT,
    high_pct: float = DEFAULT_HIGH_PCT,
    max_pos: float  = DEFAULT_MAX_POS,
) -> float:
    """
    将合成估值分位线性映射到仓位。
      分位 ≤ low_pct  → max_pos（满仓）
      分位 ≥ high_pct → 0.0（空仓）
      中间            → 线性插值
    """
    if composite_pct <= low_pct:
        return max_pos
    if composite_pct >= high_pct:
        return 0.0
    # 线性插值
    ratio = (composite_pct - low_pct) / (high_pct - low_pct)
    return round(max_pos * (1 - ratio), 4)


def generate_signals(
    df: pd.DataFrame = None,
    lookback: int    = DEFAULT_LOOKBACK,
    ma_window: int   = DEFAULT_MA_WINDOW,
    low_pct: float   = DEFAULT_LOW_PCT,
    high_pct: float  = DEFAULT_HIGH_PCT,
    max_pos: float   = DEFAULT_MAX_POS,
) -> pd.DataFrame:
    """
    生成完整信号序列。
    返回含以下列的 DataFrame：
      date, close, pe, pb, pe_pct, pb_pct, composite_pct,
      ma250, trend_up, raw_position, position
    """
    if df is None:
        df = load_merged()

    df = df.copy()

    # 1. 估值百分位
    df["pe_pct"]        = calc_percentile(df["pe"], lookback)
    df["pb_pct"]        = calc_percentile(df["pb"], lookback)
    df["composite_pct"] = (df["pe_pct"] + df["pb_pct"]) / 2

    df = df.dropna(subset=["composite_pct"]).copy()

    # 2. 趋势过滤
    df["ma250"]     = df["close"].rolling(window=ma_window, min_periods=ma_window // 2).mean()
    df["trend_up"]  = df["close"] >= df["ma250"]

    # 3. 线性仓位
    df["raw_position"] = df["composite_pct"].apply(
        lambda x: linear_position(x, low_pct, high_pct, max_pos)
    )

    # 4. 趋势过滤：均线以下仓位归零
    df["position"] = df.apply(
        lambda row: row["raw_position"] if row["trend_up"] else 0.0, axis=1
    )

    cols = ["date", "close", "pe", "pb",
            "pe_pct", "pb_pct", "composite_pct",
            "ma250", "trend_up", "raw_position", "position"]
    return df[cols].reset_index(drop=True)


def current_signal(
    lookback: int  = DEFAULT_LOOKBACK,
    ma_window: int = DEFAULT_MA_WINDOW,
    low_pct: float = DEFAULT_LOW_PCT,
    high_pct: float= DEFAULT_HIGH_PCT,
) -> dict:
    """返回当前最新一天的信号"""
    signals = generate_signals(lookback=lookback, ma_window=ma_window,
                               low_pct=low_pct, high_pct=high_pct)
    latest = signals.iloc[-1]
    return {
        "date":           latest["date"].strftime("%Y-%m-%d"),
        "close":          round(latest["close"], 2),
        "pe":             round(latest["pe"], 2),
        "pb":             round(latest["pb"], 2),
        "pe_pct":         round(latest["pe_pct"] * 100, 1),
        "pb_pct":         round(latest["pb_pct"] * 100, 1),
        "composite_pct":  round(latest["composite_pct"] * 100, 1),
        "trend_up":       bool(latest["trend_up"]),
        "raw_position":   round(latest["raw_position"] * 100, 1),
        "position":       round(latest["position"] * 100, 1),
    }


if __name__ == "__main__":
    sig = current_signal()
    print("\n=== 当前信号 ===")
    print(f"日期          : {sig['date']}")
    print(f"收盘价        : {sig['close']}")
    print(f"PE (TTM)      : {sig['pe']}  （历史百分位 {sig['pe_pct']}%）")
    print(f"PB            : {sig['pb']}  （历史百分位 {sig['pb_pct']}%）")
    print(f"合成估值分位  : {sig['composite_pct']}%")
    print(f"趋势（>MA250）: {'是' if sig['trend_up'] else '否'}")
    print(f"估值建议仓位  : {sig['raw_position']}%")
    print(f"▶ 最终建议仓位: {sig['position']}%  {'（趋势过滤已触发，强制空仓）' if not sig['trend_up'] else ''}")
