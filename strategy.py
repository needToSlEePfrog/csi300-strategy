"""
strategy.py
基于PE历史百分位生成沪深300仓位信号。

逻辑：
  - 计算当前PE在过去 lookback 天内的历史百分位
  - 百分位越低（低估）→ 仓位越高
  - 百分位越高（高估）→ 仓位越低

默认分档（可在 optimize.py 中调优）：
  百分位 < 20%  → 满仓  100%
  百分位 20-40% → 重仓   75%
  百分位 40-60% → 半仓   50%
  百分位 60-80% → 轻仓   25%
  百分位 > 80%  → 空仓    0%
"""

import pandas as pd
import numpy as np
from fetch_data import load_merged

# ── 默认参数 ──────────────────────────────────────────────
DEFAULT_LOOKBACK = 1200   # 历史回看窗口（交易日），约5年
DEFAULT_TIERS = [
    (0.20, 1.00),
    (0.40, 0.75),
    (0.60, 0.50),
    (0.80, 0.25),
    (1.01, 0.00),
]
# ─────────────────────────────────────────────────────────


def calc_pe_percentile(df: pd.DataFrame, lookback: int = DEFAULT_LOOKBACK) -> pd.DataFrame:
    """计算每日PE在过去 lookback 天内的历史百分位"""
    df = df.copy()
    df["pe_pct"] = df["pe"].rolling(window=lookback, min_periods=lookback // 2).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    return df


def pe_to_position(pe_pct: float, tiers: list = DEFAULT_TIERS) -> float:
    """将百分位映射到仓位比例"""
    for threshold, position in tiers:
        if pe_pct < threshold:
            return position
    return 0.0


def generate_signals(
    df: pd.DataFrame = None,
    lookback: int = DEFAULT_LOOKBACK,
    tiers: list = DEFAULT_TIERS,
) -> pd.DataFrame:
    """
    生成完整信号序列。
    返回含以下列的 DataFrame：
      date, close, pe, pe_pct, position
    """
    if df is None:
        df = load_merged()

    df = calc_pe_percentile(df, lookback)
    df = df.dropna(subset=["pe_pct"]).copy()
    df["position"] = df["pe_pct"].apply(lambda x: pe_to_position(x, tiers))
    return df[["date", "close", "pe", "pe_pct", "position"]].reset_index(drop=True)


def current_signal(lookback: int = DEFAULT_LOOKBACK, tiers: list = DEFAULT_TIERS) -> dict:
    """返回当前最新一天的信号"""
    signals = generate_signals(lookback=lookback, tiers=tiers)
    latest = signals.iloc[-1]
    return {
        "date":       latest["date"].strftime("%Y-%m-%d"),
        "close":      round(latest["close"], 2),
        "pe":         round(latest["pe"], 2),
        "pe_pct":     round(latest["pe_pct"] * 100, 1),   # 转成百分比显示
        "position":   latest["position"],
    }


if __name__ == "__main__":
    sig = current_signal()
    print("\n=== 当前信号 ===")
    print(f"日期      : {sig['date']}")
    print(f"收盘价    : {sig['close']}")
    print(f"PE (TTM)  : {sig['pe']}")
    print(f"PE百分位  : {sig['pe_pct']}%")
    print(f"建议仓位  : {int(sig['position'] * 100)}%")
