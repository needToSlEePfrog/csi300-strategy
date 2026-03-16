"""
backtest.py
对 strategy.py 生成的信号进行历史回测。

假设：
  - 每日收盘后按信号调仓（理想化，忽略滑点）
  - 初始资金 100,000 元
  - 无交易费用（可扩展）
  - 对比基准：全程满仓持有沪深300
"""

import pandas as pd
import numpy as np
from strategy import generate_signals, DEFAULT_LOOKBACK, DEFAULT_TIERS

INITIAL_CAPITAL = 100_000.0


def run_backtest(
    lookback: int = DEFAULT_LOOKBACK,
    tiers: list = DEFAULT_TIERS,
    initial_capital: float = INITIAL_CAPITAL,
) -> pd.DataFrame:
    """
    执行回测，返回每日净值序列。
    列：date, close, position, strategy_nav, benchmark_nav
    """
    signals = generate_signals(lookback=lookback, tiers=tiers)

    # 日收益率
    signals["daily_return"] = signals["close"].pct_change().fillna(0)

    # 策略收益：前一日仓位 × 当日涨跌
    signals["strategy_return"] = signals["position"].shift(1).fillna(0) * signals["daily_return"]

    # 净值
    signals["strategy_nav"]  = initial_capital * (1 + signals["strategy_return"]).cumprod()
    signals["benchmark_nav"] = initial_capital * (1 + signals["daily_return"]).cumprod()

    return signals[["date", "close", "pe", "pe_pct", "position",
                     "daily_return", "strategy_return",
                     "strategy_nav", "benchmark_nav"]].reset_index(drop=True)


def calc_metrics(bt: pd.DataFrame) -> dict:
    """计算关键绩效指标"""
    def _metrics(nav_series, return_series, label):
        total_return = nav_series.iloc[-1] / nav_series.iloc[0] - 1
        n_years = len(nav_series) / 252
        cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        rolling_max = nav_series.cummax()
        drawdown = (nav_series - rolling_max) / rolling_max
        max_dd = drawdown.min()

        annual_vol = return_series.std() * np.sqrt(252)
        sharpe = (return_series.mean() * 252) / (annual_vol + 1e-9)

        calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan

        return {
            f"{label}_总收益":   f"{total_return*100:.1f}%",
            f"{label}_年化收益": f"{cagr*100:.1f}%",
            f"{label}_最大回撤": f"{max_dd*100:.1f}%",
            f"{label}_年化波动": f"{annual_vol*100:.1f}%",
            f"{label}_夏普比率": f"{sharpe:.2f}",
            f"{label}_卡玛比率": f"{calmar:.2f}",
        }

    m = {}
    m.update(_metrics(bt["strategy_nav"],  bt["strategy_return"],  "策略"))
    m.update(_metrics(bt["benchmark_nav"], bt["daily_return"],      "基准"))
    m["回测起始"] = bt["date"].iloc[0].strftime("%Y-%m-%d")
    m["回测结束"] = bt["date"].iloc[-1].strftime("%Y-%m-%d")
    m["回测天数"] = len(bt)
    return m


if __name__ == "__main__":
    bt = run_backtest()
    metrics = calc_metrics(bt)
    print("\n=== 回测绩效 ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
