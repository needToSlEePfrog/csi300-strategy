"""
optimize.py
网格搜索最优PE百分位分档阈值，以夏普比率为优化目标。

搜索空间：
  - lookback  : 历史回看窗口
  - tiers     : 五档仓位的百分位分界点

耗时提示：全量网格搜索较慢，建议缩小范围后运行。
"""

import itertools
import pandas as pd
import numpy as np
from backtest import run_backtest, calc_metrics
from fetch_data import load_merged

# ── 搜索空间 ──────────────────────────────────────────────
LOOKBACK_GRID = [800, 1000, 1200, 1500]

# 各分界点候选值（百分位，升序）
TIER1_GRID = [0.15, 0.20, 0.25]   # 满仓→重仓 分界
TIER2_GRID = [0.35, 0.40, 0.45]   # 重仓→半仓 分界
TIER3_GRID = [0.55, 0.60, 0.65]   # 半仓→轻仓 分界
TIER4_GRID = [0.75, 0.80, 0.85]   # 轻仓→空仓 分界
# ─────────────────────────────────────────────────────────


def build_tiers(t1, t2, t3, t4):
    return [
        (t1, 1.00),
        (t2, 0.75),
        (t3, 0.50),
        (t4, 0.25),
        (1.01, 0.00),
    ]


def run_optimization(verbose: bool = True) -> pd.DataFrame:
    """遍历参数组合，返回所有结果排序后的 DataFrame"""
    df_raw = load_merged()  # 预加载，避免重复IO
    results = []

    combos = list(itertools.product(LOOKBACK_GRID, TIER1_GRID, TIER2_GRID, TIER3_GRID, TIER4_GRID))
    total = len(combos)
    print(f"开始网格搜索，共 {total} 个参数组合...\n")

    for i, (lb, t1, t2, t3, t4) in enumerate(combos):
        if not (t1 < t2 < t3 < t4):
            continue
        tiers = build_tiers(t1, t2, t3, t4)
        try:
            bt = run_backtest(lookback=lb, tiers=tiers)
            m = calc_metrics(bt)
            sharpe = float(m["策略_夏普比率"])
            cagr   = float(m["策略_年化收益"].strip("%")) / 100
            max_dd = float(m["策略_最大回撤"].strip("%")) / 100
            results.append({
                "lookback": lb,
                "tier1": t1, "tier2": t2, "tier3": t3, "tier4": t4,
                "sharpe":  round(sharpe, 3),
                "cagr":    f"{cagr*100:.1f}%",
                "max_dd":  f"{max_dd*100:.1f}%",
            })
        except Exception as e:
            pass

        if verbose and (i + 1) % 20 == 0:
            print(f"  进度: {i+1}/{total}")

    result_df = pd.DataFrame(results).sort_values("sharpe", ascending=False).reset_index(drop=True)
    return result_df


def best_params(top_n: int = 1) -> dict:
    """返回夏普最高的参数组合"""
    result_df = run_optimization(verbose=False)
    best = result_df.iloc[0]
    tiers = build_tiers(best["tier1"], best["tier2"], best["tier3"], best["tier4"])
    return {
        "lookback": int(best["lookback"]),
        "tiers": tiers,
        "sharpe": best["sharpe"],
        "cagr": best["cagr"],
        "max_dd": best["max_dd"],
    }


if __name__ == "__main__":
    results = run_optimization()
    print("\n=== Top 10 参数组合（按夏普排序）===")
    print(results.head(10).to_string(index=False))
    print(f"\n最优参数：lookback={int(results.iloc[0]['lookback'])}, "
          f"tiers={results.iloc[0]['tier1']}/{results.iloc[0]['tier2']}/"
          f"{results.iloc[0]['tier3']}/{results.iloc[0]['tier4']}")
