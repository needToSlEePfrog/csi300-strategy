"""
optimize.py
网格搜索最优参数，以夏普比率为优化目标。
新增 Walk-forward 验证：用前 N 年训练，后 M 年验证，衡量样本外表现。

搜索空间：
  - lookback   : 历史回看窗口
  - low_pct    : 满仓分位上限
  - high_pct   : 空仓分位下限
  - ma_window  : 趋势均线窗口
"""

import itertools
import pandas as pd
import numpy as np
from backtest import run_backtest, calc_metrics
from fetch_data import load_merged

# ── 网格搜索空间 ──────────────────────────────────────────
LOOKBACK_GRID  = [800, 1000, 1200, 1500]
LOW_PCT_GRID   = [0.15, 0.20, 0.25]
HIGH_PCT_GRID  = [0.75, 0.80, 0.85]
MA_WINDOW_GRID = [200, 250]
# ─────────────────────────────────────────────────────────

# ── Walk-forward 参数 ────────────────────────────────────
TRAIN_YEARS   = 7    # 训练集年数
TEST_YEARS    = 3    # 验证集年数
# ─────────────────────────────────────────────────────────


def _sharpe_from_bt(bt: pd.DataFrame) -> float:
    m = calc_metrics(bt)
    return float(m["策略_夏普比率"])


def run_optimization(verbose: bool = True) -> pd.DataFrame:
    """全样本网格搜索，返回按夏普排序的结果 DataFrame"""
    df_raw = load_merged()
    results = []

    combos = list(itertools.product(
        LOOKBACK_GRID, LOW_PCT_GRID, HIGH_PCT_GRID, MA_WINDOW_GRID
    ))
    total = len(combos)
    print(f"开始网格搜索，共 {total} 个参数组合...\n")

    for i, (lb, lo, hi, ma) in enumerate(combos):
        if lo >= hi:
            continue
        try:
            bt     = run_backtest(lookback=lb, low_pct=lo, high_pct=hi,
                                  ma_window=ma, df=df_raw)
            m      = calc_metrics(bt)
            sharpe = float(m["策略_夏普比率"])
            cagr   = m["策略_年化收益"]
            max_dd = m["策略_最大回撤"]
            results.append({
                "lookback": lb, "low_pct": lo, "high_pct": hi, "ma_window": ma,
                "sharpe": round(sharpe, 3),
                "cagr":   cagr,
                "max_dd": max_dd,
            })
        except Exception:
            pass

        if verbose and (i + 1) % 10 == 0:
            print(f"  进度: {i+1}/{total}")

    result_df = (pd.DataFrame(results)
                 .sort_values("sharpe", ascending=False)
                 .reset_index(drop=True))
    return result_df


def run_walk_forward(verbose: bool = True) -> pd.DataFrame:
    """
    Walk-forward 验证。
    将全量数据按时间切分为多个 (train, test) 窗口，
    每个窗口在训练集上找最优参数，然后在验证集上评估。

    返回每个窗口的样本外夏普结果。
    """
    df_raw = load_merged()

    # 按交易日切分
    train_days = int(TRAIN_YEARS * 252)
    test_days  = int(TEST_YEARS  * 252)
    total_days = len(df_raw)

    windows = []
    start = 0
    while start + train_days + test_days <= total_days:
        windows.append((start, start + train_days, start + train_days + test_days))
        start += test_days   # 滚动步长 = 验证集长度

    if not windows:
        print("数据不足，无法进行 Walk-forward 验证（需要至少10年数据）")
        return pd.DataFrame()

    print(f"Walk-forward 验证：共 {len(windows)} 个窗口 "
          f"（训练{TRAIN_YEARS}年 / 验证{TEST_YEARS}年）\n")

    wf_results = []
    for idx, (s, m, e) in enumerate(windows):
        train_df = df_raw.iloc[s:m].copy().reset_index(drop=True)
        test_df  = df_raw.iloc[m:e].copy().reset_index(drop=True)

        # 在训练集上找最优参数（简化：只搜 low/high_pct，固定 lookback=1200, ma=250）
        best_sharpe = -np.inf
        best_params = {}
        for lo, hi in itertools.product(LOW_PCT_GRID, HIGH_PCT_GRID):
            if lo >= hi:
                continue
            try:
                bt = run_backtest(lookback=min(1200, len(train_df) - 1),
                                  low_pct=lo, high_pct=hi,
                                  ma_window=250, df=train_df)
                sh = _sharpe_from_bt(bt)
                if sh > best_sharpe:
                    best_sharpe = sh
                    best_params = {"low_pct": lo, "high_pct": hi}
            except Exception:
                pass

        if not best_params:
            continue

        # 在验证集上评估
        try:
            bt_test   = run_backtest(lookback=min(1200, len(test_df) - 1),
                                     low_pct=best_params["low_pct"],
                                     high_pct=best_params["high_pct"],
                                     ma_window=250, df=test_df)
            m_test    = calc_metrics(bt_test)
            oos_sharpe = float(m_test["策略_夏普比率"])
            wf_results.append({
                "窗口":       idx + 1,
                "训练起止":   f"{train_df['date'].iloc[0].date()} ~ {train_df['date'].iloc[-1].date()}",
                "验证起止":   f"{test_df['date'].iloc[0].date()} ~ {test_df['date'].iloc[-1].date()}",
                "最优low_pct":  best_params["low_pct"],
                "最优high_pct": best_params["high_pct"],
                "训练集夏普":   round(best_sharpe, 3),
                "验证集夏普":   round(oos_sharpe, 3),
                "验证集年化":   m_test["策略_年化收益"],
                "验证集最大回撤": m_test["策略_最大回撤"],
            })
            if verbose:
                print(f"  窗口{idx+1}: 训练夏普={best_sharpe:.2f} → "
                      f"验证夏普={oos_sharpe:.2f}  "
                      f"（参数 lo={best_params['low_pct']} hi={best_params['high_pct']}）")
        except Exception:
            pass

    return pd.DataFrame(wf_results)


def best_params(top_n: int = 1) -> dict:
    """返回全样本夏普最高的参数组合"""
    result_df = run_optimization(verbose=False)
    best = result_df.iloc[0]
    return {
        "lookback":  int(best["lookback"]),
        "low_pct":   best["low_pct"],
        "high_pct":  best["high_pct"],
        "ma_window": int(best["ma_window"]),
        "sharpe":    best["sharpe"],
        "cagr":      best["cagr"],
        "max_dd":    best["max_dd"],
    }


if __name__ == "__main__":
    # 全样本优化
    print("=" * 50)
    print("【全样本网格搜索】")
    results = run_optimization()
    print("\n=== Top 10 参数组合（按夏普排序）===")
    print(results.head(10).to_string(index=False))

    # Walk-forward 验证
    print("\n" + "=" * 50)
    print("【Walk-forward 样本外验证】")
    wf = run_walk_forward()
    if not wf.empty:
        print("\n=== Walk-forward 结果 ===")
        print(wf.to_string(index=False))
        avg_oos = wf["验证集夏普"].mean()
        print(f"\n验证集平均夏普：{avg_oos:.3f}")
        degradation = wf["训练集夏普"].mean() - avg_oos
        print(f"训练→验证 夏普衰减：{degradation:.3f}  "
              f"{'（衰减较大，注意过拟合风险）' if degradation > 0.5 else '（衰减正常）'}")
