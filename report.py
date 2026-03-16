"""
report.py
生成月度投资报告：控制台摘要 + CSV明细。
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from strategy import current_signal, DEFAULT_LOOKBACK, DEFAULT_TIERS
from backtest import run_backtest, calc_metrics

REPORT_DIR = Path(__file__).parent / "reports"
REPORT_DIR.mkdir(exist_ok=True)


def print_summary(signal: dict, metrics: dict):
    date_str = datetime.today().strftime("%Y年%m月%d日")
    print("=" * 45)
    print(f"  沪深300 PE估值策略 月度报告")
    print(f"  生成时间：{date_str}")
    print("=" * 45)

    print("\n【当前信号】")
    print(f"  最新日期  : {signal['date']}")
    print(f"  沪深300   : {signal['close']}")
    print(f"  PE (TTM)  : {signal['pe']}")
    print(f"  PE历史百分位 : {signal['pe_pct']}%")
    print(f"  ▶ 建议仓位 : {int(signal['position'] * 100)}%")

    print("\n【策略 vs 基准（全历史）】")
    keys = ["总收益", "年化收益", "最大回撤", "年化波动", "夏普比率", "卡玛比率"]
    for k in keys:
        s = metrics.get(f"策略_{k}", "-")
        b = metrics.get(f"基准_{k}", "-")
        print(f"  {k:<8}: 策略 {s:>8}  |  基准 {b:>8}")

    print(f"\n  回测区间：{metrics['回测起始']} → {metrics['回测结束']}")
    print("=" * 45)


def save_report(bt: pd.DataFrame, signal: dict, metrics: dict):
    today = datetime.today().strftime("%Y%m%d")

    # 净值曲线
    nav_file = REPORT_DIR / f"nav_{today}.csv"
    bt[["date", "strategy_nav", "benchmark_nav", "position", "pe", "pe_pct"]].to_csv(
        nav_file, index=False
    )

    # 摘要
    summary_file = REPORT_DIR / f"summary_{today}.txt"
    import io, sys
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    print_summary(signal, metrics)
    sys.stdout = old_stdout
    summary_file.write_text(buf.getvalue(), encoding="utf-8")

    print(f"\n报告已保存：")
    print(f"  净值曲线：{nav_file}")
    print(f"  摘要文本：{summary_file}")


def generate_report(save: bool = True):
    signal  = current_signal(lookback=DEFAULT_LOOKBACK, tiers=DEFAULT_TIERS)
    bt      = run_backtest(lookback=DEFAULT_LOOKBACK, tiers=DEFAULT_TIERS)
    metrics = calc_metrics(bt)

    print_summary(signal, metrics)

    if save:
        save_report(bt, signal, metrics)

    return signal, bt, metrics


if __name__ == "__main__":
    generate_report()
