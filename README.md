# 沪深300 PE估值动态投资优化系统

## 目录结构

```
csi300_strategy/
├── fetch_data.py     # 数据抓取（akshare）
├── strategy.py       # PE百分位策略 & 信号生成
├── backtest.py       # 历史回测
├── optimize.py       # 参数优化（网格搜索）
├── report.py         # 报告生成
├── analysis.ipynb    # 主交互笔记本（日常使用入口）
├── data/             # 自动生成，存放CSV数据
└── reports/          # 自动生成，存放报告和图表
```

## 安装依赖

```bash
pip install akshare pandas numpy matplotlib jupyter
```

## 日常使用（每月一次）

在 VS Code 中打开 `analysis.ipynb`，从头到尾运行所有格子即可。

输出：
- 当前PE百分位和建议仓位
- 策略 vs 基准的绩效对比
- 净值曲线 / 仓位 / PE百分位图表
- 回撤对比图
- reports/ 目录下的CSV和摘要文本

## 策略逻辑

| PE历史百分位 | 建议仓位 |
|------------|--------|
| < 20%      | 100%   |
| 20% ~ 40%  | 75%    |
| 40% ~ 60%  | 50%    |
| 60% ~ 80%  | 25%    |
| > 80%      | 0%     |

百分位基于过去1200个交易日（约5年）的历史PE计算。

## 参数优化

如需重新优化分档阈值，在 `analysis.ipynb` 中取消注释第6格并运行，
或直接运行：

```bash
python optimize.py
```

网格搜索以夏普比率为目标，输出Top 10参数组合。
