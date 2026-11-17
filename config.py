from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FHPS_DIR = DATA_DIR / "fhps"
YJBB_DIR = DATA_DIR / "yjbb"
STOCK_LIST_DIR = DATA_DIR / "stock_lists"
PRICE_DIR = DATA_DIR / "prices"
BENCHMARK_DIR = DATA_DIR / "benchmark"
RESULTS_DIR = BASE_DIR / "results"

for d in [FHPS_DIR, YJBB_DIR, STOCK_LIST_DIR, PRICE_DIR, BENCHMARK_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

START_YEAR = 2015
END_YEAR = 2026
BACKTEST_START_DATE = "2016-05-01"
BACKTEST_END_DATE = "2026-11-16"

# 生成报告期列表：20151231, 20160630, 20161231, ..., 20260630
REPORT_PERIODS = []
for y in range(START_YEAR, END_YEAR + 1):
    if y > START_YEAR:
        REPORT_PERIODS.append(f"{y}0630")
    if y < END_YEAR:
        REPORT_PERIODS.append(f"{y}1231")

# 交易成本设置（单边：印花税仅卖出千0.5~千1 + 佣金万2.5 + 滑点）
BUY_COST_RATE = 0.0005   # 买入万5（含佣金+滑点）
SELL_COST_RATE = 0.0015  # 卖出千1.5（含印花税+佣金+滑点）
