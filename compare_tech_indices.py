import pandas as pd
import akshare as ak
import baostock as bs
import matplotlib.pyplot as plt

from config import DATA_DIR, BENCHMARK_DIR, RESULTS_DIR, BACKTEST_END_DATE
from backtest import load_all_prices, calc_metrics
from backtest_scoring_guarded import run_guarded_scoring_backtest

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def download_tech_benchmarks():
    """下载科创50(sh.000688, AkShare)与创业板指(sz.399006, BaoStock)历史日线"""
    kc_path = BENCHMARK_DIR / "sh.000688.parquet"
    if not kc_path.exists() or pd.read_parquet(kc_path).empty:
        df_kc = ak.index_zh_daily_sina(symbol="sh000688")
        df_kc["date"] = df_kc["date"].astype(str)
        df_kc = df_kc[df_kc["date"] <= BACKTEST_END_DATE].copy()
        df_kc["code"] = "sh.000688"
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df_kc.columns:
                df_kc[col] = pd.to_numeric(df_kc[col], errors="coerce")
        df_kc.to_parquet(kc_path, index=False)
        print(f"[AkShare] sh.000688 已保存: {len(df_kc)} 条 ({df_kc['date'].min()} ~ {df_kc['date'].max()})")

    cy_path = BENCHMARK_DIR / "sz.399006.parquet"
    if not cy_path.exists() or pd.read_parquet(cy_path).empty:
        lg = bs.login()
        if lg.error_code != "0":
            raise RuntimeError(f"BaoStock 登录失败: {lg.error_msg}")
        try:
            rs = bs.query_history_k_data_plus(
                "sz.399006",
                "date,code,open,high,low,close,volume,amount",
                start_date="2016-04-01",
                end_date=BACKTEST_END_DATE,
                frequency="d",
            )
            rows = []
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())
            df = pd.DataFrame(rows, columns=rs.fields)
            for col in ["open", "high", "low", "close", "volume", "amount"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df.to_parquet(cy_path, index=False)
            print(f"[BaoStock] sz.399006 已保存: {len(df)} 条 ({df['date'].min()} ~ {df['date'].max()})")
        finally:
            bs.logout()


def build_window_df(nav10: pd.Series, start_date: str) -> pd.DataFrame:
    """对齐各指数并从 start_date 起归一化为 1.0"""
    sub_nav = nav10.loc[start_date:].copy()
    df = pd.DataFrame({"红利多因子Top10": sub_nav / sub_nav.iloc[0]})

    for code, label in [
        ("sh.000688", "科创50"),
        ("sz.399006", "创业板指"),
        ("sh.000922", "中证红利"),
        ("sh.000300", "沪深300"),
    ]:
        df_idx = pd.read_parquet(BENCHMARK_DIR / f"{code}.parquet")
        df_idx["date"] = pd.to_datetime(df_idx["date"])
        s = df_idx.set_index("date")["close"].reindex(df.index).ffill()
        df[label] = s / s.iloc[0]
    return df


def main():
    download_tech_benchmarks()

    # 重算 Top 10 日度净值
    sched = pd.read_parquet(DATA_DIR / "rebalance_schedule.parquet")
    trade_dates = pd.read_parquet(DATA_DIR / "trade_dates.parquet")
    td_index = pd.to_datetime(trade_dates["calendar_date"]).sort_values().reset_index(drop=True)
    prices = load_all_prices()
    hfq_dict = {code: df["close_hfq"] for code, df in prices.items()}
    df_hfq_all = pd.DataFrame(hfq_dict).sort_index().ffill()

    w_best = {"div": 0.40, "pe": 0.25, "mc": 0.20, "roe": 0.15}
    nav10, _ = run_guarded_scoring_backtest(
        sched, td_index, prices, df_hfq_all, "红利多因子Top10",
        0.0, 25.0, 6.0, 18.0, w_best, top_n=10,
    )

    df_kc = pd.read_parquet(BENCHMARK_DIR / "sh.000688.parquet")
    kc_start = pd.to_datetime(df_kc["date"]).min().strftime("%Y-%m-%d")

    windows = [
        (f"科创50起步至今 ({kc_start} ~ 2026-11-16)", kc_start),
        ("2024年9月至今 (2024-09-02 ~ 2026-11-16)", "2024-09-02"),
    ]

    all_summaries = []
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    colors = {
        "红利多因子Top10": "#d62728",
        "科创50": "#9467bd",
        "创业板指": "#2ca02c",
        "中证红利": "#1f77b4",
        "沪深300": "#7f7f7f",
    }

    for ax, (title, start_dt) in zip(axes, windows):
        df_w = build_window_df(nav10, start_dt)
        summary = pd.DataFrame({col: calc_metrics(df_w[col]) for col in df_w.columns}).T
        summary.insert(0, "对比区间", title)
        all_summaries.append(summary)

        print(f"\n=== {title} ===")
        print(summary.drop(columns=["对比区间"]).to_string())

        for col in df_w.columns:
            ls = "--" if col in ("中证红利", "沪深300") else "-"
            lw = 2.3 if col == "红利多因子Top10" else (1.9 if col in ("科创50", "创业板指") else 1.3)
            ax.plot(df_w.index, df_w[col], label=col, color=colors.get(col), linestyle=ls, linewidth=lw)
        ax.set_title(title, fontsize=11.5)
        ax.set_ylabel("区间累计净值")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left", fontsize=8.5)

    pd.concat(all_summaries).to_csv(RESULTS_DIR / "compare_tech_summary.csv")
    fig.suptitle("红利多因子 Top 10 vs 科创50 / 创业板指 收益对比", fontsize=14, y=1.02)
    fig.tight_layout()
    chart_path = RESULTS_DIR / "compare_tech_indices.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    print(f"\n📊 对比图已保存至: {chart_path}")


if __name__ == "__main__":
    main()
