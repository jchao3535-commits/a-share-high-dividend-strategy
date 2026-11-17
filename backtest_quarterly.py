import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    DATA_DIR,
    YJBB_DIR,
    STOCK_LIST_DIR,
    RESULTS_DIR,
    BACKTEST_END_DATE,
    BUY_COST_RATE,
    SELL_COST_RATE,
)
from backtest import load_all_prices, _load_clean_fhps, calc_metrics

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def build_schedule(months: list[int]) -> list[dict]:
    """根据指定调仓月份生成 2016-05 ~ 2026-11 的调仓日历与对应财报/TTM分红窗口"""
    td = pd.to_datetime(pd.read_parquet(DATA_DIR / "trade_dates.parquet")["calendar_date"]).sort_values()
    sched = []
    for y in range(2016, 2027):
        for m in sorted(months):
            if y == 2016 and m < 5:
                continue
            target = pd.Timestamp(f"{y}-{m:02d}-01")
            if target > pd.Timestamp(BACKTEST_END_DATE):
                continue
            valid = td[td >= target]
            if valid.empty:
                continue
            reb_date = valid.iloc[0].strftime("%Y-%m-%d")

            if m == 5:
                yj_period, roe_scale = f"{y-1}1231", 1.0
                fh_periods = [f"{y-1}1231", f"{y-1}0630", f"{y-1}0930"]
            elif m == 9:
                yj_period, roe_scale = f"{y}0630", 2.0
                fh_periods = [f"{y-1}1231", f"{y-1}0930", f"{y}0630"]
            elif m == 11:
                yj_period, roe_scale = f"{y}0930", 4.0 / 3.0
                fh_periods = [f"{y-1}1231", f"{y}0630", f"{y}0930"]
            elif m == 2:
                yj_period, roe_scale = f"{y-1}0930", 4.0 / 3.0
                fh_periods = [f"{y-2}1231", f"{y-1}0630", f"{y-1}0930"]
            else:
                raise ValueError(f"Unsupported month: {m}")

            sched.append({
                "rebalance_date": reb_date,
                "yj_period": yj_period,
                "roe_scale": roe_scale,
                "fh_periods": fh_periods,
            })
    return sched


def get_quarterly_candidates(item: dict, prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    reb_date = item["rebalance_date"]
    reb_ts = pd.Timestamp(reb_date)

    df_st = pd.read_parquet(STOCK_LIST_DIR / f"{reb_date}.parquet")
    df_st = df_st[(df_st["tradeStatus"] == "1") & (~df_st["code_name"].str.contains("ST|退", na=False))].copy()

    # 汇总近12个月分红
    div_frames = []
    for p in item["fh_periods"]:
        df_f = _load_clean_fhps(p, reb_ts)
        if not df_f.empty:
            div_frames.append(df_f)
    if not div_frames:
        return pd.DataFrame()

    df_div_all = pd.concat(div_frames, ignore_index=True)
    df_div = df_div_all.groupby("symbol", as_index=False).agg(
        div_per_10_ttm=("div_per_10", "sum"),
        total_shares=("total_shares", "last"),
    )
    df_div = df_div[df_div["div_per_10_ttm"] > 0]

    df_yj = pd.read_parquet(YJBB_DIR / f"{item['yj_period']}.parquet")
    df_yj["symbol"] = df_yj["股票代码"].astype(str).str.zfill(6)
    df_yj["eps"] = pd.to_numeric(df_yj["每股收益"], errors="coerce")
    df_yj["net_profit"] = pd.to_numeric(df_yj["净利润-净利润"], errors="coerce")
    df_yj["roe"] = pd.to_numeric(df_yj["净资产收益率"], errors="coerce")
    df_yj = df_yj.drop_duplicates("symbol", keep="last")

    merged = df_st.merge(df_div, on="symbol", how="inner")
    merged = merged.merge(df_yj[["symbol", "eps", "net_profit", "roe"]], on="symbol", how="left")
    merged = merged[(merged["eps"] > 0) & (merged["net_profit"] > 0)].copy()

    records = []
    for _, r in merged.iterrows():
        code = r["code"]
        if code not in prices:
            continue
        df_p = prices[code]
        if reb_ts not in df_p.index:
            continue
        day_row = df_p.loc[reb_ts]
        if day_row["tradestatus"] != 1 or pd.isna(day_row["volume"]) or day_row["volume"] <= 0:
            continue
        close_raw = day_row["close_raw"]
        if pd.isna(close_raw) or close_raw <= 0 or pd.isna(r["total_shares"]):
            continue

        div_yield = (r["div_per_10_ttm"] / 10.0) / close_raw
        if div_yield <= 0 or div_yield > 0.20:
            continue

        roe_ann = r["roe"] * item["roe_scale"] if pd.notna(r["roe"]) else np.nan
        records.append({
            "code": code,
            "name": r["code_name"],
            "div_yield": div_yield,
            "peTTM": day_row["peTTM"],
            "mkt_cap_yi": close_raw * r["total_shares"] / 1e8,
            "roe_ann": roe_ann,
        })
    return pd.DataFrame(records)


def run_freq_backtest(
    sched: list[dict],
    td_index: pd.DatetimeIndex,
    prices: dict[str, pd.DataFrame],
    df_hfq_all: pd.DataFrame,
    name: str,
    top_n: int = 10,
) -> tuple[pd.Series, pd.DataFrame]:
    w = {"div": 0.40, "pe": 0.25, "mc": 0.20, "roe": 0.15}
    nav_series_list = []
    period_stats = []
    current_nav = 1.0
    prev_holdings: set[str] = set()
    end_ts = pd.Timestamp(BACKTEST_END_DATE)

    for idx, item in enumerate(sched):
        reb_date = item["rebalance_date"]
        reb_ts = pd.Timestamp(reb_date)
        next_reb_ts = pd.Timestamp(sched[idx + 1]["rebalance_date"]) if idx + 1 < len(sched) else end_ts

        df = get_quarterly_candidates(item, prices)
        df = df[
            (df["peTTM"] > 0)
            & (df["peTTM"] < 25.0)
            & (df["roe_ann"].notna())
            & (df["roe_ann"] >= 6.0)
            & (df["roe_ann"] <= 18.0)
        ].copy()

        df["s_div"] = df["div_yield"].rank(pct=True)
        df["s_pe"] = (-df["peTTM"]).rank(pct=True)
        df["s_mc"] = df["mkt_cap_yi"].rank(pct=True)
        df["s_roe"] = (-np.abs(df["roe_ann"] - 11.5)).rank(pct=True)
        df["total_score"] = (
            w["div"] * df["s_div"]
            + w["pe"] * df["s_pe"]
            + w["mc"] * df["s_mc"]
            + w["roe"] * df["s_roe"]
        )

        df_pick = df.sort_values("total_score", ascending=False).head(top_n)
        new_holdings = set(df_pick["code"].tolist())

        if not prev_holdings:
            turnover_buy, turnover_sell = 1.0, 0.0
        else:
            kept = len(prev_holdings & new_holdings)
            turnover_buy = (len(new_holdings) - kept) / len(new_holdings)
            turnover_sell = (len(prev_holdings) - kept) / len(prev_holdings)

        current_nav *= (1.0 - turnover_buy * BUY_COST_RATE - turnover_sell * SELL_COST_RATE)

        sub_dates = td_index[(td_index >= reb_ts) & (td_index <= next_reb_ts)]
        sub_hfq = df_hfq_all.reindex(sub_dates)[list(new_holdings)]
        rel_prices = sub_hfq.div(sub_hfq.iloc[0], axis=1)
        port_rel = rel_prices.mean(axis=1)
        period_nav = current_nav * port_rel

        nav_series_list.append(period_nav.iloc[1:] if nav_series_list else period_nav)
        period_stats.append({
            "调仓日": reb_date,
            "财报期": item["yj_period"],
            "候选数": len(df),
            "平均股息率": f"{df_pick['div_yield'].mean():.2%}",
            "区间收益": f"{(port_rel.iloc[-1] - 1.0):.2%}",
        })
        current_nav = period_nav.iloc[-1]
        prev_holdings = new_holdings

    nav = pd.concat(nav_series_list)
    nav.name = name
    return nav, pd.DataFrame(period_stats)


def main():
    trade_dates = pd.read_parquet(DATA_DIR / "trade_dates.parquet")
    td_index = pd.to_datetime(trade_dates["calendar_date"]).sort_values().reset_index(drop=True)
    prices = load_all_prices()
    hfq_dict = {code: df["close_hfq"] for code, df in prices.items()}
    df_hfq_all = pd.DataFrame(hfq_dict).sort_index().ffill()

    nav_4q, stats_4q = run_freq_backtest(build_schedule([2, 5, 9, 11]), td_index, prices, df_hfq_all, "季频调仓(2/5/9/11月·Top10)")
    nav_3q, stats_3q = run_freq_backtest(build_schedule([5, 9, 11]), td_index, prices, df_hfq_all, "财报季调仓(5/9/11月·Top10)")
    nav_2q, _ = run_freq_backtest(build_schedule([5, 9]), td_index, prices, df_hfq_all, "半年调仓(5/9月·Top10)")

    df_guarded = pd.read_csv(RESULTS_DIR / "scoring_guarded_nav.csv", index_col=0, parse_dates=True)
    df_res = pd.DataFrame({
        "季频调仓(2/5/9/11月·Top10)": nav_4q,
        "财报季调仓(5/9/11月·Top10)": nav_3q,
        "半年调仓(5/9月·Top10)": nav_2q,
        "中证红利": df_guarded["中证红利"].reindex(nav_4q.index).ffill(),
        "沪深300": df_guarded["沪深300"].reindex(nav_4q.index).ffill(),
    })

    df_res.to_csv(RESULTS_DIR / "quarterly_top10_nav.csv")
    summary = pd.DataFrame({col: calc_metrics(df_res[col]) for col in df_res.columns}).T
    summary.to_csv(RESULTS_DIR / "quarterly_top10_summary.csv")

    print("\n=== 季报调仓 vs 半年调仓（宽松底线+多因子打分+不限行业+Top10）===")
    print(summary.to_string())
    print("\n--- 2026年最新几次季度调仓表现 (季频 2/5/9/11月) ---")
    print(stats_4q.tail(5).to_string(index=False))

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {
        "季频调仓(2/5/9/11月·Top10)": "#d62728",
        "财报季调仓(5/9/11月·Top10)": "#2ca02c",
        "半年调仓(5/9月·Top10)": "#ff7f0e",
        "中证红利": "#1f77b4",
        "沪深300": "#7f7f7f",
    }
    for col in df_res.columns:
        ls = "--" if col in ("中证红利", "沪深300") else "-"
        lw = 2.2 if "Top10" in col else 1.4
        ax.plot(df_res.index, df_res[col], label=col, color=colors.get(col), linestyle=ls, linewidth=lw)
    ax.set_title("A股红利多因子 Top10：季报调仓 vs 半年调仓 近10年净值对比", fontsize=13)
    ax.set_ylabel("累计净值")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")
    fig.tight_layout()
    chart_path = RESULTS_DIR / "quarterly_top10_curve.png"
    fig.savefig(chart_path, dpi=150)
    print(f"\n📊 净值对比图已保存至: {chart_path}")


if __name__ == "__main__":
    main()
