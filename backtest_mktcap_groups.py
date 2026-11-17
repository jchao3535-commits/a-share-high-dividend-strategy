import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    DATA_DIR,
    YJBB_DIR,
    STOCK_LIST_DIR,
    BENCHMARK_DIR,
    RESULTS_DIR,
    BACKTEST_END_DATE,
    BUY_COST_RATE,
    SELL_COST_RATE,
)
from backtest import (
    load_all_prices,
    _load_clean_fhps,
    _get_ttm_periods,
    calc_metrics,
)

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def get_candidates_on_date(reb_date: str, period: str, prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """获取指定调仓日所有符合基本面与交易条件的高股息候选股（含总市值）"""
    reb_ts = pd.Timestamp(reb_date)
    df_st = pd.read_parquet(STOCK_LIST_DIR / f"{reb_date}.parquet")
    df_yj = pd.read_parquet(YJBB_DIR / f"{period}.parquet")

    df_st = df_st[df_st["tradeStatus"] == "1"].copy()
    df_st = df_st[~df_st["code_name"].str.contains("ST|退", na=False)]

    p_annual, p_semi = _get_ttm_periods(period)
    df_ann = _load_clean_fhps(p_annual, reb_ts).rename(
        columns={"div_per_10": "div_ann", "total_shares": "shares_ann"}
    )
    df_semi = _load_clean_fhps(p_semi, reb_ts).rename(
        columns={"div_per_10": "div_semi", "total_shares": "shares_semi"}
    )

    df_div = df_ann.merge(df_semi, on="symbol", how="outer")
    df_div["div_per_10_ttm"] = df_div["div_ann"].fillna(0.0) + df_div["div_semi"].fillna(0.0)
    df_div["total_shares"] = df_div["shares_semi"].fillna(df_div["shares_ann"])
    df_div = df_div[df_div["div_per_10_ttm"] > 0]

    df_yj["symbol"] = df_yj["股票代码"].astype(str).str.zfill(6)
    df_yj["eps"] = pd.to_numeric(df_yj["每股收益"], errors="coerce")
    df_yj["net_profit"] = pd.to_numeric(df_yj["净利润-净利润"], errors="coerce")
    df_yj["roe"] = pd.to_numeric(df_yj["净资产收益率"], errors="coerce")
    df_yj = df_yj.drop_duplicates("symbol", keep="last")

    merged = df_st.merge(df_div[["symbol", "div_per_10_ttm", "total_shares"]], on="symbol", how="inner")
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

        mkt_cap_yi = close_raw * r["total_shares"] / 1e8
        records.append({
            "code": code,
            "name": r["code_name"],
            "div_yield": div_yield,
            "roe": r["roe"],
            "peTTM": day_row["peTTM"],
            "mkt_cap_yi": mkt_cap_yi,
        })

    return pd.DataFrame(records)


def run_group_backtest(
    sched: pd.DataFrame,
    td_index: pd.DatetimeIndex,
    prices: dict[str, pd.DataFrame],
    df_hfq_all: pd.DataFrame,
    group_name: str,
    mc_min: float,
    mc_max: float,
    top_n: int = 30,
) -> tuple[pd.Series, pd.DataFrame]:
    """按指定市值区间过滤，在本组内按股息率排序取前 top_n 进行回测"""
    nav_series_list = []
    period_stats = []
    current_nav = 1.0
    prev_holdings: set[str] = set()
    end_ts = pd.Timestamp(BACKTEST_END_DATE)

    for idx, row in sched.iterrows():
        period = row["period"]
        reb_date = row["rebalance_date"]
        reb_ts = pd.Timestamp(reb_date)
        next_reb_ts = pd.Timestamp(sched.iloc[idx + 1]["rebalance_date"]) if idx + 1 < len(sched) else end_ts

        df_cand = get_candidates_on_date(reb_date, period, prices)
        df_group = df_cand[(df_cand["mkt_cap_yi"] >= mc_min) & (df_cand["mkt_cap_yi"] < mc_max)]
        df_pick = df_group.sort_values("div_yield", ascending=False).head(top_n)

        sub_dates = td_index[(td_index >= reb_ts) & (td_index <= next_reb_ts)]
        if df_pick.empty:
            period_nav = pd.Series(current_nav, index=sub_dates)
            nav_series_list.append(period_nav.iloc[1:] if nav_series_list else period_nav)
            period_stats.append({"报告期": period, "调仓日": reb_date, "组内候选": 0, "持仓数": 0, "区间收益": "0.00%"})
            continue

        new_holdings = set(df_pick["code"].tolist())
        if not prev_holdings:
            turnover_buy, turnover_sell = 1.0, 0.0
        else:
            kept = len(prev_holdings & new_holdings)
            turnover_buy = (len(new_holdings) - kept) / len(new_holdings)
            turnover_sell = (len(prev_holdings) - kept) / len(prev_holdings)

        current_nav *= (1.0 - turnover_buy * BUY_COST_RATE - turnover_sell * SELL_COST_RATE)

        sub_hfq = df_hfq_all.reindex(sub_dates)[list(new_holdings)]
        rel_prices = sub_hfq.div(sub_hfq.iloc[0], axis=1)
        port_rel = rel_prices.mean(axis=1)
        period_nav = current_nav * port_rel

        nav_series_list.append(period_nav.iloc[1:] if nav_series_list else period_nav)
        period_stats.append({
            "报告期": period,
            "调仓日": reb_date,
            "组内候选": len(df_group),
            "持仓数": len(new_holdings),
            "平均股息率": f"{df_pick['div_yield'].mean():.2%}",
            "平均ROE": f"{df_pick['roe'].mean():.2f}%",
            "区间收益": f"{(port_rel.iloc[-1] - 1.0):.2%}",
        })
        current_nav = period_nav.iloc[-1]
        prev_holdings = new_holdings

    nav = pd.concat(nav_series_list)
    nav.name = group_name
    return nav, pd.DataFrame(period_stats)


def main():
    sched = pd.read_parquet(DATA_DIR / "rebalance_schedule.parquet")
    trade_dates = pd.read_parquet(DATA_DIR / "trade_dates.parquet")
    td_index = pd.to_datetime(trade_dates["calendar_date"]).sort_values().reset_index(drop=True)
    prices = load_all_prices()

    hfq_dict = {code: df["close_hfq"] for code, df in prices.items()}
    df_hfq_all = pd.DataFrame(hfq_dict).sort_index().ffill()

    groups = [
        ("小盘(<100亿)", 0.0, 100.0),
        ("中盘(100-599亿)", 100.0, 599.0),
        ("大盘(>=599亿)", 599.0, np.inf),
    ]

    nav_dict = {}
    for gname, lo, hi in groups:
        nav, stats = run_group_backtest(sched, td_index, prices, df_hfq_all, gname, lo, hi, top_n=30)
        nav_dict[gname] = nav
        stats.to_csv(RESULTS_DIR / f"period_stats_{gname}.csv", index=False)
        print(f"\n--- {gname} 各期持仓数与区间收益摘要 ---")
        print(stats[["报告期", "调仓日", "组内候选", "持仓数", "平均股息率", "区间收益"]].to_string(index=False))

    df_res = pd.DataFrame(nav_dict)
    # 对比全市场基准策略（已有）与指数
    df_base = pd.read_csv(RESULTS_DIR / "nav_daily.csv", index_col=0, parse_dates=True)
    df_res["全市场Top30"] = df_base["策略净值"].reindex(df_res.index).ffill()
    df_res["中证红利"] = df_base["中证红利"].reindex(df_res.index).ffill()
    df_res["沪深300"] = df_base["沪深300"].reindex(df_res.index).ffill()

    df_res.to_csv(RESULTS_DIR / "mktcap_groups_nav.csv")
    summary = pd.DataFrame({col: calc_metrics(df_res[col]) for col in df_res.columns}).T
    summary.to_csv(RESULTS_DIR / "mktcap_groups_summary.csv")

    print("\n=== 市值分组高股息（各组Top30）近10年回测对比 ===")
    print(summary.to_string())

    fig, ax = plt.subplots(figsize=(12, 6.5))
    colors = {
        "小盘(<100亿)": "#ff7f0e",
        "中盘(100-599亿)": "#2ca02c",
        "大盘(>=599亿)": "#d62728",
        "全市场Top30": "#9467bd",
        "中证红利": "#1f77b4",
        "沪深300": "#7f7f7f",
    }
    for col in df_res.columns:
        ls = "--" if col in ("中证红利", "沪深300") else "-"
        lw = 2.2 if "盘" in col else 1.5
        ax.plot(df_res.index, df_res[col], label=col, color=colors.get(col), linestyle=ls, linewidth=lw)

    ax.set_title("A股高股息策略：按市值分三组（本组内股息率Top30）近10年净值对比", fontsize=13)
    ax.set_ylabel("累计净值")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")
    fig.tight_layout()
    chart_path = RESULTS_DIR / "mktcap_groups_curve.png"
    fig.savefig(chart_path, dpi=150)
    print(f"\n📊 分组对比净值曲线已保存至: {chart_path}")


if __name__ == "__main__":
    main()
