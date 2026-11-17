import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    DATA_DIR,
    YJBB_DIR,
    RESULTS_DIR,
    BACKTEST_END_DATE,
    BUY_COST_RATE,
    SELL_COST_RATE,
)
from backtest import load_all_prices, calc_metrics
from backtest_mktcap_groups import get_candidates_on_date

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def run_industry_capped_backtest(
    sched: pd.DataFrame,
    td_index: pd.DatetimeIndex,
    prices: dict[str, pd.DataFrame],
    df_hfq_all: pd.DataFrame,
    name: str,
    top_n: int = 10,
    max_per_industry: int = 3,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """宽松底线(PE<25, ROE 6-18%) + 多因子打分(40/25/20/15) + 同行业最多选 max_per_industry 只"""
    w = {"div": 0.40, "pe": 0.25, "mc": 0.20, "roe": 0.15}
    nav_series_list = []
    period_stats = []
    latest_picks = pd.DataFrame()
    current_nav = 1.0
    prev_holdings: set[str] = set()
    end_ts = pd.Timestamp(BACKTEST_END_DATE)

    for idx, row in sched.iterrows():
        period = row["period"]
        reb_date = row["rebalance_date"]
        reb_ts = pd.Timestamp(reb_date)
        next_reb_ts = pd.Timestamp(sched.iloc[idx + 1]["rebalance_date"]) if idx + 1 < len(sched) else end_ts

        df = get_candidates_on_date(reb_date, period, prices).copy()
        df["roe_ann"] = df["roe"] * (2.0 if period.endswith("0630") else 1.0)
        df = df[
            (df["peTTM"] > 0)
            & (df["peTTM"] < 25.0)
            & (df["roe_ann"].notna())
            & (df["roe_ann"] >= 6.0)
            & (df["roe_ann"] <= 18.0)
        ].copy()

        # 关联行业
        df_yj = pd.read_parquet(YJBB_DIR / f"{period}.parquet")
        df_yj["symbol"] = df_yj["股票代码"].astype(str).str.zfill(6)
        df_yj = df_yj[["symbol", "所处行业"]].drop_duplicates("symbol", keep="last")
        df["symbol"] = df["code"].str.split(".").str[1]
        df = df.merge(df_yj, on="symbol", how="left")
        df["industry"] = df["所处行业"].fillna("未知").replace({"None": "未知", "nan": "未知", "": "未知"})

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

        # 按综合得分降序遍历，同行业不超过 max_per_industry 只
        df_sorted = df.sort_values("total_score", ascending=False)
        picked_rows = []
        ind_counts: dict[str, int] = {}
        for _, r in df_sorted.iterrows():
            ind = r["industry"]
            cnt = ind_counts.get(ind, 0)
            if ind != "未知" and cnt >= max_per_industry:
                continue
            picked_rows.append(r)
            ind_counts[ind] = cnt + 1
            if len(picked_rows) >= top_n:
                break

        df_pick = pd.DataFrame(picked_rows).reset_index(drop=True)
        if idx == len(sched) - 1:
            latest_picks = df_pick.copy()

        sub_dates = td_index[(td_index >= reb_ts) & (td_index <= next_reb_ts)]
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
            "持仓数": len(new_holdings),
            "覆盖行业数": df_pick["industry"].nunique(),
            "平均股息率": f"{df_pick['div_yield'].mean():.2%}",
            "区间收益": f"{(port_rel.iloc[-1] - 1.0):.2%}",
        })
        current_nav = period_nav.iloc[-1]
        prev_holdings = new_holdings

    nav = pd.concat(nav_series_list)
    nav.name = name
    return nav, pd.DataFrame(period_stats), latest_picks


def main():
    sched = pd.read_parquet(DATA_DIR / "rebalance_schedule.parquet")
    trade_dates = pd.read_parquet(DATA_DIR / "trade_dates.parquet")
    td_index = pd.to_datetime(trade_dates["calendar_date"]).sort_values().reset_index(drop=True)
    prices = load_all_prices()
    hfq_dict = {code: df["close_hfq"] for code, df in prices.items()}
    df_hfq_all = pd.DataFrame(hfq_dict).sort_index().ffill()

    nav_cap, stats_cap, latest_cap = run_industry_capped_backtest(
        sched, td_index, prices, df_hfq_all, "Top10(同行业<=3只)", top_n=10, max_per_industry=3
    )
    nav_nocap, _, _ = run_industry_capped_backtest(
        sched, td_index, prices, df_hfq_all, "Top10(不限行业)", top_n=10, max_per_industry=999
    )

    df_guarded = pd.read_csv(RESULTS_DIR / "scoring_guarded_nav.csv", index_col=0, parse_dates=True)
    df_res = pd.DataFrame({
        "Top10(同行业<=3只)": nav_cap,
        "Top10(不限行业)": nav_nocap,
        "Top30(不限行业)": df_guarded["宽松底线(PE<25·ROE 6-18%)+打分"].reindex(nav_cap.index).ffill(),
        "中证红利": df_guarded["中证红利"].reindex(nav_cap.index).ffill(),
        "沪深300": df_guarded["沪深300"].reindex(nav_cap.index).ffill(),
    })

    df_res.to_csv(RESULTS_DIR / "industry_cap_nav.csv")
    summary = pd.DataFrame({col: calc_metrics(df_res[col]) for col in df_res.columns}).T
    summary.to_csv(RESULTS_DIR / "industry_cap_summary.csv")

    print("\n=== 行业集中度限制（同行业<=3只）Top10 近10年对比 ===")
    print(summary.to_string())

    # 计算最新一期 2026-09-01 ~ 2026-11-16 表现
    rets = []
    for code in latest_cap["code"]:
        sub = prices[code].loc["2026-09-01":"2026-11-16", "close_hfq"]
        rets.append(sub.iloc[-1] / sub.iloc[0] - 1.0)
    latest_cap["ret_to_20261116"] = rets
    out_cols = ["code", "name", "industry", "div_yield", "peTTM", "mkt_cap_yi", "roe_ann", "total_score", "ret_to_20261116"]
    latest_cap[out_cols].to_csv(RESULTS_DIR / "latest_holdings_20260901_indcap.csv", index=False)

    print("\n=== 2026-09-01 最新一期 Top10（同行业<=3只）持仓明细 ===")
    disp = latest_cap[out_cols].copy()
    disp["div_yield"] = (disp["div_yield"] * 100).round(2).astype(str) + "%"
    disp["peTTM"] = disp["peTTM"].round(2)
    disp["mkt_cap_yi"] = disp["mkt_cap_yi"].round(1)
    disp["roe_ann"] = disp["roe_ann"].round(2).astype(str) + "%"
    disp["total_score"] = disp["total_score"].round(4)
    disp["ret_to_20261116"] = (disp["ret_to_20261116"] * 100).round(2).astype(str) + "%"
    print(disp.to_string(index=False))

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {
        "Top10(同行业<=3只)": "#d62728",
        "Top10(不限行业)": "#ff7f0e",
        "Top30(不限行业)": "#2ca02c",
        "中证红利": "#1f77b4",
        "沪深300": "#7f7f7f",
    }
    for col in df_res.columns:
        ls = "--" if col in ("中证红利", "沪深300") else "-"
        lw = 2.2 if "Top10" in col else 1.4
        ax.plot(df_res.index, df_res[col], label=col, color=colors.get(col), linestyle=ls, linewidth=lw)
    ax.set_title("A股红利多因子 Top10：同行业<=3只 vs 不限行业 近10年净值对比", fontsize=13)
    ax.set_ylabel("累计净值")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")
    fig.tight_layout()
    chart_path = RESULTS_DIR / "industry_cap_curve.png"
    fig.savefig(chart_path, dpi=150)
    print(f"\n📊 净值对比图已保存至: {chart_path}")


if __name__ == "__main__":
    main()
