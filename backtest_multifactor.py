import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    DATA_DIR,
    RESULTS_DIR,
    BACKTEST_END_DATE,
    BUY_COST_RATE,
    SELL_COST_RATE,
)
from backtest import load_all_prices, calc_metrics
from backtest_mktcap_groups import get_candidates_on_date

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def run_multifactor_backtest(
    sched: pd.DataFrame,
    td_index: pd.DatetimeIndex,
    prices: dict[str, pd.DataFrame],
    df_hfq_all: pd.DataFrame,
    name: str,
    mc_min: float,
    pe_min: float,
    pe_max: float,
    roe_min: float,
    roe_max: float,
    top_n: int = 30,
) -> tuple[pd.Series, pd.DataFrame]:
    """多因子组合筛选：市值 + PE_TTM + 年化ROE，组内按TTM股息率降序取Top N"""
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

        df_cand = get_candidates_on_date(reb_date, period, prices).copy()
        df_cand["roe_ann"] = df_cand["roe"] * (2.0 if period.endswith("0630") else 1.0)

        df_group = df_cand[
            (df_cand["mkt_cap_yi"] >= mc_min)
            & (df_cand["peTTM"] > pe_min)
            & (df_cand["peTTM"] < pe_max)
            & (df_cand["roe_ann"].notna())
            & (df_cand["roe_ann"] >= roe_min)
            & (df_cand["roe_ann"] < roe_max)
        ]
        df_pick = df_group.sort_values("div_yield", ascending=False).head(top_n)

        sub_dates = td_index[(td_index >= reb_ts) & (td_index <= next_reb_ts)]
        if df_pick.empty:
            period_nav = pd.Series(current_nav, index=sub_dates)
            nav_series_list.append(period_nav.iloc[1:] if nav_series_list else period_nav)
            period_stats.append({"报告期": period, "调仓日": reb_date, "候选数": 0, "持仓数": 0, "区间收益": "0.00%"})
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
            "候选数": len(df_group),
            "持仓数": len(new_holdings),
            "平均股息率": f"{df_pick['div_yield'].mean():.2%}",
            "区间收益": f"{(port_rel.iloc[-1] - 1.0):.2%}",
        })
        current_nav = period_nav.iloc[-1]
        prev_holdings = new_holdings

    nav = pd.concat(nav_series_list)
    nav.name = name
    return nav, pd.DataFrame(period_stats)


def main():
    sched = pd.read_parquet(DATA_DIR / "rebalance_schedule.parquet")
    trade_dates = pd.read_parquet(DATA_DIR / "trade_dates.parquet")
    td_index = pd.to_datetime(trade_dates["calendar_date"]).sort_values().reset_index(drop=True)
    prices = load_all_prices()

    hfq_dict = {code: df["close_hfq"] for code, df in prices.items()}
    df_hfq_all = pd.DataFrame(hfq_dict).sort_index().ffill()

    configs = [
        ("中大盘(>=100亿)+低PE(0-20)+中ROE(8-15%)", 100.0, 0.0, 20.0, 8.0, 15.0),
        ("大盘(>=300亿)+低PE(0-20)+中ROE(8-15%)", 300.0, 0.0, 20.0, 8.0, 15.0),
        ("超大盘(>=599亿)+低PE(0-20)+中ROE(8-15%)", 599.0, 0.0, 20.0, 8.0, 15.0),
    ]

    nav_dict = {}
    for name, mc_min, pe_min, pe_max, roe_min, roe_max in configs:
        nav, stats = run_multifactor_backtest(
            sched, td_index, prices, df_hfq_all, name,
            mc_min, pe_min, pe_max, roe_min, roe_max, top_n=30,
        )
        nav_dict[name] = nav
        print(f"\n--- {name} 各期持仓数: 最小={stats['持仓数'].min()}, 中位={stats['持仓数'].median():.0f}, 最大={stats['持仓数'].max()} ---")

    df_res = pd.DataFrame(nav_dict)
    df_base = pd.read_csv(RESULTS_DIR / "nav_daily.csv", index_col=0, parse_dates=True)
    df_res["全市场Top30"] = df_base["策略净值"].reindex(df_res.index).ffill()
    df_res["中证红利"] = df_base["中证红利"].reindex(df_res.index).ffill()
    df_res["沪深300"] = df_base["沪深300"].reindex(df_res.index).ffill()

    df_res.to_csv(RESULTS_DIR / "multifactor_nav.csv")
    summary = pd.DataFrame({col: calc_metrics(df_res[col]) for col in df_res.columns}).T
    summary.to_csv(RESULTS_DIR / "multifactor_summary.csv")

    print("\n=== 多因子组合高股息策略（市值 + 低PE + 中ROE + 股息率Top30）近10年回测对比 ===")
    print(summary.to_string())

    fig, ax = plt.subplots(figsize=(12, 6.5))
    colors = {
        "中大盘(>=100亿)+低PE(0-20)+中ROE(8-15%)": "#d62728",
        "大盘(>=300亿)+低PE(0-20)+中ROE(8-15%)": "#2ca02c",
        "超大盘(>=599亿)+低PE(0-20)+中ROE(8-15%)": "#ff7f0e",
        "全市场Top30": "#9467bd",
        "中证红利": "#1f77b4",
        "沪深300": "#7f7f7f",
    }
    for col in df_res.columns:
        ls = "--" if col in ("中证红利", "沪深300") else "-"
        lw = 2.2 if "+" in col else 1.5
        ax.plot(df_res.index, df_res[col], label=col, color=colors.get(col), linestyle=ls, linewidth=lw)

    ax.set_title("A股高股息多因子组合策略（市值 + 低PE + 中ROE）近10年净值对比", fontsize=13)
    ax.set_ylabel("累计净值")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    chart_path = RESULTS_DIR / "multifactor_curve.png"
    fig.savefig(chart_path, dpi=150)
    print(f"\n📊 多因子组合净值曲线已保存至: {chart_path}")


if __name__ == "__main__":
    main()
