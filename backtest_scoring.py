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


def run_scoring_backtest(
    sched: pd.DataFrame,
    td_index: pd.DatetimeIndex,
    prices: dict[str, pd.DataFrame],
    df_hfq_all: pd.DataFrame,
    name: str,
    weights: dict[str, float],
    roe_mode: str = "mid",
    top_n: int = 30,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    不做硬性底线过滤，直接在全候选池中按截面分位数（0~1）计算多因子综合得分，取前 top_n 满额持仓：
    - div: 股息率越高分越高
    - pe:  PE_TTM 越低分越高
    - mc:  总市值越大分越高
    - roe: 'mid' 表示越接近稳健区间中心 11.5% 分越高；'high' 表示年化 ROE 越高分越高
    """
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

        df = get_candidates_on_date(reb_date, period, prices).copy()
        df["roe_ann"] = df["roe"] * (2.0 if period.endswith("0630") else 1.0)
        df = df[(df["peTTM"] > 0) & (df["roe_ann"].notna()) & (df["roe_ann"] > 0)].copy()

        # 截面分位数打分 [0, 1]
        df["s_div"] = df["div_yield"].rank(pct=True)
        df["s_pe"] = (-df["peTTM"]).rank(pct=True)
        df["s_mc"] = df["mkt_cap_yi"].rank(pct=True)
        if roe_mode == "mid":
            df["s_roe"] = (-np.abs(df["roe_ann"] - 11.5)).rank(pct=True)
        else:
            df["s_roe"] = df["roe_ann"].rank(pct=True)

        df["total_score"] = (
            weights["div"] * df["s_div"]
            + weights["pe"] * df["s_pe"]
            + weights["mc"] * df["s_mc"]
            + weights["roe"] * df["s_roe"]
        )

        df_pick = df.sort_values("total_score", ascending=False).head(top_n)
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
            "候选总数": len(df),
            "持仓数": len(new_holdings),
            "平均股息率": f"{df_pick['div_yield'].mean():.2%}",
            "PE中位": f"{df_pick['peTTM'].median():.1f}",
            "市值中位(亿)": f"{df_pick['mkt_cap_yi'].median():.0f}",
            "年化ROE中位": f"{df_pick['roe_ann'].median():.1f}%",
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

    schemes = [
        ("红利偏重打分(40/25/20/15·适中ROE)", {"div": 0.40, "pe": 0.25, "mc": 0.20, "roe": 0.15}, "mid"),
        ("等权打分(25/25/25/25·适中ROE)", {"div": 0.25, "pe": 0.25, "mc": 0.25, "roe": 0.25}, "mid"),
        ("等权打分(25/25/25/25·传统高ROE)", {"div": 0.25, "pe": 0.25, "mc": 0.25, "roe": 0.25}, "high"),
    ]

    nav_dict = {}
    for name, w, mode in schemes:
        nav, stats = run_scoring_backtest(sched, td_index, prices, df_hfq_all, name, w, roe_mode=mode, top_n=30)
        nav_dict[name] = nav
        if "红利偏重" in name:
            stats.to_csv(RESULTS_DIR / "period_stats_scoring_best.csv", index=False)
            print(f"\n--- {name} 各期持仓特征 ---")
            print(stats.to_string(index=False))

    df_res = pd.DataFrame(nav_dict)
    df_base = pd.read_csv(RESULTS_DIR / "nav_daily.csv", index_col=0, parse_dates=True)
    df_res["纯高股息Top30"] = df_base["策略净值"].reindex(df_res.index).ffill()
    df_res["中证红利"] = df_base["中证红利"].reindex(df_res.index).ffill()
    df_res["沪深300"] = df_base["沪深300"].reindex(df_res.index).ffill()

    df_res.to_csv(RESULTS_DIR / "scoring_nav.csv")
    summary = pd.DataFrame({col: calc_metrics(df_res[col]) for col in df_res.columns}).T
    summary.to_csv(RESULTS_DIR / "scoring_summary.csv")

    print("\n=== 多因子综合打分（无硬性过滤·每期满额Top30）近10年回测对比 ===")
    print(summary.to_string())

    fig, ax = plt.subplots(figsize=(12, 6.5))
    colors = {
        "红利偏重打分(40/25/20/15·适中ROE)": "#d62728",
        "等权打分(25/25/25/25·适中ROE)": "#2ca02c",
        "等权打分(25/25/25/25·传统高ROE)": "#ff7f0e",
        "纯高股息Top30": "#9467bd",
        "中证红利": "#1f77b4",
        "沪深300": "#7f7f7f",
    }
    for col in df_res.columns:
        ls = "--" if col in ("中证红利", "沪深300") else "-"
        lw = 2.2 if "打分" in col else 1.5
        ax.plot(df_res.index, df_res[col], label=col, color=colors.get(col), linestyle=ls, linewidth=lw)

    ax.set_title("A股高股息多因子综合打分（无底线过滤·满额Top30）近10年净值对比", fontsize=13)
    ax.set_ylabel("累计净值")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    chart_path = RESULTS_DIR / "scoring_curve.png"
    fig.savefig(chart_path, dpi=150)
    print(f"\n📊 多因子打分净值曲线已保存至: {chart_path}")


if __name__ == "__main__":
    main()
