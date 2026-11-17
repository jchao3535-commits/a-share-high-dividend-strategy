import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    DATA_DIR,
    FHPS_DIR,
    YJBB_DIR,
    STOCK_LIST_DIR,
    PRICE_DIR,
    BENCHMARK_DIR,
    RESULTS_DIR,
    BACKTEST_END_DATE,
    BUY_COST_RATE,
    SELL_COST_RATE,
)

# macOS 中文字体配置
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def load_all_prices() -> dict[str, pd.DataFrame]:
    """加载本地已缓存的候选池日线数据"""
    prices = {}
    for p in PRICE_DIR.glob("*.parquet"):
        code = p.stem
        df = pd.read_parquet(p)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").drop_duplicates("date").set_index("date")
        prices[code] = df
    print(f"已加载 {len(prices)} 只股票的日线数据")
    return prices


def _load_clean_fhps(period: str, reb_ts: pd.Timestamp) -> pd.DataFrame:
    """读取单期分红送配表并过滤公告日 <= 调仓日、有效分红"""
    path = FHPS_DIR / f"{period}.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["symbol", "div_per_10", "total_shares"])
    df = pd.read_parquet(path)
    df["symbol"] = df["代码"].astype(str).str.zfill(6)
    df["div_per_10"] = pd.to_numeric(df["现金分红-现金分红比例"], errors="coerce")
    df["total_shares"] = pd.to_numeric(df["总股本"], errors="coerce")
    df["notice_date"] = pd.to_datetime(df["最新公告日期"], errors="coerce")
    df["plan_date"] = pd.to_datetime(df["预案公告日"], errors="coerce")
    df["eff_date"] = df["notice_date"].fillna(df["plan_date"])

    df = df[
        (df["div_per_10"] > 0)
        & (df["eff_date"].notna())
        & (df["eff_date"] <= reb_ts)
        & (~df["方案进度"].astype(str).str.contains("不分配|停止|未通过", na=False))
    ].copy()
    df = df.sort_values("eff_date").drop_duplicates("symbol", keep="last")
    return df[["symbol", "div_per_10", "total_shares"]]


def _get_ttm_periods(period: str) -> tuple[str, str]:
    """
    返回构建 TTM（近12个月）分红所需的两个报告期：
    - 若当前为 YYYY1231（5月调仓）：使用 YYYY1231 年报 + YYYY0630 中报
    - 若当前为 YYYY0630（9月调仓）：使用 (YYYY-1)1231 年报 + YYYY0630 中报
    """
    year = int(period[:4])
    if period.endswith("1231"):
        return f"{year}1231", f"{year}0630"
    return f"{year - 1}1231", f"{year}0630"


def select_portfolio_on_date(
    reb_date: str,
    period: str,
    prices: dict[str, pd.DataFrame],
    top_n: int = 30,
) -> pd.DataFrame:
    """在指定调仓日，严格无未来函数地筛选 TTM 高股息前 N 只股票"""
    reb_ts = pd.Timestamp(reb_date)
    df_st = pd.read_parquet(STOCK_LIST_DIR / f"{reb_date}.parquet")
    df_yj = pd.read_parquet(YJBB_DIR / f"{period}.parquet")

    # 1. 历史全市场名单过滤正常交易且非 ST/退市股
    df_st = df_st[df_st["tradeStatus"] == "1"].copy()
    df_st = df_st[~df_st["code_name"].str.contains("ST|退", na=False)]

    # 2. 合并近 12 个月（年报 + 中报）现金分红，计算 TTM 每10股派息
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

    # 3. 业绩报表截面表（过滤亏损陷阱，同时保留 ROE 供后续多因子研究）
    df_yj["symbol"] = df_yj["股票代码"].astype(str).str.zfill(6)
    df_yj["eps"] = pd.to_numeric(df_yj["每股收益"], errors="coerce")
    df_yj["net_profit"] = pd.to_numeric(df_yj["净利润-净利润"], errors="coerce")
    df_yj["roe"] = pd.to_numeric(df_yj["净资产收益率"], errors="coerce")
    df_yj = df_yj.drop_duplicates("symbol", keep="last")

    merged = df_st.merge(df_div[["symbol", "div_per_10_ttm", "total_shares"]], on="symbol", how="inner")
    merged = merged.merge(df_yj[["symbol", "eps", "net_profit", "roe"]], on="symbol", how="left")
    merged = merged[(merged["eps"] > 0) & (merged["net_profit"] > 0)].copy()

    # 4. 匹配调仓日当天的真实不复权收盘价、交易状态、PE_TTM
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
        if pd.isna(close_raw) or close_raw <= 0:
            continue

        div_yield = (r["div_per_10_ttm"] / 10.0) / close_raw
        # 过滤异常高股息陷阱（>20% 往往为一次性特别分红或数据异常）
        if div_yield <= 0 or div_yield > 0.20:
            continue

        mkt_cap = close_raw * r["total_shares"] / 1e8 if pd.notna(r["total_shares"]) else np.nan
        records.append({
            "period": period,
            "rebalance_date": reb_date,
            "code": code,
            "name": r["code_name"],
            "close_raw": close_raw,
            "div_per_10_ttm": r["div_per_10_ttm"],
            "div_yield": div_yield,
            "roe": r["roe"],
            "peTTM": day_row["peTTM"],
            "pbMRQ": day_row["pbMRQ"],
            "mkt_cap_yi": mkt_cap,
        })

    if not records:
        return pd.DataFrame()
    df_cand = pd.DataFrame(records)
    return df_cand.sort_values("div_yield", ascending=False).head(top_n).reset_index(drop=True)


def calc_metrics(nav: pd.Series, rf: float = 0.02) -> dict:
    """计算回测核心绩效指标"""
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1.0
    days = (nav.index[-1] - nav.index[0]).days
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (365.0 / days) - 1.0 if days > 0 else 0.0
    daily_ret = nav.pct_change().dropna()
    ann_vol = daily_ret.std() * np.sqrt(252)
    sharpe = (cagr - rf) / ann_vol if ann_vol > 0 else 0.0
    rolling_max = nav.cummax()
    drawdown = nav / rolling_max - 1.0
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0
    return {
        "累计收益率": f"{total_ret:.2%}",
        "年化收益率": f"{cagr:.2%}",
        "年化波动率": f"{ann_vol:.2%}",
        "夏普比率": f"{sharpe:.2f}",
        "最大回撤": f"{max_dd:.2%}",
        "卡尔玛比率": f"{calmar:.2f}",
    }


def run_backtest(top_n: int = 30):
    sched = pd.read_parquet(DATA_DIR / "rebalance_schedule.parquet")
    trade_dates = pd.read_parquet(DATA_DIR / "trade_dates.parquet")
    td_index = pd.to_datetime(trade_dates["calendar_date"]).sort_values().reset_index(drop=True)
    prices = load_all_prices()

    # 构建宽表：所有候选股的后复权收盘价（用于快速计算日度组合净值）
    hfq_dict = {code: df["close_hfq"] for code, df in prices.items()}
    df_hfq_all = pd.DataFrame(hfq_dict).sort_index().ffill()

    nav_series_list = []
    holdings_all = []
    current_nav = 1.0
    prev_holdings: set[str] = set()

    end_ts = pd.Timestamp(BACKTEST_END_DATE)
    period_stats = []

    for idx, row in sched.iterrows():
        period = row["period"]
        reb_date = row["rebalance_date"]
        reb_ts = pd.Timestamp(reb_date)

        if idx + 1 < len(sched):
            next_reb_ts = pd.Timestamp(sched.iloc[idx + 1]["rebalance_date"])
        else:
            next_reb_ts = end_ts

        # 选股
        df_pick = select_portfolio_on_date(reb_date, period, prices, top_n=top_n)
        if df_pick.empty:
            print(f"[警告] {reb_date} ({period}) 无满足条件股票，空仓")
            continue

        new_holdings = set(df_pick["code"].tolist())
        holdings_all.append(df_pick)

        # 计算换手率并扣除双边交易成本
        if not prev_holdings:
            turnover_buy = 1.0
            turnover_sell = 0.0
        else:
            kept = len(prev_holdings & new_holdings)
            turnover_buy = (len(new_holdings) - kept) / len(new_holdings)
            turnover_sell = (len(prev_holdings) - kept) / len(prev_holdings)

        cost = turnover_buy * BUY_COST_RATE + turnover_sell * SELL_COST_RATE
        current_nav *= (1.0 - cost)

        # 提取本持仓期内的交易日：[reb_ts, next_reb_ts]
        sub_dates = td_index[(td_index >= reb_ts) & (td_index <= next_reb_ts)]
        sub_hfq = df_hfq_all.reindex(sub_dates)[list(new_holdings)]

        # 以调仓日收盘价为基准归一化（等权组合）
        base_prices = sub_hfq.iloc[0]
        rel_prices = sub_hfq.div(base_prices, axis=1)
        port_rel = rel_prices.mean(axis=1)
        period_nav = current_nav * port_rel

        if nav_series_list:
            nav_series_list.append(period_nav.iloc[1:])
        else:
            nav_series_list.append(period_nav)

        period_ret = port_rel.iloc[-1] - 1.0
        period_stats.append({
            "报告期": period,
            "调仓日": reb_date,
            "结束日": sub_dates.iloc[-1].strftime("%Y-%m-%d"),
            "持仓数": len(new_holdings),
            "平均股息率": f"{df_pick['div_yield'].mean():.2%}",
            "平均ROE": f"{df_pick['roe'].mean():.2f}%",
            "区间收益": f"{period_ret:.2%}",
        })

        current_nav = period_nav.iloc[-1]
        prev_holdings = new_holdings

    strategy_nav = pd.concat(nav_series_list)
    strategy_nav.name = f"高股息Top{top_n}"

    # 加载基准指数并对齐日期
    df_res = pd.DataFrame({"策略净值": strategy_nav})
    for idx_code, label in [("sh.000922", "中证红利"), ("sh.000300", "沪深300")]:
        df_idx = pd.read_parquet(BENCHMARK_DIR / f"{idx_code}.parquet")
        df_idx["date"] = pd.to_datetime(df_idx["date"])
        df_idx = df_idx.set_index("date")["close"].reindex(df_res.index).ffill()
        df_res[label] = df_idx / df_idx.iloc[0]

    # 保存结果
    df_res.to_csv(RESULTS_DIR / "nav_daily.csv")
    pd.concat(holdings_all, ignore_index=True).to_csv(RESULTS_DIR / "holdings_history.csv", index=False)
    df_period = pd.DataFrame(period_stats)
    df_period.to_csv(RESULTS_DIR / "period_stats.csv", index=False)

    summary = pd.DataFrame({
        col: calc_metrics(df_res[col]) for col in df_res.columns
    }).T
    summary.to_csv(RESULTS_DIR / "performance_summary.csv")

    print("\n=== 各报告期调仓表现 ===")
    print(df_period.to_string(index=False))
    print("\n=== 近10年回测绩效汇总 (2016-05 至 2026-11-16) ===")
    print(summary.to_string())

    # 绘制净值曲线
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_res.index, df_res["策略净值"], label=f"高股息Top{top_n}（等权·TTM股息·后复权）", linewidth=2.2, color="#d62728")
    ax.plot(df_res.index, df_res["中证红利"], label="中证红利 (000922)", linewidth=1.6, color="#1f77b4", linestyle="--")
    ax.plot(df_res.index, df_res["沪深300"], label="沪深300 (000300)", linewidth=1.4, color="#7f7f7f", alpha=0.8)
    ax.set_title("A股纯高股息策略（半年调仓）近10年净值走势 (2016-05 ~ 2026-11-16)", fontsize=13)
    ax.set_ylabel("累计净值")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")
    fig.tight_layout()
    chart_path = RESULTS_DIR / "nav_curve.png"
    fig.savefig(chart_path, dpi=150)
    print(f"\n📊 净值曲线已保存至: {chart_path}")


if __name__ == "__main__":
    run_backtest(top_n=30)
