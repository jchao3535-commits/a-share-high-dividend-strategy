import numpy as np
import pandas as pd

from config import RESULTS_DIR, BACKTEST_END_DATE
from backtest import load_all_prices
from backtest_quarterly import get_quarterly_candidates


def generate_portfolio(
    title: str,
    item: dict,
    prices: dict[str, pd.DataFrame],
    max_price: float | None = 20.0,
    total_capital: float = 20000.0,
    top_n: int = 10,
    end_dt: str = BACKTEST_END_DATE,
) -> pd.DataFrame:
    """生成指定调仓期的多因子打分 Top N 实盘持仓清单（可选小资金股价上限过滤与手数计算）"""
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
        0.40 * df["s_div"]
        + 0.25 * df["s_pe"]
        + 0.20 * df["s_mc"]
        + 0.15 * df["s_roe"]
    )
    df = df.sort_values("total_score", ascending=False)

    per_stock_budget = total_capital / top_n
    rows = []
    for _, r in df.iterrows():
        c = r["code"]
        p_now = float(prices[c].loc[end_dt, "close_raw"])
        if max_price is not None and p_now > max_price:
            continue
        sub = prices[c].loc[item["rebalance_date"]:end_dt, "close_hfq"]
        ret = float(sub.iloc[-1] / sub.iloc[0] - 1.0)
        lots = max(1, int(per_stock_budget // (p_now * 100)))
        rows.append({
            "code": c,
            "name": r["name"],
            "price_now": round(p_now, 2),
            "lots": lots,
            "cost": round(lots * p_now * 100, 2),
            "div_yield": round(float(r["div_yield"]), 4),
            "peTTM": round(float(r["peTTM"]), 2),
            "mkt_cap_yi": round(float(r["mkt_cap_yi"]), 1),
            "roe_ann": round(float(r["roe_ann"]), 2),
            "total_score": round(float(r["total_score"]), 4),
            "ret_since_rebalance": round(ret, 4),
        })
        if len(rows) >= top_n:
            break

    out = pd.DataFrame(rows)
    print(f"\n=== {title} (股价上限={max_price}元, 总资金={total_capital:.0f}元) ===")
    print(out.to_string(index=False))
    print(f"实际占用: {out['cost'].sum():.0f} 元 | 剩余现金: {total_capital - out['cost'].sum():.0f} 元")
    return out


def main():
    prices = load_all_prices()

    df_semi = generate_portfolio(
        "半年调仓(2026-09-01 半年报)",
        {
            "rebalance_date": "2026-09-01",
            "yj_period": "20260630",
            "roe_scale": 2.0,
            "fh_periods": ["20251231", "20260630"],
        },
        prices,
        max_price=20.0,
        total_capital=20000.0,
        top_n=10,
    )
    df_semi.to_csv(RESULTS_DIR / "live_holdings_semi_under20.csv", index=False)

    df_q3 = generate_portfolio(
        "季频调仓(2026-11-02 三季报)",
        {
            "rebalance_date": "2026-11-02",
            "yj_period": "20260930",
            "roe_scale": 4.0 / 3.0,
            "fh_periods": ["20251231", "20260630", "20260930"],
        },
        prices,
        max_price=20.0,
        total_capital=20000.0,
        top_n=10,
    )
    df_q3.to_csv(RESULTS_DIR / "live_holdings_q3_under20.csv", index=False)


if __name__ == "__main__":
    main()
