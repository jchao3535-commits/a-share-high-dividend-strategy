import pandas as pd
import baostock as bs

from config import (
    DATA_DIR,
    FHPS_DIR,
    YJBB_DIR,
    STOCK_LIST_DIR,
    PRICE_DIR,
    BACKTEST_START_DATE,
    BACKTEST_END_DATE,
)


def select_candidate_codes(top_n_per_period: int = 80) -> list[str]:
    """
    基于已下载的 22 期截面数据（fhps + yjbb + 历史A股名单），
    先筛选出每期股息率排名前 top_n_per_period 的候选股票，取并集后按需下载日线。
    """
    sched = pd.read_parquet(DATA_DIR / "rebalance_schedule.parquet")
    candidate_codes = set()

    for _, row in sched.iterrows():
        period = row["period"]
        reb_date = row["rebalance_date"]

        st_path = STOCK_LIST_DIR / f"{reb_date}.parquet"
        fh_path = FHPS_DIR / f"{period}.parquet"
        yj_path = YJBB_DIR / f"{period}.parquet"

        if not (st_path.exists() and fh_path.exists() and yj_path.exists()):
            continue

        df_st = pd.read_parquet(st_path)
        df_fh = pd.read_parquet(fh_path)
        df_yj = pd.read_parquet(yj_path)

        # 过滤正常交易、非 ST/退市股
        df_st = df_st[df_st["tradeStatus"] == "1"].copy()
        df_st = df_st[~df_st["code_name"].str.contains("ST|退", na=False)]

        # 清洗分红表
        df_fh["symbol"] = df_fh["代码"].astype(str).str.zfill(6)
        df_fh["div_per_10"] = pd.to_numeric(df_fh["现金分红-现金分红比例"], errors="coerce")
        df_fh["div_yield_snap"] = pd.to_numeric(df_fh["现金分红-股息率"], errors="coerce")
        df_fh["notice_date"] = pd.to_datetime(df_fh["最新公告日期"], errors="coerce")
        df_fh["plan_date"] = pd.to_datetime(df_fh["预案公告日"], errors="coerce")
        df_fh["eff_date"] = df_fh["notice_date"].fillna(df_fh["plan_date"])

        reb_ts = pd.Timestamp(reb_date)
        df_fh = df_fh[
            (df_fh["div_per_10"] > 0)
            & (df_fh["eff_date"].notna())
            & (df_fh["eff_date"] <= reb_ts)
            & (~df_fh["方案进度"].astype(str).str.contains("不分配|停止|未通过", na=False))
        ].copy()

        # 同一报告期如有多条记录，取最新一条
        df_fh = df_fh.sort_values("eff_date").drop_duplicates("symbol", keep="last")

        # 清洗业绩表（过滤亏损陷阱）
        df_yj["symbol"] = df_yj["股票代码"].astype(str).str.zfill(6)
        df_yj["eps"] = pd.to_numeric(df_yj["每股收益"], errors="coerce")
        df_yj["net_profit"] = pd.to_numeric(df_yj["净利润-净利润"], errors="coerce")
        df_yj = df_yj.drop_duplicates("symbol", keep="last")

        merged = df_st.merge(df_fh[["symbol", "div_per_10", "div_yield_snap"]], on="symbol", how="inner")
        merged = merged.merge(df_yj[["symbol", "eps", "net_profit"]], on="symbol", how="left")

        # 过滤异常高股息陷阱：要求净利润为正、EPS为正、快照股息率 <= 20%
        merged = merged[
            (merged["eps"] > 0)
            & (merged["net_profit"] > 0)
            & (merged["div_yield_snap"] > 0)
            & (merged["div_yield_snap"] <= 0.20)
        ]

        top_df = merged.sort_values("div_yield_snap", ascending=False).head(top_n_per_period)
        candidate_codes.update(top_df["code"].tolist())
        print(f"[{reb_date} | {period}] 满足初筛 {len(merged)} 只，取前 {len(top_df)} 只入候选池")

    codes = sorted(candidate_codes)
    print(f"\n📌 10年共 22 期去重后，需要下载日线的候选股票总计: {len(codes)} 只（远少于全市场 5300+ 只）")
    return codes


def download_stock_prices(codes: list[str]):
    """从 BaoStock 批量下载候选股票的后复权与不复权日线（含 peTTM、pbMRQ）"""
    print("[BaoStock] 登录中...")
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"BaoStock 登录失败: {lg.error_msg}")

    try:
        total = len(codes)
        downloaded = 0
        skipped = 0
        for i, code in enumerate(codes, 1):
            out_path = PRICE_DIR / f"{code}.parquet"
            if out_path.exists():
                skipped += 1
                continue

            # 1. 拉取不复权价格 + PE/PB（用于调仓日计算真实股息率）
            rs_raw = bs.query_history_k_data_plus(
                code,
                "date,code,close,volume,tradestatus,peTTM,pbMRQ",
                start_date=BACKTEST_START_DATE,
                end_date=BACKTEST_END_DATE,
                frequency="d",
                adjustflag="3",
            )
            raw_rows = []
            while rs_raw.error_code == "0" and rs_raw.next():
                raw_rows.append(rs_raw.get_row_data())
            if not raw_rows:
                continue
            df_raw = pd.DataFrame(raw_rows, columns=rs_raw.fields)
            df_raw.rename(columns={"close": "close_raw"}, inplace=True)

            # 2. 拉取后复权收盘价（用于计算分红再投资的真实持有收益）
            rs_hfq = bs.query_history_k_data_plus(
                code,
                "date,close",
                start_date=BACKTEST_START_DATE,
                end_date=BACKTEST_END_DATE,
                frequency="d",
                adjustflag="2",
            )
            hfq_rows = []
            while rs_hfq.error_code == "0" and rs_hfq.next():
                hfq_rows.append(rs_hfq.get_row_data())
            df_hfq = pd.DataFrame(hfq_rows, columns=rs_hfq.fields)
            df_hfq.rename(columns={"close": "close_hfq"}, inplace=True)

            df = df_raw.merge(df_hfq, on="date", how="inner")
            for col in ["close_raw", "close_hfq", "volume", "peTTM", "pbMRQ"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["tradestatus"] = pd.to_numeric(df["tradestatus"], errors="coerce")
            df.to_parquet(out_path, index=False)
            downloaded += 1

            if i % 50 == 0 or i == total:
                print(f"[日线下载进度] {i}/{total} (新下载 {downloaded}, 跳过缓存 {skipped})")
    finally:
        bs.logout()


if __name__ == "__main__":
    codes = select_candidate_codes(top_n_per_period=80)
    download_stock_prices(codes)
    print("\n✅ 候选池日线数据按需下载完成！")
