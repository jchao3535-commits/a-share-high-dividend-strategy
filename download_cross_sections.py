import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import akshare as ak
import baostock as bs

from config import (
    DATA_DIR,
    FHPS_DIR,
    YJBB_DIR,
    STOCK_LIST_DIR,
    BENCHMARK_DIR,
    REPORT_PERIODS,
    BACKTEST_END_DATE,
)


def is_a_share(bs_code: str) -> bool:
    """判断是否为沪深 A 股（含主板、创业板、科创板）"""
    if not isinstance(bs_code, str) or "." not in bs_code:
        return False
    mkt, num = bs_code.split(".", 1)
    if mkt == "sh":
        return num.startswith(("600", "601", "603", "605", "688", "689"))
    if mkt == "sz":
        return num.startswith(("000", "001", "002", "003", "300", "301"))
    return False


def get_rebalance_schedule(trade_dates: pd.Series) -> pd.DataFrame:
    """
    根据交易日历，为每个报告期匹配无未来函数的调仓日：
    - YYYY1231（年报）：次年 5 月第 1 个交易日
    - YYYY0630（中报）：当年 9 月第 1 个交易日
    """
    td = pd.to_datetime(trade_dates).sort_values().reset_index(drop=True)
    rows = []
    for period in REPORT_PERIODS:
        year = int(period[:4])
        mmdd = period[4:]
        if mmdd == "1231":
            target_start = pd.Timestamp(f"{year + 1}-05-01")
        else:
            target_start = pd.Timestamp(f"{year}-09-01")
        valid = td[td >= target_start]
        if valid.empty:
            continue
        reb_date = valid.iloc[0].strftime("%Y-%m-%d")
        rows.append({"period": period, "rebalance_date": reb_date})
    return pd.DataFrame(rows)


def download_baostock_meta():
    """下载交易日历、各调仓日历史全市场股票名单、以及基准指数日线"""
    print("[BaoStock] 登录中...")
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"BaoStock 登录失败: {lg.error_msg}")

    try:
        # 1. 交易日历
        td_path = DATA_DIR / "trade_dates.parquet"
        rs_td = bs.query_trade_dates(start_date="2016-01-01", end_date=BACKTEST_END_DATE)
        td_rows = []
        while rs_td.error_code == "0" and rs_td.next():
            td_rows.append(rs_td.get_row_data())
        df_td = pd.DataFrame(td_rows, columns=rs_td.fields)
        df_td = df_td[df_td["is_trading_day"] == "1"].reset_index(drop=True)
        df_td.to_parquet(td_path, index=False)
        print(f"[BaoStock] 交易日历已保存: {len(df_td)} 天 -> {td_path}")

        # 2. 调仓日映射表
        df_sched = get_rebalance_schedule(df_td["calendar_date"])
        sched_path = DATA_DIR / "rebalance_schedule.parquet"
        df_sched.to_parquet(sched_path, index=False)
        print(f"[BaoStock] 调仓日历已生成: {len(df_sched)} 期 -> {sched_path}")

        # 3. 各调仓日历史全市场股票名单
        for _, row in df_sched.iterrows():
            reb_date = row["rebalance_date"]
            out_path = STOCK_LIST_DIR / f"{reb_date}.parquet"
            if out_path.exists():
                continue
            rs_st = bs.query_all_stock(day=reb_date)
            st_rows = []
            while rs_st.error_code == "0" and rs_st.next():
                st_rows.append(rs_st.get_row_data())
            df_st = pd.DataFrame(st_rows, columns=rs_st.fields)
            df_st = df_st[df_st["code"].apply(is_a_share)].reset_index(drop=True)
            df_st["symbol"] = df_st["code"].str.split(".").str[1]
            df_st.to_parquet(out_path, index=False)
            print(f"[BaoStock] {reb_date} 历史A股名单: {len(df_st)} 只")

        # 4. 基准指数日线：中证红利(sh.000922)、沪深300(sh.000300)
        for idx_code in ["sh.000922", "sh.000300"]:
            idx_path = BENCHMARK_DIR / f"{idx_code}.parquet"
            rs_idx = bs.query_history_k_data_plus(
                idx_code,
                "date,code,open,high,low,close,volume,amount",
                start_date="2016-04-01",
                end_date=BACKTEST_END_DATE,
                frequency="d",
            )
            idx_rows = []
            while rs_idx.error_code == "0" and rs_idx.next():
                idx_rows.append(rs_idx.get_row_data())
            df_idx = pd.DataFrame(idx_rows, columns=rs_idx.fields)
            for col in ["open", "high", "low", "close", "volume", "amount"]:
                df_idx[col] = pd.to_numeric(df_idx[col], errors="coerce")
            df_idx.to_parquet(idx_path, index=False)
            print(f"[BaoStock] 基准指数 {idx_code} 已保存: {len(df_idx)} 条")
    finally:
        bs.logout()


def fetch_one_fhps(period: str) -> tuple[str, int]:
    out_path = FHPS_DIR / f"{period}.parquet"
    if out_path.exists():
        return period, -1
    for attempt in range(3):
        try:
            df = ak.stock_fhps_em(date=period)
            if df is not None and not df.empty:
                df["报告期"] = period
                # 统一转字符串避免混合类型写 parquet 报错
                for col in ["代码", "名称", "预案公告日", "股权登记日", "除权除息日", "方案进度", "最新公告日期"]:
                    if col in df.columns:
                        df[col] = df[col].astype(str)
                df.to_parquet(out_path, index=False)
                return period, len(df)
        except Exception as e:
            print(f"[FHPS] {period} 第 {attempt + 1} 次失败: {e}")
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"下载 fhps {period} 失败")


def fetch_one_yjbb(period: str) -> tuple[str, int]:
    out_path = YJBB_DIR / f"{period}.parquet"
    if out_path.exists():
        return period, -1
    for attempt in range(3):
        try:
            df = ak.stock_yjbb_em(date=period)
            if df is not None and not df.empty:
                df["报告期"] = period
                for col in ["股票代码", "股票简称", "所处行业", "最新公告日期"]:
                    if col in df.columns:
                        df[col] = df[col].astype(str)
                df.to_parquet(out_path, index=False)
                return period, len(df)
        except Exception as e:
            print(f"[YJBB] {period} 第 {attempt + 1} 次失败: {e}")
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"下载 yjbb {period} 失败")


def download_akshare_cross_sections():
    print(f"\n[AkShare] 开始拉取 {len(REPORT_PERIODS)} 个报告期的分红送配(fhps)与业绩报表(yjbb)...")
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {}
        for p in REPORT_PERIODS:
            futures[pool.submit(fetch_one_fhps, p)] = ("fhps", p)
            futures[pool.submit(fetch_one_yjbb, p)] = ("yjbb", p)
        for fut in as_completed(futures):
            kind, period = futures[fut]
            _, cnt = fut.result()
            if cnt == -1:
                print(f"[{kind.upper()}] {period} 已存在缓存，跳过")
            else:
                print(f"[{kind.upper()}] {period} 下载完成: {cnt} 条")


if __name__ == "__main__":
    download_baostock_meta()
    download_akshare_cross_sections()
    print("\n✅ 所有低频截面表与历史股票名单下载完成！")
