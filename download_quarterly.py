import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import akshare as ak
import baostock as bs

from config import DATA_DIR, FHPS_DIR, YJBB_DIR, STOCK_LIST_DIR, BACKTEST_END_DATE
from download_cross_sections import is_a_share, fetch_one_fhps, fetch_one_yjbb


def get_quarterly_dates():
    """生成 2016~2026 年的 2月、5月、9月、11月首个交易日名单"""
    td = pd.to_datetime(pd.read_parquet(DATA_DIR / "trade_dates.parquet")["calendar_date"]).sort_values()
    dates = []
    for y in range(2016, 2027):
        for m in [2, 5, 9, 11]:
            if y == 2016 and m == 2:
                continue
            target = pd.Timestamp(f"{y}-{m:02d}-01")
            if target > pd.Timestamp(BACKTEST_END_DATE):
                continue
            valid = td[td >= target]
            if not valid.empty:
                dates.append(valid.iloc[0].strftime("%Y-%m-%d"))
    return sorted(set(dates))


def download_missing_stock_lists(dates: list[str]):
    need = [d for d in dates if not (STOCK_LIST_DIR / f"{d}.parquet").exists()]
    if not need:
        print("[BaoStock] 所有季度调仓日股票名单已缓存")
        return
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"BaoStock 登录失败: {lg.error_msg}")
    try:
        for d in need:
            rs = bs.query_all_stock(day=d)
            rows = []
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())
            df = pd.DataFrame(rows, columns=rs.fields)
            df = df[df["code"].apply(is_a_share)].reset_index(drop=True)
            df["symbol"] = df["code"].str.split(".").str[1]
            df.to_parquet(STOCK_LIST_DIR / f"{d}.parquet", index=False)
            print(f"[BaoStock] {d} 历史A股名单: {len(df)} 只")
    finally:
        bs.logout()


def download_q1_q3_reports():
    yj_periods = []
    fh_periods = []
    for y in range(2016, 2027):
        yj_periods.append(f"{y}0331")
        yj_periods.append(f"{y}0930")
        fh_periods.append(f"{y}0930")

    print(f"[AkShare] 开始拉取 Q1/Q3 业绩报表({len(yj_periods)}期) 与 Q3 分红({len(fh_periods)}期)...")
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {}
        for p in yj_periods:
            futures[pool.submit(fetch_one_yjbb, p)] = ("yjbb", p)
        for p in fh_periods:
            futures[pool.submit(fetch_one_fhps, p)] = ("fhps", p)
        for fut in as_completed(futures):
            kind, period = futures[fut]
            try:
                _, cnt = fut.result()
                if cnt != -1:
                    print(f"[{kind.upper()}] {period} 下载完成: {cnt} 条")
            except Exception as e:
                print(f"[{kind.upper()}] {period} 无数据或跳过: {e}")


if __name__ == "__main__":
    dates = get_quarterly_dates()
    download_missing_stock_lists(dates)
    download_q1_q3_reports()
    print("\n✅ 季报截面数据与历史股票名单补充完成！")
