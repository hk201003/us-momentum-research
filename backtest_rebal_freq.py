"""
リバランス頻度 感度分析
━━━━━━━━━━━━━━━━━━━
年4回(SMT) / 隔月 / 月次 / 隔週 / 週次 の5パターンを比較。
改良版ベース（トレンドフィルター + 逆ボラ + セクター分散）
パラメータ: Short=6M, Mid=12M, Long=36M（SMTデフォルト）
"""
import gc
import io
import urllib.request
import yfinance as yf
import pandas as pd
import numpy as np
import sys

sys.stdout.reconfigure(encoding='utf-8')

N_PER_BUCKET = 7
TREND_SMA = 200
TREND_REDUCE = 0.5
VOL_WINDOW = 60
MAX_PER_SECTOR = 3
LB_SHORT = 126
LB_MID = 252
LB_LONG = 756

BT_START = '2011-03-01'
BT_END = '2026-03-07'
DATA_START = '2008-01-01'

print("=" * 70)
print("  Rebalance Frequency Sensitivity Analysis")
print("=" * 70, flush=True)

# ============================================================
# データ取得
# ============================================================
print("\n=== Fetching S&P500 ===", flush=True)
ticker_sector = {}
try:
    req = urllib.request.Request(
        'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
        headers={'User-Agent': 'Mozilla/5.0'}
    )
    with urllib.request.urlopen(req) as resp:
        html_content = resp.read().decode('utf-8')
    sp500_table = pd.read_html(io.StringIO(html_content))[0]
    sp500_tickers = sp500_table['Symbol'].str.replace('.', '-', regex=False).tolist()
    for col in sp500_table.columns:
        if 'GICS' in str(col) and 'Sector' in str(col):
            for _, row in sp500_table.iterrows():
                tk = str(row['Symbol']).replace('.', '-')
                ticker_sector[tk] = str(row[col])
            break
    print(f"  {len(sp500_tickers)} stocks", flush=True)
except Exception as e:
    print(f"  Failed: {e}")
    exit(1)

print("\n=== Downloading price data ===", flush=True)
all_dl_tickers = sorted(set(sp500_tickers + ['SPY']))
batch_size = 50
all_close = {}
all_open = {}

for i in range(0, len(all_dl_tickers), batch_size):
    batch = all_dl_tickers[i:i+batch_size]
    try:
        df = yf.download(' '.join(batch), start=DATA_START, end=BT_END,
                         progress=False, threads=True)
        if df.empty:
            continue
        close = df['Close']
        open_ = df['Open']
        if isinstance(close, pd.Series):
            t = batch[0]
            c = close.dropna()
            if len(c) >= 100:
                all_close[t] = c
                all_open[t] = open_.reindex(c.index).fillna(c)
        else:
            for col in close.columns:
                t = str(col)
                c = close[col].dropna()
                if len(c) >= 100:
                    all_close[t] = c
                    all_open[t] = open_[col].reindex(c.index).fillna(c)
        n_batches = (len(all_dl_tickers) + batch_size - 1) // batch_size
        print(f"  Batch {i//batch_size+1}/{n_batches}: {len(all_close)} stocks", flush=True)
    except Exception as e:
        print(f"  Batch error: {e}", flush=True)

print(f"Downloaded: {len(all_close)} stocks", flush=True)

spy_close_ser = all_close.pop('SPY', None)
spy_open_ser = all_open.pop('SPY', None)

# DataFrames
print("\n=== Building DataFrames ===", flush=True)
all_dates = sorted(set().union(*(s.index for s in all_close.values())))
date_index = pd.DatetimeIndex(all_dates)

close_df = pd.concat({t: all_close[t].reindex(date_index) for t in all_close}, axis=1)
open_df = pd.concat({t: all_open[t].reindex(date_index) for t in all_open}, axis=1)

spy_series = spy_close_ser.reindex(date_index).ffill() if spy_close_ser is not None else None
spy_sma200 = spy_series.rolling(TREND_SMA).mean() if spy_series is not None else None

ret_df = close_df.pct_change(fill_method=None)
gap_ret_df = (open_df - close_df.shift(1)) / close_df.shift(1)
intra_ret_df = (close_df - open_df) / open_df.replace(0, np.nan)
vol_df = ret_df.rolling(VOL_WINDOW).std()

del all_close, all_open, open_df
gc.collect()

mom_short = close_df.pct_change(LB_SHORT, fill_method=None)
mom_mid = close_df.pct_change(LB_MID, fill_method=None)
mom_long = close_df.pct_change(LB_LONG, fill_method=None)

print(f"Valid tickers: {len(close_df.columns)}", flush=True)

bt_dates = date_index[(date_index >= BT_START) & (date_index <= BT_END)]

# numpy
ret_np = ret_df.values
gap_ret_np = gap_ret_df.values
intra_ret_np = intra_ret_df.values
col_to_idx = {c: i for i, c in enumerate(ret_df.columns)}

# ============================================================
# リバランス日生成（5パターン）
# ============================================================
def get_rebal_exec(rebal_dates_list):
    rebal_exec = []
    for rd in rebal_dates_list:
        future = bt_dates[bt_dates > rd]
        if len(future) > 0:
            rebal_exec.append((rd, future[0]))
    return rebal_exec

# 1) 年4回（SMTオリジナル: 2/5/8/11月末）
quarterly_dates = []
for dt in bt_dates:
    if dt.month in {2, 5, 8, 11}:
        month_dates = bt_dates[(bt_dates.month == dt.month) & (bt_dates.year == dt.year)]
        if len(month_dates) > 0 and month_dates[-1] not in quarterly_dates:
            quarterly_dates.append(month_dates[-1])
quarterly_exec = get_rebal_exec(sorted(set(quarterly_dates)))

# 2) 隔月（偶数月末）
bimonthly_dates = []
for dt in bt_dates:
    if dt.month % 2 == 0:
        month_dates = bt_dates[(bt_dates.month == dt.month) & (bt_dates.year == dt.year)]
        if len(month_dates) > 0 and month_dates[-1] not in bimonthly_dates:
            bimonthly_dates.append(month_dates[-1])
bimonthly_exec = get_rebal_exec(sorted(set(bimonthly_dates)))

# 3) 月次（毎月末）
monthly_dates = []
for dt in bt_dates:
    month_dates = bt_dates[(bt_dates.month == dt.month) & (bt_dates.year == dt.year)]
    if len(month_dates) > 0 and month_dates[-1] not in monthly_dates:
        monthly_dates.append(month_dates[-1])
monthly_exec = get_rebal_exec(sorted(set(monthly_dates)))

# 4) 隔週（2週ごとの金曜日）
biweekly_dates = []
fridays = [d for d in bt_dates if d.weekday() == 4]  # 金曜日
for i in range(0, len(fridays), 2):
    biweekly_dates.append(fridays[i])
biweekly_exec = get_rebal_exec(biweekly_dates)

# 5) 週次（毎週金曜日）
weekly_dates = fridays
weekly_exec = get_rebal_exec(weekly_dates)

rebal_configs = [
    ("Quarterly (SMT)", quarterly_exec),
    ("Bimonthly",       bimonthly_exec),
    ("Monthly",         monthly_exec),
    ("Biweekly",        biweekly_exec),
    ("Weekly",          weekly_exec),
]

for name, exec_list in rebal_configs:
    print(f"  {name:<20} {len(exec_list):>4} rebalances", flush=True)

# ============================================================
# バックテスト
# ============================================================
def run_backtest(rebal_list):
    current_holdings = {}
    next_rebal_idx = 0
    portfolio_returns = []

    for i, date in enumerate(bt_dates):
        date_idx = date_index.get_loc(date)

        is_rebalance = False
        if next_rebal_idx < len(rebal_list):
            judge_date, exec_date = rebal_list[next_rebal_idx]
            if date == exec_date:
                is_rebalance = True
                next_rebal_idx += 1

        if is_rebalance:
            old_holdings = dict(current_holdings)
            jidx = date_index.get_loc(judge_date)

            ms = mom_short.iloc[jidx].dropna().sort_values(ascending=False)
            mm = mom_mid.iloc[jidx].dropna().sort_values(ascending=False)
            ml = mom_long.iloc[jidx].dropna().sort_values(ascending=False)

            selected = []
            sector_count = {}

            def can_add(tk):
                if tk in selected: return False
                sec = ticker_sector.get(tk, 'Unknown')
                return sector_count.get(sec, 0) < MAX_PER_SECTOR

            def add_tk(tk):
                selected.append(tk)
                sec = ticker_sector.get(tk, 'Unknown')
                sector_count[sec] = sector_count.get(sec, 0) + 1

            for bucket_df in [ms, mm, ml]:
                count = 0
                for tk in bucket_df.index:
                    if count >= N_PER_BUCKET: break
                    if can_add(tk):
                        add_tk(tk)
                        count += 1

            if not selected:
                current_holdings = {}
                portfolio_returns.append(0.0)
                continue

            # 逆ボラ加重
            vols = {}
            for t in selected:
                if t in vol_df.columns:
                    v = vol_df.iloc[jidx][t]
                    if not np.isnan(v) and v > 0:
                        vols[t] = v
            if vols:
                inv_vols = {t: 1.0 / v for t, v in vols.items()}
                median_inv = np.median(list(inv_vols.values()))
                for t in selected:
                    if t not in inv_vols:
                        inv_vols[t] = median_inv
                total = sum(inv_vols.values())
                weights = {t: inv_vols[t] / total for t in selected}
            else:
                w = 1.0 / len(selected)
                weights = {t: w for t in selected}

            # トレンドフィルター
            if spy_sma200 is not None:
                spy_val = spy_series.get(judge_date, np.nan)
                sma_val = spy_sma200.get(judge_date, np.nan)
                if not np.isnan(spy_val) and not np.isnan(sma_val) and spy_val < sma_val:
                    weights = {t: w * TREND_REDUCE for t, w in weights.items()}

            current_holdings = weights

            if old_holdings:
                gap_r = sum(gap_ret_np[date_idx, col_to_idx[t]] * w
                           for t, w in old_holdings.items()
                           if t in col_to_idx and not np.isnan(gap_ret_np[date_idx, col_to_idx[t]]))
                intra_r = sum(intra_ret_np[date_idx, col_to_idx[t]] * w
                             for t, w in current_holdings.items()
                             if t in col_to_idx and not np.isnan(intra_ret_np[date_idx, col_to_idx[t]]))
                r = (1 + gap_r) * (1 + intra_r) - 1
            else:
                r = sum(intra_ret_np[date_idx, col_to_idx[t]] * w
                       for t, w in current_holdings.items()
                       if t in col_to_idx and not np.isnan(intra_ret_np[date_idx, col_to_idx[t]]))

            portfolio_returns.append(r)
        else:
            if not current_holdings:
                portfolio_returns.append(0.0)
                continue
            r = sum(ret_np[date_idx, col_to_idx[t]] * w
                   for t, w in current_holdings.items()
                   if t in col_to_idx and not np.isnan(ret_np[date_idx, col_to_idx[t]]))
            portfolio_returns.append(r)

    return pd.Series(portfolio_returns, index=bt_dates[:len(portfolio_returns)])

# ============================================================
# 全パターン実行
# ============================================================
print(f"\n=== Running {len(rebal_configs)} rebalance frequencies ===", flush=True)

all_results = {}
for name, exec_list in rebal_configs:
    print(f"  Running {name}...", flush=True)
    rets = run_backtest(exec_list)

    cum = (1 + rets).cumprod()
    total = cum.iloc[-1] - 1
    years = len(rets) / 252
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 else 0
    vol = rets.std() * np.sqrt(252)
    sharpe = cagr / vol if vol > 0 else 0
    peak = cum.cummax()
    dd = (cum - peak) / peak
    mdd = dd.min()

    # ターンオーバー概算（リバランス回数 / 年）
    n_rebal = len(exec_list)
    rebal_per_year = n_rebal / years

    # 年次リターン
    annual = {}
    for year in range(2011, 2027):
        yr_mask = rets.index.year == year
        if yr_mask.sum() > 0:
            annual[year] = (1 + rets[yr_mask]).prod() - 1

    # 下落局面
    drawdowns = {}
    for pname, start, end in [("COVID", "2020-02-01", "2020-03-31"),
                               ("2022Bear", "2022-01-01", "2022-10-31"),
                               ("2018Q4", "2018-10-01", "2018-12-31")]:
        mask = (rets.index >= pd.Timestamp(start)) & (rets.index <= pd.Timestamp(end))
        pr = rets[mask]
        if len(pr) >= 2:
            cum_pr = (1 + pr).cumprod()
            drawdowns[pname] = {
                'return': cum_pr.iloc[-1] - 1,
                'mdd': ((cum_pr - cum_pr.cummax()) / cum_pr.cummax()).min()
            }

    all_results[name] = {
        'sharpe': sharpe, 'cagr': cagr, 'mdd': mdd, 'vol': vol,
        'rebal_per_year': rebal_per_year, 'n_rebal': n_rebal,
        'annual': annual, 'drawdowns': drawdowns, 'rets': rets
    }

# SPY ベンチマーク
spy_bt = spy_series.reindex(bt_dates).pct_change().fillna(0)
spy_cum = (1 + spy_bt).cumprod()
spy_total = spy_cum.iloc[-1] - 1
spy_years = len(spy_bt) / 252
spy_cagr = (1 + spy_total) ** (1 / spy_years) - 1
spy_vol = spy_bt.std() * np.sqrt(252)
spy_sharpe = spy_cagr / spy_vol if spy_vol > 0 else 0
spy_mdd = ((spy_cum - spy_cum.cummax()) / spy_cum.cummax()).min()

# ============================================================
# 結果出力
# ============================================================
print("\n" + "=" * 90)
print("  REBALANCE FREQUENCY COMPARISON")
print("=" * 90)
print(f"  {'Frequency':<22} {'Rebal/yr':>9} {'Sharpe':>8} {'CAGR':>8} {'MDD':>8} {'Vol':>7}")
print("-" * 90)

for name in all_results:
    r = all_results[name]
    marker = " << SMT" if "Quarterly" in name else ""
    print(f"  {name:<22} {r['rebal_per_year']:>8.1f} {r['sharpe']:>7.2f} "
          f"{r['cagr']*100:>+7.1f}% {r['mdd']*100:>+7.1f}% {r['vol']*100:>6.1f}%{marker}")

print(f"  {'SPY (BM)':<22} {'':>9} {spy_sharpe:>7.2f} "
      f"{spy_cagr*100:>+7.1f}% {spy_mdd*100:>+7.1f}% {spy_vol*100:>6.1f}%")
print("-" * 90)

# ============================================================
# 年次リターン比較
# ============================================================
print(f"\n  [Annual Returns]")
names = list(all_results.keys())
header = f"  {'Year':>6}"
for n in names:
    short_name = n.split('(')[0].strip()[:8]
    header += f"  {short_name:>10}"
header += f"  {'SPY':>10}"
print(header)
print(f"  {'-'*6}" + f"  {'-'*10}" * (len(names) + 1))

for year in range(2011, 2027):
    line = f"  {year:>6}"
    for n in names:
        yr_ret = all_results[n]['annual'].get(year, None)
        if yr_ret is not None:
            line += f"  {yr_ret*100:>+9.1f}%"
        else:
            line += f"  {'':>10}"
    # SPY
    spy_yr_mask = spy_bt.index.year == year
    if spy_yr_mask.sum() > 0:
        spy_yr_ret = (1 + spy_bt[spy_yr_mask]).prod() - 1
        line += f"  {spy_yr_ret*100:>+9.1f}%"
    print(line)

# ============================================================
# 下落局面比較
# ============================================================
print(f"\n  [Drawdown Periods - MaxDD]")
dd_names = ["COVID", "2022Bear", "2018Q4"]
dd_labels = ["COVID-19 (2020/2-3)", "2022 Bear (2022/1-10)", "2018 Q4 (2018/10-12)"]

for dd_name, dd_label in zip(dd_names, dd_labels):
    print(f"\n  {dd_label}:")
    for name in all_results:
        dd = all_results[name]['drawdowns'].get(dd_name)
        if dd:
            short_name = name.split('(')[0].strip()
            print(f"    {short_name:<18} Return: {dd['return']*100:>+7.1f}%  MDD: {dd['mdd']*100:>+7.1f}%")

# ============================================================
# 頻度 vs Sharpe のトレンド
# ============================================================
print(f"\n  [Frequency vs Sharpe Trend]")
print(f"  リバランス回数を増やすほどSharpeは上がるか？")
print(f"  {'Frequency':<22} {'回/年':>6} {'Sharpe':>8} {'CAGR':>8} {'Sharpe差(vs Quarterly)':>24}")
print(f"  {'-'*22} {'-'*6} {'-'*8} {'-'*8} {'-'*24}")

base_sharpe = all_results["Quarterly (SMT)"]['sharpe']
for name in all_results:
    r = all_results[name]
    diff = r['sharpe'] - base_sharpe
    sign = "+" if diff >= 0 else ""
    print(f"  {name:<22} {r['rebal_per_year']:>5.1f} {r['sharpe']:>7.2f} "
          f"{r['cagr']*100:>+7.1f}% {sign}{diff:>22.2f}")

print("\n" + "=" * 90)
print("  Done.", flush=True)
print("=" * 90)
