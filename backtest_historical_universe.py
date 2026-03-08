"""
S&P500 追加日ベース・バックテスト（生存バイアス除去版）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
各銘柄をS&P500に追加された日以降のみユニバースに含める。
「現在のS&P500で過去を計算」する生存バイアスを除去。

比較:
  A) Naive: 現在の構成銘柄で全期間計算（従来のバックテスト）
  B) Historical: S&P500追加日以降のみ参加（生存バイアス除去）
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
print("  Historical Universe Backtest (Survivorship Bias Removal)")
print("=" * 70, flush=True)

# ============================================================
# 1. S&P500 構成銘柄 + 追加日 取得
# ============================================================
print("\n=== Fetching S&P500 with Date Added ===", flush=True)

ticker_sector = {}
ticker_date_added = {}  # ticker → pd.Timestamp (S&P500追加日)

try:
    req = urllib.request.Request(
        'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
        headers={'User-Agent': 'Mozilla/5.0'}
    )
    with urllib.request.urlopen(req) as resp:
        html_content = resp.read().decode('utf-8')
    sp500_table = pd.read_html(io.StringIO(html_content))[0]
    sp500_tickers = sp500_table['Symbol'].str.replace('.', '-', regex=False).tolist()

    # セクター
    for col in sp500_table.columns:
        if 'GICS' in str(col) and 'Sector' in str(col):
            for _, row in sp500_table.iterrows():
                tk = str(row['Symbol']).replace('.', '-')
                ticker_sector[tk] = str(row[col])
            break

    # Date added
    date_col = None
    for col in sp500_table.columns:
        if 'date' in str(col).lower() and 'added' in str(col).lower():
            date_col = col
            break
    if date_col is None:
        for col in sp500_table.columns:
            if 'Date' in str(col):
                date_col = col
                break

    if date_col is not None:
        n_parsed = 0
        n_missing = 0
        for _, row in sp500_table.iterrows():
            tk = str(row['Symbol']).replace('.', '-')
            date_str = str(row[date_col]).strip()
            if date_str and date_str != 'nan' and date_str != 'NaT':
                try:
                    dt = pd.to_datetime(date_str)
                    ticker_date_added[tk] = dt
                    n_parsed += 1
                except:
                    n_missing += 1
            else:
                n_missing += 1
        print(f"  Date added: {n_parsed} parsed, {n_missing} missing", flush=True)
    else:
        print("  WARNING: 'Date added' column not found!", flush=True)

    print(f"  {len(sp500_tickers)} stocks, {len(set(ticker_sector.values()))} sectors", flush=True)

except Exception as e:
    print(f"  Failed: {e}")
    exit(1)

# 追加日の統計
if ticker_date_added:
    dates_added = pd.Series(ticker_date_added)
    print(f"\n  === Date Added Statistics ===")
    print(f"  Earliest: {dates_added.min().strftime('%Y-%m-%d')}")
    print(f"  Latest:   {dates_added.max().strftime('%Y-%m-%d')}")
    print(f"  Median:   {dates_added.median().strftime('%Y-%m-%d')}")

    # 年代別
    for year_start in [2000, 2010, 2015, 2020, 2024]:
        count = (dates_added >= pd.Timestamp(f'{year_start}-01-01')).sum()
        print(f"  {year_start}年以降に追加: {count}銘柄 ({count/len(dates_added)*100:.0f}%)")

    # BT_START以降に追加された銘柄（これが生存バイアスの原因）
    bt_start_ts = pd.Timestamp(BT_START)
    added_after_start = (dates_added > bt_start_ts).sum()
    print(f"\n  ** BT開始({BT_START})以降に追加: {added_after_start}銘柄 ({added_after_start/len(dates_added)*100:.0f}%) **")
    print(f"  → これらがNaive版では「最初からいた」前提で計算されている")

# ============================================================
# 2. 株価ダウンロード
# ============================================================
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

# ============================================================
# 3. ユニバースマスク作成
# ============================================================
print("\n=== Building Universe Masks ===", flush=True)

# Historical mask: 銘柄ごとにS&P500追加日以降のみTrue
historical_mask = pd.DataFrame(False, index=date_index, columns=close_df.columns)

for tk in close_df.columns:
    if tk in ticker_date_added:
        add_date = ticker_date_added[tk]
        historical_mask.loc[date_index >= add_date, tk] = True
    else:
        # 追加日不明 → 古い銘柄と見なして全期間True
        historical_mask[tk] = True

# Naive mask: 全銘柄全期間True
naive_mask = pd.DataFrame(True, index=date_index, columns=close_df.columns)

# 各時点のユニバースサイズ
hist_size = historical_mask.sum(axis=1)
print(f"  Historical universe size:")
for year in [2011, 2013, 2015, 2017, 2019, 2021, 2023, 2025]:
    yr_dates = hist_size[hist_size.index.year == year]
    if len(yr_dates) > 0:
        avg = yr_dates.mean()
        print(f"    {year}: avg {avg:.0f} stocks (vs Naive: {len(close_df.columns)})")

# ============================================================
# 4. バックテストエンジン（ユニバースマスク対応）
# ============================================================
bt_dates = date_index[(date_index >= BT_START) & (date_index <= BT_END)]

# 隔月リバランス
rebal_dates = []
for dt in bt_dates:
    if dt.month % 2 == 0:
        month_dates = bt_dates[(bt_dates.month == dt.month) & (bt_dates.year == dt.year)]
        if len(month_dates) > 0 and month_dates[-1] not in rebal_dates:
            rebal_dates.append(month_dates[-1])
rebal_dates = sorted(set(rebal_dates))
rebal_exec_dates = []
for rd in rebal_dates:
    future = bt_dates[bt_dates > rd]
    if len(future) > 0:
        rebal_exec_dates.append((rd, future[0]))

# 年4回リバランス
quarterly_dates = []
for dt in bt_dates:
    if dt.month in {2, 5, 8, 11}:
        month_dates = bt_dates[(bt_dates.month == dt.month) & (bt_dates.year == dt.year)]
        if len(month_dates) > 0 and month_dates[-1] not in quarterly_dates:
            quarterly_dates.append(month_dates[-1])
quarterly_dates = sorted(set(quarterly_dates))
quarterly_exec = []
for rd in quarterly_dates:
    future = bt_dates[bt_dates > rd]
    if len(future) > 0:
        quarterly_exec.append((rd, future[0]))

ret_np = ret_df.values
gap_ret_np = gap_ret_df.values
intra_ret_np = intra_ret_df.values
col_to_idx = {c: i for i, c in enumerate(ret_df.columns)}

def run_backtest(universe_mask, rebal_list, use_trend, use_invvol, use_sector_cap, label=""):
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

            # ユニバースマスク適用
            mask_row = universe_mask.iloc[jidx]
            eligible = set(mask_row[mask_row].index)

            ms = mom_short.iloc[jidx].dropna()
            mm = mom_mid.iloc[jidx].dropna()
            ml = mom_long.iloc[jidx].dropna()

            # eligible銘柄のみ
            ms = ms[ms.index.isin(eligible)].sort_values(ascending=False)
            mm = mm[mm.index.isin(eligible)].sort_values(ascending=False)
            ml = ml[ml.index.isin(eligible)].sort_values(ascending=False)

            selected = []
            sector_count = {}

            def can_add(tk):
                if tk in selected: return False
                if use_sector_cap:
                    sec = ticker_sector.get(tk, 'Unknown')
                    return sector_count.get(sec, 0) < MAX_PER_SECTOR
                return True

            def add_tk(tk):
                selected.append(tk)
                if use_sector_cap:
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

            # ウエイト計算
            if use_invvol:
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
                    total_w = sum(inv_vols.values())
                    weights = {t: inv_vols[t] / total_w for t in selected}
                else:
                    w = 1.0 / len(selected)
                    weights = {t: w for t in selected}
            else:
                w = 1.0 / len(selected)
                weights = {t: w for t in selected}

            # トレンドフィルター
            if use_trend and spy_sma200 is not None:
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

def calc_stats(rets):
    cum = (1 + rets).cumprod()
    total = cum.iloc[-1] - 1
    years = len(rets) / 252
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 else 0
    vol = rets.std() * np.sqrt(252)
    sharpe = cagr / vol if vol > 0 else 0
    peak = cum.cummax()
    dd = (cum - peak) / peak
    mdd = dd.min()
    return sharpe, cagr, mdd, vol

# ============================================================
# 5. 全パターン実行
# ============================================================
print(f"\n=== Running backtests ===", flush=True)

configs = [
    # (name, mask, rebal, trend, invvol, sector_cap)
    ("Naive SMT Original",     naive_mask,      quarterly_exec,    False, False, False),
    ("Historical SMT Original", historical_mask, quarterly_exec,    False, False, False),
    ("Naive Improved",         naive_mask,      rebal_exec_dates,  True,  True,  True),
    ("Historical Improved",    historical_mask, rebal_exec_dates,  True,  True,  True),
]

results = {}
for name, mask, rebal, trend, invvol, sector_cap in configs:
    print(f"  Running: {name}...", flush=True)
    rets = run_backtest(mask, rebal, trend, invvol, sector_cap, name)
    sharpe, cagr, mdd, vol = calc_stats(rets)
    results[name] = {'sharpe': sharpe, 'cagr': cagr, 'mdd': mdd, 'vol': vol, 'rets': rets}
    print(f"    Sharpe={sharpe:.2f}  CAGR={cagr*100:+.1f}%  MDD={mdd*100:.1f}%", flush=True)

# SPY
spy_bt = spy_series.reindex(bt_dates).pct_change().fillna(0)
spy_sharpe, spy_cagr, spy_mdd, spy_vol = calc_stats(spy_bt)
results['SPY'] = {'sharpe': spy_sharpe, 'cagr': spy_cagr, 'mdd': spy_mdd, 'vol': spy_vol, 'rets': spy_bt}

# ============================================================
# 6. 結果出力
# ============================================================
print("\n" + "=" * 85)
print("  NAIVE vs HISTORICAL UNIVERSE COMPARISON")
print("=" * 85)
print(f"  {'Strategy':<28} {'Sharpe':>7} {'CAGR':>8} {'MDD':>8} {'Vol':>7}")
print("-" * 85)
for name in ['Naive SMT Original', 'Historical SMT Original',
             'Naive Improved', 'Historical Improved', 'SPY']:
    r = results[name]
    print(f"  {name:<28} {r['sharpe']:>6.2f} {r['cagr']*100:>+7.1f}% "
          f"{r['mdd']*100:>+7.1f}% {r['vol']*100:>6.1f}%")
print("-" * 85)

# 生存バイアスの定量化
print(f"\n  === Survivorship Bias Quantification ===")

for naive_name, hist_name in [("Naive SMT Original", "Historical SMT Original"),
                               ("Naive Improved", "Historical Improved")]:
    n = results[naive_name]
    h = results[hist_name]
    ds = n['sharpe'] - h['sharpe']
    dc = (n['cagr'] - h['cagr']) * 100
    label = "SMT Original" if "Original" in naive_name else "Improved"
    print(f"\n  [{label}]")
    print(f"  Naive Sharpe:       {n['sharpe']:.2f}")
    print(f"  Historical Sharpe:  {h['sharpe']:.2f}")
    print(f"  Bias (Sharpe):      {ds:+.2f}")
    print(f"  Naive CAGR:         {n['cagr']*100:+.1f}%")
    print(f"  Historical CAGR:    {h['cagr']*100:+.1f}%")
    print(f"  Bias (CAGR):        {dc:+.1f}%")

# ============================================================
# 7. 年次リターン比較
# ============================================================
print(f"\n  [Annual Returns]")
print(f"  {'Year':>6}  {'NaiveOrig':>10}  {'HistOrig':>10}  {'NaiveImpr':>10}  {'HistImpr':>10}  {'SPY':>10}")
print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

for year in range(2011, 2027):
    line = f"  {year:>6}"
    for name in ['Naive SMT Original', 'Historical SMT Original',
                 'Naive Improved', 'Historical Improved', 'SPY']:
        rets = results[name]['rets']
        yr_mask = rets.index.year == year
        if yr_mask.sum() > 0:
            yr_ret = (1 + rets[yr_mask]).prod() - 1
            line += f"  {yr_ret*100:>+9.1f}%"
        else:
            line += f"  {'':>10}"
    print(line)

# ============================================================
# 8. 下落局面での差
# ============================================================
print(f"\n  [Crash Periods: Naive vs Historical]")
crashes = [
    ("COVID-19", "2020-02-01", "2020-03-31"),
    ("2022 Bear", "2022-01-01", "2022-10-31"),
    ("2018 Q4", "2018-10-01", "2018-12-31"),
]

for cname, start, end in crashes:
    print(f"\n  {cname}:")
    for name in ['Naive Improved', 'Historical Improved', 'SPY']:
        rets = results[name]['rets']
        mask = (rets.index >= pd.Timestamp(start)) & (rets.index <= pd.Timestamp(end))
        pr = rets[mask]
        if len(pr) < 2: continue
        cum_pr = (1 + pr).cumprod()
        pret = cum_pr.iloc[-1] - 1
        pmdd = ((cum_pr - cum_pr.cummax()) / cum_pr.cummax()).min()
        short = name.replace("Improved", "Impr")
        print(f"    {short:<28} Return: {pret*100:>+7.1f}%  MDD: {pmdd*100:>+7.1f}%")

# ============================================================
# 9. どの銘柄がバイアスの原因か
# ============================================================
print(f"\n  === Top Bias-Contributing Stocks ===")
print(f"  (BT開始後にS&P500入りした高モメンタム銘柄)")

bt_start_ts = pd.Timestamp(BT_START)
bias_stocks = []
for tk, add_date in ticker_date_added.items():
    if add_date > bt_start_ts and tk in close_df.columns:
        # 追加時点のモメンタムランク
        idx = date_index.get_indexer([add_date], method='ffill')[0]
        if idx >= LB_MID and idx < len(mom_mid):
            mom_val = mom_mid.iloc[idx].get(tk, np.nan)
            if not np.isnan(mom_val):
                # パーセンタイルランク
                all_moms = mom_mid.iloc[idx].dropna()
                pctile = (all_moms < mom_val).mean() * 100
                bias_stocks.append({
                    'ticker': tk,
                    'date_added': add_date.strftime('%Y-%m-%d'),
                    'mom_12m': mom_val,
                    'percentile': pctile,
                    'sector': ticker_sector.get(tk, 'Unknown')
                })

bias_df = pd.DataFrame(bias_stocks).sort_values('percentile', ascending=False)

print(f"\n  {'Ticker':<8} {'Added':>12} {'12M Mom':>10} {'Pctile':>8} {'Sector'}")
print(f"  {'-'*8} {'-'*12} {'-'*10} {'-'*8} {'-'*20}")
for _, row in bias_df.head(20).iterrows():
    print(f"  {row['ticker']:<8} {row['date_added']:>12} {row['mom_12m']*100:>+9.1f}% "
          f"{row['percentile']:>7.0f}% {row['sector']}")

print(f"\n  Top20%モメンタムで追加された銘柄: "
      f"{(bias_df['percentile'] >= 80).sum()}/{len(bias_df)} "
      f"({(bias_df['percentile'] >= 80).mean()*100:.0f}%)")

print("\n" + "=" * 85)
print("  Done.", flush=True)
print("=" * 85)
