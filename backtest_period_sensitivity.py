"""
SMT 米国モメンタム: ルックバック期間 感度分析
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
短期・中期・長期のモメンタム計算期間を網羅的に変えて
Sharpe, CAGR, MDD への影響を検証する。

ベース: 改良版（トレンドフィルター + 逆ボラ + 月次リバランス + セクター分散）
"""
import gc
import io
import urllib.request
import yfinance as yf
import pandas as pd
import numpy as np
import itertools
import json
from datetime import datetime

import sys
sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# 固定パラメータ（改良版）
# ============================================================
N_PER_BUCKET = 7
TREND_SMA = 200
TREND_REDUCE = 0.5
VOL_WINDOW = 60
VOL_EXPONENT = 1.0
MAX_PER_SECTOR = 3

BT_START = '2011-03-01'
BT_END = '2026-03-07'
DATA_START = '2008-01-01'

# ============================================================
# テストする期間の組み合わせ
# ============================================================
SHORT_CANDIDATES = [63, 126, 189]       # 3M, 6M, 9M
MID_CANDIDATES   = [126, 252, 378]      # 6M, 12M, 18M
LONG_CANDIDATES  = [504, 756, 1008]     # 24M, 36M, 48M

# SMTオリジナル
SMT_DEFAULT = (126, 252, 756)  # 6M, 12M, 36M

print("=" * 70)
print("  Momentum Lookback Period Sensitivity Analysis")
print("=" * 70)
print(f"  Short: {SHORT_CANDIDATES} days")
print(f"  Mid:   {MID_CANDIDATES} days")
print(f"  Long:  {LONG_CANDIDATES} days")

combos = []
for s, m, l in itertools.product(SHORT_CANDIDATES, MID_CANDIDATES, LONG_CANDIDATES):
    if s < m < l:  # 短期 < 中期 < 長期 を保証
        combos.append((s, m, l))

print(f"  Valid combinations: {len(combos)}")
print(flush=True)

# ============================================================
# 1. データ取得（1回だけ）
# ============================================================
print("\n=== Fetching S&P500 constituents ===", flush=True)
sp500_tickers = None
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

    print(f"  {len(sp500_tickers)} stocks, {len(set(ticker_sector.values()))} sectors", flush=True)
except Exception as e:
    print(f"  Wikipedia failed: {e}", flush=True)
    try:
        import csv
        url = 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            content = resp.read().decode('utf-8')
        reader = csv.DictReader(io.StringIO(content))
        sp500_tickers = []
        for row in reader:
            tk = row['Symbol'].replace('.', '-')
            sp500_tickers.append(tk)
            if 'Sector' in row:
                ticker_sector[tk] = row['Sector']
        print(f"  GitHub CSV: {len(sp500_tickers)} stocks", flush=True)
    except Exception as e2:
        print(f"  FATAL: {e2}")
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
        print(f"  Batch {i//batch_size+1}/{n_batches}: {len(all_close)} stocks",
              flush=True)
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

# 全候補期間のモメンタムを事前計算
print("Pre-computing all momentum lookbacks...", flush=True)
all_lookbacks = sorted(set(SHORT_CANDIDATES + MID_CANDIDATES + LONG_CANDIDATES))
mom_cache = {}
for lb in all_lookbacks:
    mom_cache[lb] = close_df.pct_change(lb, fill_method=None)
    print(f"  LB={lb} done", flush=True)

print(f"Valid tickers: {len(close_df.columns)}", flush=True)

# リバランス日（月次）
bt_dates = date_index[(date_index >= BT_START) & (date_index <= BT_END)]

rebal_dates = []
for dt in bt_dates:
    month_dates = bt_dates[(bt_dates.month == dt.month) & (bt_dates.year == dt.year)]
    if len(month_dates) > 0:
        last_day = month_dates[-1]
        if last_day not in rebal_dates:
            rebal_dates.append(last_day)
rebal_dates = sorted(set(rebal_dates))

rebal_exec_dates = []
for rd in rebal_dates:
    future = bt_dates[bt_dates > rd]
    if len(future) > 0:
        rebal_exec_dates.append((rd, future[0]))

print(f"Rebalance count: {len(rebal_exec_dates)}", flush=True)

# numpy配列
ret_np = ret_df.values
gap_ret_np = gap_ret_df.values
intra_ret_np = intra_ret_df.values
col_to_idx = {c: i for i, c in enumerate(ret_df.columns)}

# ============================================================
# 2. バックテスト関数
# ============================================================
def run_single(lb_short, lb_mid, lb_long):
    """指定された期間でバックテストを1回実行"""
    ms_df = mom_cache[lb_short]
    mm_df = mom_cache[lb_mid]
    ml_df = mom_cache[lb_long]

    current_holdings = {}
    next_rebal_idx = 0
    portfolio_returns = []

    for i, date in enumerate(bt_dates):
        date_idx = date_index.get_loc(date)

        is_rebalance = False
        if next_rebal_idx < len(rebal_exec_dates):
            judge_date, exec_date = rebal_exec_dates[next_rebal_idx]
            if date == exec_date:
                is_rebalance = True
                next_rebal_idx += 1

        if is_rebalance:
            old_holdings = dict(current_holdings)

            # 銘柄選定（セクターキャップ付き）
            jidx = date_index.get_loc(judge_date)
            ms = ms_df.iloc[jidx].dropna().sort_values(ascending=False)
            mm = mm_df.iloc[jidx].dropna().sort_values(ascending=False)
            ml = ml_df.iloc[jidx].dropna().sort_values(ascending=False)

            selected = []
            sector_count = {}

            def can_add(tk):
                if tk in selected:
                    return False
                sec = ticker_sector.get(tk, 'Unknown')
                return sector_count.get(sec, 0) < MAX_PER_SECTOR

            def add_tk(tk):
                selected.append(tk)
                sec = ticker_sector.get(tk, 'Unknown')
                sector_count[sec] = sector_count.get(sec, 0) + 1

            for bucket_df in [ms, mm, ml]:
                count = 0
                for tk in bucket_df.index:
                    if count >= N_PER_BUCKET:
                        break
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

            # gap + intraday
            if old_holdings:
                gap_r = 0.0
                for t, w in old_holdings.items():
                    if t in col_to_idx:
                        g = gap_ret_np[date_idx, col_to_idx[t]]
                        if not np.isnan(g):
                            gap_r += g * w
                intra_r = 0.0
                for t, w in current_holdings.items():
                    if t in col_to_idx:
                        ir = intra_ret_np[date_idx, col_to_idx[t]]
                        if not np.isnan(ir):
                            intra_r += ir * w
                r = (1 + gap_r) * (1 + intra_r) - 1
            else:
                intra_r = 0.0
                for t, w in current_holdings.items():
                    if t in col_to_idx:
                        ir = intra_ret_np[date_idx, col_to_idx[t]]
                        if not np.isnan(ir):
                            intra_r += ir * w
                r = intra_r

            portfolio_returns.append(r)
        else:
            if not current_holdings:
                portfolio_returns.append(0.0)
                continue
            r = 0.0
            for t, w in current_holdings.items():
                if t in col_to_idx:
                    daily_r = ret_np[date_idx, col_to_idx[t]]
                    if not np.isnan(daily_r):
                        r += daily_r * w
            portfolio_returns.append(r)

    rets = np.array(portfolio_returns)
    cum = np.cumprod(1 + rets)
    total = cum[-1] - 1
    years = len(rets) / 252
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 else 0
    vol = np.std(rets) * np.sqrt(252)
    sharpe = cagr / vol if vol > 0 else 0
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    mdd = dd.min()

    return sharpe, cagr, mdd, vol

# ============================================================
# 3. 全組み合わせ実行
# ============================================================
print(f"\n=== Running {len(combos)} combinations ===", flush=True)

results = []
for i, (s, m, l) in enumerate(combos):
    sharpe, cagr, mdd, vol = run_single(s, m, l)
    is_default = (s, m, l) == SMT_DEFAULT
    results.append({
        'short': s, 'mid': m, 'long': l,
        'short_m': s // 21, 'mid_m': m // 21, 'long_m': l // 21,
        'sharpe': sharpe, 'cagr': cagr, 'mdd': mdd, 'vol': vol,
        'is_default': is_default
    })
    marker = " <<<< SMT DEFAULT" if is_default else ""
    print(f"  [{i+1}/{len(combos)}] S={s}({s//21}M) M={m}({m//21}M) L={l}({l//21}M)"
          f"  Sharpe={sharpe:.2f}  CAGR={cagr*100:+.1f}%  MDD={mdd*100:.1f}%{marker}",
          flush=True)

# ============================================================
# 4. 結果分析
# ============================================================
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('sharpe', ascending=False).reset_index(drop=True)

print("\n" + "=" * 80)
print("  RESULTS RANKED BY SHARPE")
print("=" * 80)
print(f"  {'Rank':>4}  {'Short':>6}  {'Mid':>6}  {'Long':>6}  "
      f"{'Sharpe':>7}  {'CAGR':>7}  {'MDD':>7}  {'Vol':>6}  Note")
print("-" * 80)

for idx, row in results_df.iterrows():
    note = "** SMT **" if row['is_default'] else ""
    if idx == 0:
        note = "BEST" + (" / SMT" if row['is_default'] else "")
    print(f"  {idx+1:>4}  {row['short_m']:>4}M  {row['mid_m']:>4}M  {row['long_m']:>4}M  "
          f"{row['sharpe']:>6.2f}  {row['cagr']*100:>+6.1f}%  {row['mdd']*100:>+6.1f}%  "
          f"{row['vol']*100:>5.1f}%  {note}")

# 統計サマリー
print(f"\n  Sharpe: min={results_df['sharpe'].min():.2f}  "
      f"mean={results_df['sharpe'].mean():.2f}  "
      f"max={results_df['sharpe'].max():.2f}  "
      f"std={results_df['sharpe'].std():.2f}")
print(f"  CAGR:   min={results_df['cagr'].min()*100:+.1f}%  "
      f"mean={results_df['cagr'].mean()*100:+.1f}%  "
      f"max={results_df['cagr'].max()*100:+.1f}%")
print(f"  MDD:    min={results_df['mdd'].min()*100:.1f}%  "
      f"mean={results_df['mdd'].mean()*100:.1f}%  "
      f"max={results_df['mdd'].max()*100:.1f}%")

# SMTデフォルトの順位
smt_row = results_df[results_df['is_default']]
if len(smt_row) > 0:
    smt_rank = smt_row.index[0] + 1
    print(f"\n  SMTデフォルト(6M/12M/36M)の順位: {smt_rank}/{len(results_df)}")

# ============================================================
# 5. 各パラメータの感度分析
# ============================================================
print("\n" + "=" * 80)
print("  PARAMETER SENSITIVITY")
print("=" * 80)

for param_name, param_col, candidates in [
    ("Short-term", "short", SHORT_CANDIDATES),
    ("Mid-term", "mid", MID_CANDIDATES),
    ("Long-term", "long", LONG_CANDIDATES)
]:
    print(f"\n  [{param_name} Lookback]")
    print(f"  {'Period':>8}  {'Avg Sharpe':>11}  {'Avg CAGR':>10}  {'Avg MDD':>9}  {'Count':>6}")
    print(f"  {'-'*8}  {'-'*11}  {'-'*10}  {'-'*9}  {'-'*6}")
    for val in candidates:
        subset = results_df[results_df[param_col] == val]
        if len(subset) == 0:
            continue
        print(f"  {val//21:>5}M   {subset['sharpe'].mean():>10.2f}  "
              f"{subset['cagr'].mean()*100:>+9.1f}%  "
              f"{subset['mdd'].mean()*100:>+8.1f}%  {len(subset):>5}")

# ============================================================
# 6. ヒートマップ用データ出力
# ============================================================
# Short固定（SMTデフォルト=126）でのMid×Longヒートマップ
print("\n" + "=" * 80)
print("  HEATMAP: Short=6M fixed, Mid x Long")
print("=" * 80)
hm = results_df[results_df['short'] == 126].pivot_table(
    values='sharpe', index='mid_m', columns='long_m', aggfunc='first'
)
if not hm.empty:
    print("\n  Sharpe (Short=6M):")
    header = 'Mid\\Long'
    print(f"  {header:>10}", end='')
    for col in hm.columns:
        print(f"  {col}M".rjust(8), end='')
    print()
    for idx_val in hm.index:
        print(f"  {idx_val}M".rjust(10), end='')
        for col in hm.columns:
            v = hm.loc[idx_val, col]
            if pd.notna(v):
                print(f"  {v:.2f}".rjust(8), end='')
            else:
                print("      -".rjust(8), end='')
        print()

# Mid固定（SMTデフォルト=252）でのShort×Longヒートマップ
print(f"\n  Sharpe (Mid=12M):")
hm2 = results_df[results_df['mid'] == 252].pivot_table(
    values='sharpe', index='short_m', columns='long_m', aggfunc='first'
)
if not hm2.empty:
    header2 = 'Short\\Long'
    print(f"  {header2:>10}", end='')
    for col in hm2.columns:
        print(f"  {col}M".rjust(8), end='')
    print()
    for idx_val in hm2.index:
        print(f"  {idx_val}M".rjust(10), end='')
        for col in hm2.columns:
            v = hm2.loc[idx_val, col]
            if pd.notna(v):
                print(f"  {v:.2f}".rjust(8), end='')
            else:
                print("      -".rjust(8), end='')
        print()

print("\n" + "=" * 80)
print("  Done.", flush=True)
print("=" * 80)
