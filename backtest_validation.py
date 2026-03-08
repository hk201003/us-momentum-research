"""
SMT 米国モメンタム バックテスト検証スクリプト
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Part 1: 実在ETFとの比較（答え合わせ）
Part 2: 売買コスト・ターンオーバー分析
Part 3: 生存バイアス推定（Monte Carlo）

使い方:
  python backtest_validation.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import gc
import io
import random
import urllib.request
import warnings
import yfinance as yf
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

# ============================================================
# 共通パラメータ
# ============================================================
N_SHORT = 7
N_MID = 7
N_LONG = 7
LB_SHORT = 126
LB_MID = 252
LB_LONG = 756

REBAL_MONTHS_QUARTERLY = [2, 5, 8, 11]

BT_START = '2011-03-01'
BT_END = '2026-03-07'
DATA_START = '2008-01-01'

# 改良版パラメータ
TREND_SMA = 200
TREND_REDUCE = 0.5
VOL_WINDOW = 60
VOL_EXPONENT = 1.0
MAX_PER_SECTOR = 3

print("=" * 70)
print("  SMT US Momentum - Comprehensive Validation")
print("=" * 70)
print(f"  期間: {BT_START} ~ {BT_END}")
print(flush=True)

# ============================================================
# 1. S&P500 構成銘柄取得
# ============================================================
print("=== Fetching S&P500 constituents ===", flush=True)
sp500_tickers = None
ticker_sector = {}
sp500_table_global = None

try:
    req = urllib.request.Request(
        'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
        headers={'User-Agent': 'Mozilla/5.0'}
    )
    with urllib.request.urlopen(req) as resp:
        html_content = resp.read().decode('utf-8')
    tables = pd.read_html(io.StringIO(html_content))
    sp500_table_global = tables[0]
    sp500_tickers = sp500_table_global['Symbol'].str.replace('.', '-', regex=False).tolist()

    sector_col = None
    for col in sp500_table_global.columns:
        if 'GICS' in str(col) and 'Sector' in str(col):
            sector_col = col
            break
    if sector_col is None:
        for col in sp500_table_global.columns:
            if 'sector' in str(col).lower():
                sector_col = col
                break
    if sector_col is not None:
        for _, row in sp500_table_global.iterrows():
            tk = str(row['Symbol']).replace('.', '-')
            ticker_sector[tk] = str(row[sector_col])

    print(f"  Wikipedia: {len(sp500_tickers)} stocks, {len(set(ticker_sector.values()))} sectors", flush=True)
except Exception as e:
    print(f"  Wikipedia failed: {e}", flush=True)

if sp500_tickers is None:
    try:
        import csv
        url = 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv'
        req2 = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req2) as resp:
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
        print(f"  GitHub CSV failed: {e2}", flush=True)

if sp500_tickers is None or len(sp500_tickers) < 400:
    print("  ERROR: Cannot get S&P500 list.", flush=True)
    sys.exit(1)

# ============================================================
# 2. 株価データダウンロード（S&P500 + ETFs）
# ============================================================
print("\n=== Downloading price data ===", flush=True)

etf_tickers = ['SPY', 'MTUM', 'QMOM', 'PDP', 'SPMO']
all_dl_tickers = sorted(set(sp500_tickers + etf_tickers))
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
        print(f"  Batch {i//batch_size+1} error: {e}", flush=True)

print(f"Downloaded: {len(all_close)} stocks total", flush=True)

# ETFデータを分離
etf_close = {}
etf_open = {}
for tk in etf_tickers:
    if tk in all_close:
        etf_close[tk] = all_close.pop(tk)
        etf_open[tk] = all_open.pop(tk)
        print(f"  ETF {tk}: {len(etf_close[tk])} days ({etf_close[tk].index[0].strftime('%Y-%m-%d')} ~)", flush=True)
    else:
        print(f"  ETF {tk}: NOT AVAILABLE", flush=True)

spy_close_ser = etf_close.get('SPY')
spy_open_ser = etf_open.get('SPY')

# ============================================================
# 3. データフレーム構築
# ============================================================
print("\n=== Building DataFrames ===", flush=True)
all_dates = sorted(set().union(*(s.index for s in all_close.values())))
date_index = pd.DatetimeIndex(all_dates)

close_df = pd.concat({t: all_close[t].reindex(date_index) for t in all_close}, axis=1)
open_df = pd.concat({t: all_open[t].reindex(date_index) for t in all_open}, axis=1)

if spy_close_ser is not None:
    spy_series = spy_close_ser.reindex(date_index).ffill()
    spy_sma200 = spy_series.rolling(TREND_SMA).mean()
else:
    spy_series = None
    spy_sma200 = None

ret_df = close_df.pct_change(fill_method=None)
gap_ret_df = (open_df - close_df.shift(1)) / close_df.shift(1)
intra_ret_df = (close_df - open_df) / open_df.replace(0, np.nan)
vol_df = ret_df.rolling(VOL_WINDOW).std()

del all_close, all_open, open_df
gc.collect()

# Momentum
print("Computing momentum...", flush=True)
mom_short = close_df.pct_change(LB_SHORT, fill_method=None)
mom_mid = close_df.pct_change(LB_MID, fill_method=None)
mom_long = close_df.pct_change(LB_LONG, fill_method=None)

print(f"Valid tickers: {len(close_df.columns)}", flush=True)

# ============================================================
# 4. リバランス日の特定
# ============================================================
bt_dates = date_index[(date_index >= BT_START) & (date_index <= BT_END)]

def get_rebal_exec_dates(months_set):
    rebal_dates = []
    for dt in bt_dates:
        if dt.month in months_set:
            month_dates = bt_dates[(bt_dates.month == dt.month) & (bt_dates.year == dt.year)]
            if len(month_dates) > 0:
                last_day = month_dates[-1]
                if last_day not in rebal_dates:
                    rebal_dates.append(last_day)
    rebal_dates = sorted(set(rebal_dates))
    exec_dates = []
    for rd in rebal_dates:
        future = bt_dates[bt_dates > rd]
        if len(future) > 0:
            exec_dates.append((rd, future[0]))
    return exec_dates

quarterly_exec = get_rebal_exec_dates({2, 5, 8, 11})
monthly_exec = get_rebal_exec_dates(set(range(1, 13)))
bimonthly_exec = get_rebal_exec_dates({1, 3, 5, 7, 9, 11})

print(f"Rebalance counts: quarterly={len(quarterly_exec)}, bimonthly={len(bimonthly_exec)}, monthly={len(monthly_exec)}", flush=True)

# ============================================================
# 5. 銘柄選定ロジック
# ============================================================
ret_np = ret_df.values
gap_ret_np = gap_ret_df.values
intra_ret_np = intra_ret_df.values
col_to_idx = {c: i for i, c in enumerate(ret_df.columns)}

def select_original(date, universe=None):
    """SMT original selection. universe=None means all columns."""
    idx = date_index.get_loc(date)
    ms = mom_short.iloc[idx].dropna().sort_values(ascending=False)
    mm = mom_mid.iloc[idx].dropna().sort_values(ascending=False)
    ml = mom_long.iloc[idx].dropna().sort_values(ascending=False)

    if universe is not None:
        ms = ms[ms.index.isin(universe)]
        mm = mm[mm.index.isin(universe)]
        ml = ml[ml.index.isin(universe)]

    selected = []
    short_picks, mid_picks, long_picks = [], [], []
    for ticker in ms.index:
        if len(short_picks) >= N_SHORT: break
        if ticker not in selected:
            short_picks.append(ticker); selected.append(ticker)
    for ticker in mm.index:
        if len(mid_picks) >= N_MID: break
        if ticker not in selected:
            mid_picks.append(ticker); selected.append(ticker)
    for ticker in ml.index:
        if len(long_picks) >= N_LONG: break
        if ticker not in selected:
            long_picks.append(ticker); selected.append(ticker)
    return selected, short_picks, mid_picks, long_picks


def select_improved(date, universe=None):
    """Improved selection with sector cap."""
    idx = date_index.get_loc(date)
    ms = mom_short.iloc[idx].dropna().sort_values(ascending=False)
    mm = mom_mid.iloc[idx].dropna().sort_values(ascending=False)
    ml = mom_long.iloc[idx].dropna().sort_values(ascending=False)

    if universe is not None:
        ms = ms[ms.index.isin(universe)]
        mm = mm[mm.index.isin(universe)]
        ml = ml[ml.index.isin(universe)]

    selected = []
    sector_count = {}

    def can_add(ticker):
        if ticker in selected:
            return False
        sec = ticker_sector.get(ticker, 'Unknown')
        if sector_count.get(sec, 0) >= MAX_PER_SECTOR:
            return False
        return True

    def add_ticker(ticker):
        selected.append(ticker)
        sec = ticker_sector.get(ticker, 'Unknown')
        sector_count[sec] = sector_count.get(sec, 0) + 1

    short_picks, mid_picks, long_picks = [], [], []
    for ticker in ms.index:
        if len(short_picks) >= N_SHORT: break
        if can_add(ticker):
            short_picks.append(ticker); add_ticker(ticker)
    for ticker in mm.index:
        if len(mid_picks) >= N_MID: break
        if can_add(ticker):
            mid_picks.append(ticker); add_ticker(ticker)
    for ticker in ml.index:
        if len(long_picks) >= N_LONG: break
        if can_add(ticker):
            long_picks.append(ticker); add_ticker(ticker)
    return selected, short_picks, mid_picks, long_picks


def equal_weight(stocks, date):
    w = 1.0 / len(stocks) if stocks else 0
    return {t: w for t in stocks}


def inv_vol_weight(stocks, date):
    idx = date_index.get_loc(date)
    vols = {}
    for t in stocks:
        if t in vol_df.columns:
            v = vol_df.iloc[idx][t]
            if not np.isnan(v) and v > 0:
                vols[t] = v
    if not vols:
        w = 1.0 / len(stocks) if stocks else 0
        return {t: w for t in stocks}
    inv_vols = {t: (1.0 / v) ** VOL_EXPONENT for t, v in vols.items()}
    if len(inv_vols) < len(stocks):
        median_inv = np.median(list(inv_vols.values()))
        for t in stocks:
            if t not in inv_vols:
                inv_vols[t] = median_inv
    total = sum(inv_vols.values())
    if total <= 0:
        w = 1.0 / len(stocks)
        return {t: w for t in stocks}
    return {t: inv_vols[t] / total for t in stocks}


# ============================================================
# 6. 汎用バックテストエンジン（ターンオーバー追跡付き）
# ============================================================
def run_backtest(select_fn, weight_fn, use_trend, rebal_list, track_turnover=False):
    """Returns (daily_returns_series, turnover_records)"""
    portfolio_returns = []
    portfolio_dates = []
    current_holdings = {}
    next_rebal_idx = 0
    turnover_records = []

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
            stocks, sp, mp, lp = select_fn(judge_date)

            if stocks:
                weights = weight_fn(stocks, judge_date)
                if use_trend and spy_sma200 is not None:
                    spy_val = spy_series.get(judge_date, np.nan)
                    sma_val = spy_sma200.get(judge_date, np.nan)
                    if not np.isnan(spy_val) and not np.isnan(sma_val):
                        if spy_val < sma_val:
                            weights = {t: w * TREND_REDUCE for t, w in weights.items()}
                current_holdings = weights
            else:
                current_holdings = {}

            # Track turnover
            if track_turnover:
                old_set = set(old_holdings.keys())
                new_set = set(current_holdings.keys())
                if old_set or new_set:
                    # Turnover = sum of absolute weight changes / 2
                    all_tks = old_set | new_set
                    turnover = sum(abs(current_holdings.get(t, 0) - old_holdings.get(t, 0)) for t in all_tks) / 2
                    n_changed = len(old_set.symmetric_difference(new_set))
                    n_trades = len(new_set - old_set) + len(old_set - new_set)
                    turnover_records.append({
                        'date': date,
                        'turnover': turnover,
                        'n_changed': n_changed,
                        'n_trades': n_trades,
                        'n_old': len(old_set),
                        'n_new': len(new_set),
                    })

            # gap/intraday
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
            portfolio_dates.append(date)
        else:
            if not current_holdings:
                portfolio_returns.append(0.0)
                portfolio_dates.append(date)
                continue
            r = 0.0
            for t, w in current_holdings.items():
                if t in col_to_idx:
                    daily_r = ret_np[date_idx, col_to_idx[t]]
                    if not np.isnan(daily_r):
                        r += daily_r * w
            portfolio_returns.append(r)
            portfolio_dates.append(date)

    return pd.Series(portfolio_returns, index=portfolio_dates), turnover_records


# ============================================================
# 7. メインバックテスト実行
# ============================================================
print("\n=== Running main backtests ===", flush=True)
print("  [1/2] SMT Original (quarterly, equal weight)...", flush=True)
ret_original, turnover_quarterly = run_backtest(
    select_original, equal_weight, False, quarterly_exec, track_turnover=True)

print("  [2/2] Improved (monthly, inv-vol, trend, sector cap)...", flush=True)
ret_improved, turnover_monthly = run_backtest(
    select_improved, inv_vol_weight, True, monthly_exec, track_turnover=True)

print("  Done.", flush=True)

# ============================================================
# 統計計算関数
# ============================================================
def calc_stats(rets, label=None, verbose=False):
    cum_r = (1 + rets).cumprod()
    total_r = cum_r.iloc[-1] - 1
    years = len(rets) / 252
    cagr = (1 + total_r) ** (1 / years) - 1 if years > 0 else 0
    vol = rets.std() * np.sqrt(252)
    sharpe = cagr / vol if vol > 0 else 0
    peak = cum_r.cummax()
    dd = (cum_r - peak) / peak
    mdd = dd.min()
    if verbose and label:
        print(f"  [{label}] Sharpe={sharpe:.2f} CAGR={cagr*100:+.1f}% MDD={mdd*100:.1f}% Vol={vol*100:.1f}%")
    return {'sharpe': sharpe, 'cagr': cagr, 'mdd': mdd, 'vol': vol}


# ############################################################
#  Part 1: 実在ETFとの比較（答え合わせ）
# ############################################################
print("\n")
print("#" * 70)
print("#  Part 1: Real ETF Comparison (答え合わせ)")
print("#" * 70)

# ETFの日次リターン
etf_returns = {}
for tk in etf_tickers:
    if tk in etf_close:
        ser = etf_close[tk].reindex(date_index).ffill()
        etf_returns[tk] = ser.pct_change().fillna(0)

# --- 共通期間の統計比較 ---
# それぞれのETFの開始日を把握
etf_starts = {}
for tk, ser in etf_close.items():
    etf_starts[tk] = ser.index[0]

print(f"\n  ETF data availability:")
for tk in etf_tickers:
    if tk in etf_starts:
        print(f"    {tk}: since {etf_starts[tk].strftime('%Y-%m-%d')}")
    else:
        print(f"    {tk}: NOT AVAILABLE")

# 全ETFが揃う期間と、各ETFとの2者比較を行う
# まずSPYが確実にある期間での比較
print(f"\n  --- Full Period Stats ({BT_START} ~ {BT_END}) ---")
print(f"  {'Strategy/ETF':<28} {'Sharpe':>8} {'CAGR':>8} {'MDD':>8} {'Vol':>8}")
print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

all_comparison = {}
for name, rets in [("SMT Original", ret_original), ("SMT Improved", ret_improved)]:
    stats = calc_stats(rets)
    all_comparison[name] = stats
    print(f"  {name:<28} {stats['sharpe']:>7.2f} {stats['cagr']*100:>+7.1f}% {stats['mdd']*100:>+7.1f}% {stats['vol']*100:>7.1f}%")

for tk in etf_tickers:
    if tk in etf_returns:
        # Align to backtest dates
        aligned = etf_returns[tk].reindex(ret_original.index).fillna(0)
        stats = calc_stats(aligned)
        all_comparison[tk] = stats
        print(f"  {tk:<28} {stats['sharpe']:>7.2f} {stats['cagr']*100:>+7.1f}% {stats['mdd']*100:>+7.1f}% {stats['vol']*100:>7.1f}%")

# --- Annual returns comparison ---
print(f"\n  --- Annual Returns ---")
header_items = ["Year", "SMT Orig", "SMT Impr"]
etfs_available = [tk for tk in ['SPY', 'MTUM', 'PDP'] if tk in etf_returns]
header_items.extend(etfs_available)
header = "  " + "  ".join(f"{h:>10}" for h in header_items)
print(header)
print("  " + "  ".join("-" * 10 for _ in header_items))

for year in range(2011, 2027):
    yr_mask = ret_original.index.year == year
    if yr_mask.sum() == 0:
        continue
    vals = [f"{year:>10}"]
    vals.append(f"{((1+ret_original[yr_mask]).prod()-1)*100:>+9.1f}%")
    vals.append(f"{((1+ret_improved[yr_mask]).prod()-1)*100:>+9.1f}%")
    for tk in etfs_available:
        aligned = etf_returns[tk].reindex(ret_original.index).fillna(0)
        yr_ret = (1 + aligned[yr_mask]).prod() - 1
        vals.append(f"{yr_ret*100:>+9.1f}%")
    print("  " + "  ".join(vals))

# --- Correlation matrix ---
print(f"\n  --- Correlation Matrix (daily returns) ---")
corr_data = {}
corr_data['SMT_Orig'] = ret_original
corr_data['SMT_Impr'] = ret_improved
for tk in etf_tickers:
    if tk in etf_returns:
        corr_data[tk] = etf_returns[tk].reindex(ret_original.index).fillna(0)

corr_df = pd.DataFrame(corr_data)
corr_matrix = corr_df.corr()
labels = list(corr_matrix.columns)
print(f"  {'':>12}", end="")
for lb in labels:
    print(f"{lb:>12}", end="")
print()
for i, lb1 in enumerate(labels):
    print(f"  {lb1:>12}", end="")
    for j, lb2 in enumerate(labels):
        print(f"{corr_matrix.iloc[i, j]:>12.3f}", end="")
    print()

# --- Rolling 1-year excess return vs SPY ---
if 'SPY' in etf_returns:
    print(f"\n  --- Rolling 1-Year Excess Return vs SPY (sampled every 6 months) ---")
    spy_aligned = etf_returns['SPY'].reindex(ret_original.index).fillna(0)

    def rolling_annual(rets, window=252):
        cum = (1 + rets).cumprod()
        rolling = cum / cum.shift(window) - 1
        return rolling

    roll_orig = rolling_annual(ret_original)
    roll_impr = rolling_annual(ret_improved)
    roll_spy = rolling_annual(spy_aligned)

    excess_orig = roll_orig - roll_spy
    excess_impr = roll_impr - roll_spy

    # MTUM excess
    excess_mtum = None
    if 'MTUM' in etf_returns:
        mtum_aligned = etf_returns['MTUM'].reindex(ret_original.index).fillna(0)
        roll_mtum = rolling_annual(mtum_aligned)
        excess_mtum = roll_mtum - roll_spy

    # Sample every ~126 days
    sample_dates = roll_orig.dropna().index[::126]
    print(f"  {'Date':>12}  {'SMT Orig':>10}  {'SMT Impr':>10}", end="")
    if excess_mtum is not None:
        print(f"  {'MTUM':>10}", end="")
    print()
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}", end="")
    if excess_mtum is not None:
        print(f"  {'-'*10}", end="")
    print()

    for d in sample_dates:
        eo = excess_orig.get(d, np.nan)
        ei = excess_impr.get(d, np.nan)
        if np.isnan(eo):
            continue
        print(f"  {d.strftime('%Y-%m-%d'):>12}  {eo*100:>+9.1f}%  {ei*100:>+9.1f}%", end="")
        if excess_mtum is not None:
            em = excess_mtum.get(d, np.nan)
            if not np.isnan(em):
                print(f"  {em*100:>+9.1f}%", end="")
            else:
                print(f"  {'N/A':>10}", end="")
        print()

# ############################################################
#  Part 2: 売買コスト・ターンオーバー分析
# ############################################################
print("\n")
print("#" * 70)
print("#  Part 2: Transaction Cost & Turnover Analysis")
print("#" * 70)

def analyze_turnover(turnover_records, name, returns):
    to_df = pd.DataFrame(turnover_records)
    if to_df.empty:
        print(f"\n  [{name}] No turnover data.")
        return {}

    avg_turnover = to_df['turnover'].mean()
    avg_trades = to_df['n_trades'].mean()
    years = len(returns) / 252

    # Annual turnover
    to_df['year'] = pd.DatetimeIndex(to_df['date']).year
    annual_turnover = to_df.groupby('year')['turnover'].sum()
    avg_annual_turnover = annual_turnover.mean()

    # Number of rebalances per year
    rebals_per_year = len(to_df) / years

    print(f"\n  [{name}]")
    print(f"  Rebalances: {len(to_df)} ({rebals_per_year:.1f}/year)")
    print(f"  Avg turnover per rebalance: {avg_turnover*100:.1f}%")
    print(f"  Avg trades per rebalance: {avg_trades:.1f}")
    print(f"  Avg annual turnover (one-way): {avg_annual_turnover*100:.0f}%")

    print(f"\n  Annual Turnover:")
    print(f"  {'Year':>6}  {'Turnover':>10}  {'Rebals':>8}")
    for yr in sorted(annual_turnover.index):
        n_rebals = len(to_df[to_df['year'] == yr])
        print(f"  {yr:>6}  {annual_turnover[yr]*100:>9.0f}%  {n_rebals:>8}")

    return {
        'avg_turnover_per_rebal': avg_turnover,
        'avg_annual_turnover': avg_annual_turnover,
        'rebals_per_year': rebals_per_year,
        'avg_trades': avg_trades,
    }


print("\n  --- Turnover Statistics ---")
to_stats_q = analyze_turnover(turnover_quarterly, "SMT Original (Quarterly)", ret_original)
to_stats_m = analyze_turnover(turnover_monthly, "SMT Improved (Monthly)", ret_improved)

# --- Cost scenarios ---
print(f"\n  --- Cost Impact Analysis ---")

def apply_costs(returns, turnover_records, cost_per_trade, annual_fee=0.0, name=""):
    """Apply transaction costs and annual fees to returns."""
    if not turnover_records:
        return returns

    # Build a daily cost series
    cost_series = pd.Series(0.0, index=returns.index)

    # Annual fee as daily deduction
    daily_fee = annual_fee / 252

    # Transaction costs at rebalance dates
    for rec in turnover_records:
        d = rec['date']
        if d in cost_series.index:
            # Cost = turnover (one-way) * cost_per_trade * 2 (buy + sell)
            cost_series[d] = rec['turnover'] * cost_per_trade * 2

    # Apply
    adj_returns = returns - cost_series - daily_fee
    return adj_returns


scenarios = [
    ("A: SMT Fund (0.77% fee + 0.1% slip)", 0.001, 0.0077),
    ("B: Discount Broker (0.05% per trade)", 0.0005, 0.0),
    ("C: Higher Cost (0.3% per trade)", 0.003, 0.0),
]

for strat_name, ret, to_recs in [("SMT Original", ret_original, turnover_quarterly),
                                   ("SMT Improved", ret_improved, turnover_monthly)]:
    print(f"\n  [{strat_name}]")
    base_stats = calc_stats(ret)
    print(f"  {'Scenario':<42} {'Sharpe':>8} {'CAGR':>8} {'MDD':>8} {'Cost/yr':>8}")
    print(f"  {'-'*42} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'No costs (baseline)':<42} {base_stats['sharpe']:>7.2f} {base_stats['cagr']*100:>+7.1f}% {base_stats['mdd']*100:>+7.1f}%  {'--':>6}")

    for sc_name, cpt, annual_fee in scenarios:
        adj = apply_costs(ret, to_recs, cpt, annual_fee)
        adj_stats = calc_stats(adj)
        cost_drag = (base_stats['cagr'] - adj_stats['cagr']) * 100
        print(f"  {sc_name:<42} {adj_stats['sharpe']:>7.2f} {adj_stats['cagr']*100:>+7.1f}% {adj_stats['mdd']*100:>+7.1f}% {cost_drag:>+7.1f}%")


# ############################################################
#  Part 3: 生存バイアス推定
# ############################################################
print("\n")
print("#" * 70)
print("#  Part 3: Survivorship Bias Estimation")
print("#" * 70)

# --- 3a: Data availability by era ---
print(f"\n  --- Data Availability by Era ---")
stock_cols = [c for c in close_df.columns if c not in etf_tickers]
for cutoff_year, cutoff_date in [(2011, '2011-03-01'), (2016, '2016-01-01'), (2020, '2020-01-01')]:
    n_available = 0
    for t in stock_cols:
        valid = close_df[t].dropna()
        if len(valid) > 0 and valid.index[0] <= pd.Timestamp(cutoff_date):
            n_available += 1
    print(f"  Stocks with data back to {cutoff_year}: {n_available} / {len(stock_cols)} ({n_available/len(stock_cols)*100:.0f}%)")

# --- 3b: Recently added stocks momentum ranking ---
print(f"\n  --- Recently Added S&P500 Stocks: Were they high-momentum at addition? ---")

date_added_col = None
if sp500_table_global is not None:
    for col in sp500_table_global.columns:
        if 'date' in str(col).lower() and 'added' in str(col).lower():
            date_added_col = col
            break
        if 'date' in str(col).lower() and 'first' in str(col).lower():
            date_added_col = col
            break

if date_added_col is not None:
    print(f"  Using column: '{date_added_col}'")
    recent_adds = []
    for _, row in sp500_table_global.iterrows():
        tk = str(row['Symbol']).replace('.', '-')
        da = str(row[date_added_col])
        try:
            add_date = pd.Timestamp(da)
            if add_date >= pd.Timestamp('2018-01-01'):
                recent_adds.append((tk, add_date))
        except:
            pass

    print(f"  Stocks added since 2018: {len(recent_adds)}")

    # For each recently added stock, check their 12M momentum rank
    # at the nearest available date to their addition
    high_mom_count = 0
    analyzed = 0
    print(f"\n  {'Ticker':<8} {'Added':>12} {'12M Mom Rank':>14} {'Percentile':>12} {'High Mom?':>10}")
    print(f"  {'-'*8} {'-'*12} {'-'*14} {'-'*12} {'-'*10}")

    for tk, add_date in sorted(recent_adds, key=lambda x: x[1], reverse=True)[:30]:
        if tk not in close_df.columns:
            continue
        # Find nearest date in index
        nearest_dates = date_index[date_index <= add_date]
        if len(nearest_dates) == 0:
            continue
        check_date = nearest_dates[-1]
        idx = date_index.get_loc(check_date)
        if idx < LB_MID:
            continue

        mom_vals = mom_mid.iloc[idx].dropna().sort_values(ascending=False)
        if tk not in mom_vals.index:
            continue

        rank = (mom_vals.index.get_loc(tk) + 1) if tk in mom_vals.index else None
        total = len(mom_vals)
        if rank is not None:
            pctile = rank / total * 100
            is_high = pctile <= 20  # top 20%
            if is_high:
                high_mom_count += 1
            analyzed += 1
            marker = "YES" if is_high else ""
            print(f"  {tk:<8} {add_date.strftime('%Y-%m-%d'):>12} {rank:>8}/{total:<5} {pctile:>10.0f}%  {marker:>8}")

    if analyzed > 0:
        pct_high = high_mom_count / analyzed * 100
        print(f"\n  Summary: {high_mom_count}/{analyzed} ({pct_high:.0f}%) of recently added stocks")
        print(f"  were in the top 20% momentum ranking at addition time.")
        if pct_high > 30:
            print(f"  --> This suggests significant survivorship bias: stocks enter S&P500")
            print(f"      partly BECAUSE they had high momentum (strong price performance).")
        else:
            print(f"  --> Survivorship bias from recent additions appears moderate.")
else:
    print(f"  Could not find 'Date added' column in Wikipedia table.")
    if sp500_table_global is not None:
        print(f"  Available columns: {list(sp500_table_global.columns)}")

# --- 3c: Degraded Universe Monte Carlo ---
print(f"\n  --- Degraded Universe Monte Carlo ---")
print(f"  Running backtest with randomly removed stocks to test sensitivity...")
print(f"  Degradation levels: 10%, 20%, 30% of universe removed")
print(f"  Iterations per level: 20")

full_universe = set(close_df.columns)
full_stats = calc_stats(ret_original)
print(f"\n  Full Universe: {len(full_universe)} stocks, Sharpe={full_stats['sharpe']:.2f}, CAGR={full_stats['cagr']*100:+.1f}%")

degradation_results = {}
random.seed(42)

for pct_remove in [0.10, 0.20, 0.30]:
    n_remove = int(len(full_universe) * pct_remove)
    sharpes = []
    cagrs = []
    print(f"\n  Removing {pct_remove*100:.0f}% ({n_remove} stocks): ", end="", flush=True)

    for iteration in range(20):
        # Random subset
        removed = set(random.sample(list(full_universe), n_remove))
        remaining = full_universe - removed

        def select_degraded(date, _universe=remaining):
            return select_original(date, universe=_universe)

        mc_ret, _ = run_backtest(select_degraded, equal_weight, False, quarterly_exec, track_turnover=False)
        stats = calc_stats(mc_ret)
        sharpes.append(stats['sharpe'])
        cagrs.append(stats['cagr'])
        print(f".", end="", flush=True)

    print(f" done")
    degradation_results[pct_remove] = {
        'sharpes': sharpes,
        'cagrs': cagrs,
        'mean_sharpe': np.mean(sharpes),
        'std_sharpe': np.std(sharpes),
        'mean_cagr': np.mean(cagrs),
        'min_sharpe': np.min(sharpes),
        'max_sharpe': np.max(sharpes),
    }

# Summary table
print(f"\n  --- Monte Carlo Results ---")
print(f"  {'Universe':<20} {'Mean Sharpe':>12} {'Std':>8} {'Min':>8} {'Max':>8} {'Mean CAGR':>10}")
print(f"  {'-'*20} {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
print(f"  {'Full (0% removed)':<20} {full_stats['sharpe']:>11.2f} {'--':>8} {'--':>8} {'--':>8} {full_stats['cagr']*100:>+9.1f}%")

for pct in [0.10, 0.20, 0.30]:
    r = degradation_results[pct]
    label = f"-{pct*100:.0f}% ({int(len(full_universe)*pct)} stocks)"
    print(f"  {label:<20} {r['mean_sharpe']:>11.2f} {r['std_sharpe']:>7.2f} {r['min_sharpe']:>7.2f} {r['max_sharpe']:>7.2f} {r['mean_cagr']*100:>+9.1f}%")

# Bias estimate
sharpe_drop_10 = full_stats['sharpe'] - degradation_results[0.10]['mean_sharpe']
sharpe_drop_20 = full_stats['sharpe'] - degradation_results[0.20]['mean_sharpe']
sharpe_drop_30 = full_stats['sharpe'] - degradation_results[0.30]['mean_sharpe']

print(f"\n  Sharpe degradation from full universe:")
print(f"    -10%: {sharpe_drop_10:+.3f}")
print(f"    -20%: {sharpe_drop_20:+.3f}")
print(f"    -30%: {sharpe_drop_30:+.3f}")

if sharpe_drop_10 < 0.05:
    print(f"\n  --> Strategy is ROBUST to universe changes (small Sharpe impact).")
    print(f"      Survivorship bias has limited effect on the momentum ranking.")
elif sharpe_drop_10 < 0.15:
    print(f"\n  --> MODERATE sensitivity to universe composition.")
    print(f"      Some survivorship bias is likely inflating results.")
else:
    print(f"\n  --> HIGH sensitivity to universe composition.")
    print(f"      Survivorship bias is likely a significant factor in backtest results.")
    print(f"      Real-world performance may be meaningfully worse.")

# ============================================================
# Final Summary
# ============================================================
print("\n")
print("=" * 70)
print("  VALIDATION SUMMARY")
print("=" * 70)

print(f"""
  Part 1 - ETF Comparison:
    Our SMT Original Sharpe: {all_comparison.get('SMT Original', {}).get('sharpe', 0):.2f}
    Our SMT Improved Sharpe: {all_comparison.get('SMT Improved', {}).get('sharpe', 0):.2f}
    MTUM Sharpe:             {all_comparison.get('MTUM', {}).get('sharpe', 0):.2f}
    SPY Sharpe:              {all_comparison.get('SPY', {}).get('sharpe', 0):.2f}

  Part 2 - Cost Impact:
    Quarterly rebal avg annual turnover: {to_stats_q.get('avg_annual_turnover', 0)*100:.0f}%
    Monthly rebal avg annual turnover:   {to_stats_m.get('avg_annual_turnover', 0)*100:.0f}%
    Cost drag (discount broker): ~0.05% per trade

  Part 3 - Survivorship Bias:
    Sharpe drop with -10% universe: {sharpe_drop_10:+.3f}
    Sharpe drop with -20% universe: {sharpe_drop_20:+.3f}
    Sharpe drop with -30% universe: {sharpe_drop_30:+.3f}
""")
print("=" * 70)
print("  Done.", flush=True)
print("=" * 70)
