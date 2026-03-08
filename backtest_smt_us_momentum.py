"""
SMT 米国株式モメンタムファンド バックテスト
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ルール（公開情報に基づく再現）:
  - ユニバース: 時価総額上位約500銘柄 → S&P500構成銘柄で代用
  - 短期枠: 6ヶ月(126営業日)リターン上位7銘柄
  - 中期枠: 12ヶ月(252営業日)リターン上位7銘柄
  - 長期枠: 36ヶ月(756営業日)リターン上位7銘柄
  - 重複: 短期優先→中期→長期。重複銘柄は次点に繰り下げ
  - 配分: 21銘柄均等配分
  - リバランス: 年4回（2/5/8/11月末基準、翌営業日寄りで執行）
  - gap/intraday分離: リバランス日はgap(旧)+intraday(新)

注意:
  - ユニバースは現在のS&P500構成銘柄で固定（生存バイアスあり）
  - 信託報酬・売買コストは未控除
  - WikipediaからS&P500銘柄リストを自動取得します

使い方:
  pip install yfinance pandas numpy
  python backtest_smt_us_momentum.py
"""
import gc
import io
import urllib.request
import yfinance as yf
import pandas as pd
import numpy as np

# ============================================================
# パラメータ
# ============================================================
N_SHORT = 7       # 短期枠の銘柄数
N_MID = 7         # 中期枠の銘柄数
N_LONG = 7        # 長期枠の銘柄数
LB_SHORT = 126    # 6ヶ月 ≒ 126営業日
LB_MID = 252      # 12ヶ月 ≒ 252営業日
LB_LONG = 756     # 36ヶ月 ≒ 756営業日
REBAL_MONTHS = [2, 5, 8, 11]  # リバランス月

BT_START = '2011-03-01'
BT_END = '2026-03-07'
DATA_START = '2008-01-01'  # 36ヶ月ルックバック用に余裕を持つ

print(f"=== SMT US Momentum Fund Backtest: {BT_START} ~ {BT_END} ===", flush=True)

# ============================================================
# 1. S&P500 構成銘柄取得
# ============================================================
print("\n=== Fetching S&P500 constituents ===", flush=True)
sp500_tickers = None

# 方法1: Wikipedia
try:
    req = urllib.request.Request(
        'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
        headers={'User-Agent': 'Mozilla/5.0'}
    )
    with urllib.request.urlopen(req) as resp:
        html_content = resp.read().decode('utf-8')
    sp500_table = pd.read_html(io.StringIO(html_content))[0]
    sp500_tickers = sp500_table['Symbol'].str.replace('.', '-', regex=False).tolist()
    print(f"  Wikipedia: {len(sp500_tickers)} stocks", flush=True)
except Exception as e:
    print(f"  Wikipedia failed: {e}", flush=True)

# 方法2: GitHub CSV（フォールバック）
if sp500_tickers is None:
    try:
        import csv
        url = 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            content = resp.read().decode('utf-8')
        reader = csv.DictReader(io.StringIO(content))
        sp500_tickers = [row['Symbol'].replace('.', '-') for row in reader]
        print(f"  GitHub CSV: {len(sp500_tickers)} stocks", flush=True)
    except Exception as e:
        print(f"  GitHub CSV failed: {e}", flush=True)

if sp500_tickers is None or len(sp500_tickers) < 400:
    print("  ERROR: Cannot get S&P500 list.", flush=True)
    exit(1)

# ============================================================
# 2. 株価データダウンロード
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
        print(f"  Batch {i//batch_size+1}/{n_batches}: {len(all_close)} stocks",
              flush=True)
    except Exception as e:
        print(f"  Batch {i//batch_size+1} error: {e}", flush=True)

print(f"Downloaded: {len(all_close)} stocks", flush=True)

# Benchmark
print("\n=== Benchmark (SPY) ===", flush=True)
spy_close_ser = all_close.pop('SPY', None)
spy_open_ser = all_open.pop('SPY', None)

if spy_close_ser is None:
    print("  WARNING: SPY data not available", flush=True)

# ============================================================
# 3. データフレーム構築
# ============================================================
print("\n=== Building DataFrames ===", flush=True)
all_dates = sorted(set().union(*(s.index for s in all_close.values())))
date_index = pd.DatetimeIndex(all_dates)

close_df = pd.concat({t: all_close[t].reindex(date_index) for t in all_close},
                     axis=1)
open_df = pd.concat({t: all_open[t].reindex(date_index) for t in all_open},
                    axis=1)

if spy_close_ser is not None:
    spy_series = spy_close_ser.reindex(date_index).ffill()
else:
    spy_series = None

ret_df = close_df.pct_change(fill_method=None)
gap_ret_df = (open_df - close_df.shift(1)) / close_df.shift(1)
intra_ret_df = (close_df - open_df) / open_df.replace(0, np.nan)

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

rebal_dates = []
for dt in bt_dates:
    if dt.month in REBAL_MONTHS:
        month_dates = bt_dates[(bt_dates.month == dt.month) &
                               (bt_dates.year == dt.year)]
        if len(month_dates) > 0:
            last_day = month_dates[-1]
            if last_day not in rebal_dates:
                rebal_dates.append(last_day)

rebal_dates = sorted(set(rebal_dates))

# 翌営業日（執行日）
rebal_exec_dates = []
for rd in rebal_dates:
    future = bt_dates[bt_dates > rd]
    if len(future) > 0:
        rebal_exec_dates.append((rd, future[0]))

print(f"Rebalance count: {len(rebal_exec_dates)}", flush=True)

# ============================================================
# 5. 銘柄選定ロジック
# ============================================================
def select_momentum_stocks(date):
    """SMT fund stock selection logic"""
    idx = date_index.get_loc(date)

    ms = mom_short.iloc[idx].dropna().sort_values(ascending=False)
    mm = mom_mid.iloc[idx].dropna().sort_values(ascending=False)
    ml = mom_long.iloc[idx].dropna().sort_values(ascending=False)

    selected = []

    # Short-term: top 7 by 6-month return
    short_picks = []
    for ticker in ms.index:
        if len(short_picks) >= N_SHORT:
            break
        if ticker not in selected:
            short_picks.append(ticker)
            selected.append(ticker)

    # Mid-term: top 7 by 12-month return (no overlap)
    mid_picks = []
    for ticker in mm.index:
        if len(mid_picks) >= N_MID:
            break
        if ticker not in selected:
            mid_picks.append(ticker)
            selected.append(ticker)

    # Long-term: top 7 by 36-month return (no overlap)
    long_picks = []
    for ticker in ml.index:
        if len(long_picks) >= N_LONG:
            break
        if ticker not in selected:
            long_picks.append(ticker)
            selected.append(ticker)

    return selected, short_picks, mid_picks, long_picks

# ============================================================
# 6. バックテスト実行
# ============================================================
print("\n=== Running backtest ===", flush=True)

ret_np = ret_df.values
gap_ret_np = gap_ret_df.values
intra_ret_np = intra_ret_df.values
col_to_idx = {c: i for i, c in enumerate(ret_df.columns)}

portfolio_returns = []
portfolio_dates = []
current_holdings = []
weight = 0.0
next_rebal_idx = 0

for i, date in enumerate(bt_dates):
    date_idx = date_index.get_loc(date)

    is_rebalance = False
    if next_rebal_idx < len(rebal_exec_dates):
        judge_date, exec_date = rebal_exec_dates[next_rebal_idx]
        if date == exec_date:
            is_rebalance = True
            next_rebal_idx += 1

    if is_rebalance:
        old_holdings = current_holdings.copy()
        new_holdings, sp, mp, lp = select_momentum_stocks(judge_date)
        current_holdings = new_holdings
        weight = 1.0 / len(current_holdings) if current_holdings else 0.0

        if old_holdings:
            old_weight = 1.0 / len(old_holdings)
            gap_r = 0.0
            for t in old_holdings:
                if t in col_to_idx:
                    g = gap_ret_np[date_idx, col_to_idx[t]]
                    if not np.isnan(g):
                        gap_r += g * old_weight

            intra_r = 0.0
            for t in current_holdings:
                if t in col_to_idx:
                    ir = intra_ret_np[date_idx, col_to_idx[t]]
                    if not np.isnan(ir):
                        intra_r += ir * weight

            r = (1 + gap_r) * (1 + intra_r) - 1
        else:
            intra_r = 0.0
            for t in current_holdings:
                if t in col_to_idx:
                    ir = intra_ret_np[date_idx, col_to_idx[t]]
                    if not np.isnan(ir):
                        intra_r += ir * weight
            r = intra_r

        portfolio_returns.append(r)
        portfolio_dates.append(date)

        if len(rebal_exec_dates) <= 50 or next_rebal_idx % 10 == 1:
            print(f"  Rebalance {date.strftime('%Y-%m-%d')}: {len(current_holdings)} stocks "
                  f"(S:{len(sp)} M:{len(mp)} L:{len(lp)})", flush=True)
    else:
        if not current_holdings:
            portfolio_returns.append(0.0)
            portfolio_dates.append(date)
            continue

        r = 0.0
        for t in current_holdings:
            if t in col_to_idx:
                daily_r = ret_np[date_idx, col_to_idx[t]]
                if not np.isnan(daily_r):
                    r += daily_r * weight

        portfolio_returns.append(r)
        portfolio_dates.append(date)

# ============================================================
# 7. 成績計算
# ============================================================
print("\n" + "=" * 60, flush=True)
print("  SMT US Momentum Fund - Backtest Results", flush=True)
print("=" * 60, flush=True)

returns = pd.Series(portfolio_returns, index=portfolio_dates)
cum = (1 + returns).cumprod()

spy_bt = spy_series.reindex(returns.index).pct_change().fillna(0) if spy_series is not None else None


def calc_stats(rets, label):
    cum_r = (1 + rets).cumprod()
    total_r = cum_r.iloc[-1] - 1
    years = len(rets) / 252
    cagr = (1 + total_r) ** (1 / years) - 1 if years > 0 else 0
    vol = rets.std() * np.sqrt(252)
    sharpe = cagr / vol if vol > 0 else 0

    peak = cum_r.cummax()
    dd = (cum_r - peak) / peak
    mdd = dd.min()

    weekly = rets.resample('W').sum()
    win_rate = (weekly > 0).mean()

    print(f"\n  [{label}]")
    print(f"  Sharpe:     {sharpe:.2f}")
    print(f"  CAGR:       {cagr*100:+.1f}%")
    print(f"  MDD:        {mdd*100:.1f}%")
    print(f"  Vol:        {vol*100:.1f}%")
    print(f"  Total:      {total_r*100:+.1f}%")
    print(f"  Win rate:   {win_rate*100:.1f}% (weekly)")

    return sharpe, cagr, mdd, vol


smt_stats = calc_stats(returns, 'SMT US Momentum')
spy_stats = calc_stats(spy_bt, 'SPY') if spy_bt is not None else None

# Annual returns
print("\n  [Annual Returns]")
print(f"  {'Year':>6}  {'SMT':>10}  {'SPY':>10}  {'Excess':>10}")
print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}")
for year in range(2011, 2027):
    yr_mask = returns.index.year == year
    if yr_mask.sum() == 0:
        continue
    smt_yr = (1 + returns[yr_mask]).prod() - 1
    if spy_bt is not None:
        spy_yr = (1 + spy_bt[yr_mask]).prod() - 1
        excess = smt_yr - spy_yr
        print(f"  {year:>6}  {smt_yr*100:>+9.1f}%  {spy_yr*100:>+9.1f}%  {excess*100:>+9.1f}%")
    else:
        print(f"  {year:>6}  {smt_yr*100:>+9.1f}%")

print("\n" + "=" * 60)
print("  Done.", flush=True)
print("=" * 60)
