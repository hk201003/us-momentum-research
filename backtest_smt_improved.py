"""
SMT 米国株式モメンタムファンド 改良版バックテスト
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SMTオリジナルの構造的弱点を4つ改善:
  1. トレンドフィルター追加: SPY >= SMA200 → フル投資 / SPY < SMA200 → 50%縮小
  2. 逆ボラティリティ加重: 等ウエイト → 低ボラ銘柄に多く配分（リスク寄与を均等化）
  3. 月次リバランス: 年4回 → 毎月末（モメンタム減衰への対応）
  4. セクター分散: 1セクターから最大3銘柄まで

ベース戦略（SMTと同じ部分）:
  - ユニバース: S&P500構成銘柄
  - 短期枠(6M) 7銘柄 + 中期枠(12M) 7銘柄 + 長期枠(36M) 7銘柄 = 21銘柄
  - 重複: 短期優先→中期→長期
  - gap/intraday分離

使い方:
  pip install yfinance pandas numpy
  python backtest_smt_improved.py
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
N_SHORT = 7
N_MID = 7
N_LONG = 7
LB_SHORT = 126    # 6ヶ月
LB_MID = 252      # 12ヶ月
LB_LONG = 756     # 36ヶ月

# --- 改良パラメータ ---
TREND_SMA = 200           # トレンドフィルター: SPY vs SMA(200)
TREND_REDUCE = 0.5        # 下落局面での投資比率
VOL_WINDOW = 60           # 逆ボラ計算ウィンドウ（営業日）
VOL_EXPONENT = 1.0        # 逆ボラ加重の指数
MAX_PER_SECTOR = 3        # 1セクターあたり最大銘柄数
MONTHLY_REBAL = True      # True=毎月 / False=年4回(SMT同等)

BT_START = '2011-03-01'
BT_END = '2026-03-07'
DATA_START = '2008-01-01'

print("=" * 70)
print("  SMT US Momentum - Improved Version Backtest")
print("=" * 70)
print(f"期間: {BT_START} ~ {BT_END}")
print(f"改良点:")
print(f"  1. トレンドフィルター: SPY >= SMA{TREND_SMA} → 100% / < → {TREND_REDUCE:.0%}")
print(f"  2. 逆ボラ加重: (1/vol_{VOL_WINDOW}d)^{VOL_EXPONENT}")
print(f"  3. リバランス: {'毎月末' if MONTHLY_REBAL else '年4回(2/5/8/11月)'}")
print(f"  4. セクター分散: 1セクター最大{MAX_PER_SECTOR}銘柄")
print(flush=True)

# ============================================================
# 1. S&P500 構成銘柄取得（セクター情報付き）
# ============================================================
print("\n=== Fetching S&P500 constituents ===", flush=True)
sp500_tickers = None
ticker_sector = {}  # ticker → GICSSector

try:
    req = urllib.request.Request(
        'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
        headers={'User-Agent': 'Mozilla/5.0'}
    )
    with urllib.request.urlopen(req) as resp:
        html_content = resp.read().decode('utf-8')
    sp500_table = pd.read_html(io.StringIO(html_content))[0]
    sp500_tickers = sp500_table['Symbol'].str.replace('.', '-', regex=False).tolist()

    # セクター情報を取得
    sector_col = None
    for col in sp500_table.columns:
        if 'GICS' in str(col) and 'Sector' in str(col):
            sector_col = col
            break
    if sector_col is None:
        for col in sp500_table.columns:
            if 'sector' in str(col).lower():
                sector_col = col
                break

    if sector_col is not None:
        for _, row in sp500_table.iterrows():
            tk = str(row['Symbol']).replace('.', '-')
            ticker_sector[tk] = str(row[sector_col])
        print(f"  セクター情報: {len(set(ticker_sector.values()))} sectors", flush=True)

    print(f"  Wikipedia: {len(sp500_tickers)} stocks", flush=True)
except Exception as e:
    print(f"  Wikipedia failed: {e}", flush=True)

if sp500_tickers is None:
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
spy_close_ser = all_close.pop('SPY', None)
spy_open_ser = all_open.pop('SPY', None)

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
    spy_sma200 = spy_series.rolling(TREND_SMA).mean()
else:
    spy_series = None
    spy_sma200 = None

ret_df = close_df.pct_change(fill_method=None)
gap_ret_df = (open_df - close_df.shift(1)) / close_df.shift(1)
intra_ret_df = (close_df - open_df) / open_df.replace(0, np.nan)

# ボラティリティ（逆ボラ加重用）
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

if MONTHLY_REBAL:
    rebal_months_set = set(range(1, 13))  # 全月
else:
    rebal_months_set = {2, 5, 8, 11}

rebal_dates = []
for dt in bt_dates:
    if dt.month in rebal_months_set:
        month_dates = bt_dates[(bt_dates.month == dt.month) &
                               (bt_dates.year == dt.year)]
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

# ============================================================
# 5. 銘柄選定ロジック（改良版: セクター分散付き）
# ============================================================
def select_momentum_stocks_improved(date):
    """改良版: セクター分散キャップ付き銘柄選定"""
    idx = date_index.get_loc(date)

    ms = mom_short.iloc[idx].dropna().sort_values(ascending=False)
    mm = mom_mid.iloc[idx].dropna().sort_values(ascending=False)
    ml = mom_long.iloc[idx].dropna().sort_values(ascending=False)

    selected = []
    sector_count = {}  # sector → count

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

    # Short-term
    short_picks = []
    for ticker in ms.index:
        if len(short_picks) >= N_SHORT:
            break
        if can_add(ticker):
            short_picks.append(ticker)
            add_ticker(ticker)

    # Mid-term
    mid_picks = []
    for ticker in mm.index:
        if len(mid_picks) >= N_MID:
            break
        if can_add(ticker):
            mid_picks.append(ticker)
            add_ticker(ticker)

    # Long-term
    long_picks = []
    for ticker in ml.index:
        if len(long_picks) >= N_LONG:
            break
        if can_add(ticker):
            long_picks.append(ticker)
            add_ticker(ticker)

    return selected, short_picks, mid_picks, long_picks

# ============================================================
# 6. ウエイト計算（逆ボラティリティ加重）
# ============================================================
def compute_weights(stocks, date):
    """逆ボラ加重でウエイト計算"""
    idx = date_index.get_loc(date)
    vols = {}
    for t in stocks:
        if t in vol_df.columns:
            v = vol_df.iloc[idx][t]
            if not np.isnan(v) and v > 0:
                vols[t] = v

    if not vols:
        # フォールバック: 等ウエイト
        w = 1.0 / len(stocks) if stocks else 0
        return {t: w for t in stocks}

    # 逆ボラ加重
    inv_vols = {t: (1.0 / v) ** VOL_EXPONENT for t, v in vols.items()}

    # ボラ情報がない銘柄はmedianで補完
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
# 7. バックテスト実行（オリジナル + 改良版 同時実行）
# ============================================================
print("\n=== Running backtest (Original + Improved) ===", flush=True)

ret_np = ret_df.values
gap_ret_np = gap_ret_df.values
intra_ret_np = intra_ret_df.values
col_to_idx = {c: i for i, c in enumerate(ret_df.columns)}

def run_backtest(name, select_fn, weight_fn, use_trend_filter, rebal_list):
    """汎用バックテストエンジン"""
    portfolio_returns = []
    portfolio_dates = []
    current_holdings = {}  # ticker → weight
    next_rebal_idx = 0

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

                # トレンドフィルター適用
                if use_trend_filter and spy_sma200 is not None:
                    spy_val = spy_series.get(judge_date, np.nan)
                    sma_val = spy_sma200.get(judge_date, np.nan)
                    if not np.isnan(spy_val) and not np.isnan(sma_val):
                        if spy_val < sma_val:
                            weights = {t: w * TREND_REDUCE for t, w in weights.items()}
                            # 残りはキャッシュ（リターン0）

                current_holdings = weights
            else:
                current_holdings = {}

            # gap（旧ポジ）+ intraday（新ポジ）
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

    return pd.Series(portfolio_returns, index=portfolio_dates)

# --- オリジナルSMTのリバランス日（年4回） ---
orig_rebal_months = {2, 5, 8, 11}
orig_rebal_dates = []
for dt in bt_dates:
    if dt.month in orig_rebal_months:
        month_dates = bt_dates[(bt_dates.month == dt.month) &
                               (bt_dates.year == dt.year)]
        if len(month_dates) > 0:
            last_day = month_dates[-1]
            if last_day not in orig_rebal_dates:
                orig_rebal_dates.append(last_day)
orig_rebal_dates = sorted(set(orig_rebal_dates))
orig_rebal_exec = []
for rd in orig_rebal_dates:
    future = bt_dates[bt_dates > rd]
    if len(future) > 0:
        orig_rebal_exec.append((rd, future[0]))

# --- 等ウエイト関数 ---
def equal_weight(stocks, date):
    w = 1.0 / len(stocks) if stocks else 0
    return {t: w for t in stocks}

# --- オリジナルSMT選定関数 ---
def select_original(date):
    idx = date_index.get_loc(date)
    ms = mom_short.iloc[idx].dropna().sort_values(ascending=False)
    mm = mom_mid.iloc[idx].dropna().sort_values(ascending=False)
    ml = mom_long.iloc[idx].dropna().sort_values(ascending=False)
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

# ========== 5パターン実行 ==========
print("  [1/5] SMTオリジナル（年4回・等ウエイト・フィルターなし）...", flush=True)
ret_original = run_backtest("Original", select_original, equal_weight,
                            False, orig_rebal_exec)

print("  [2/5] +トレンドフィルター...", flush=True)
ret_trend = run_backtest("+Trend", select_original, equal_weight,
                         True, orig_rebal_exec)

print("  [3/5] +逆ボラ加重...", flush=True)
ret_invvol = run_backtest("+InvVol", select_original, compute_weights,
                          False, orig_rebal_exec)

print("  [4/5] +月次リバランス...", flush=True)
ret_monthly = run_backtest("+Monthly", select_original, equal_weight,
                           False, rebal_exec_dates)

print("  [5/5] 全部入り（トレンド+逆ボラ+月次+セクター分散）...", flush=True)
ret_all = run_backtest("AllIn", select_momentum_stocks_improved, compute_weights,
                       True, rebal_exec_dates)

# ============================================================
# 8. 成績計算・比較
# ============================================================
spy_bt = spy_series.reindex(ret_original.index).pct_change().fillna(0) if spy_series is not None else None

def calc_stats(rets, label, verbose=True):
    cum_r = (1 + rets).cumprod()
    total_r = cum_r.iloc[-1] - 1
    years = len(rets) / 252
    cagr = (1 + total_r) ** (1 / years) - 1 if years > 0 else 0
    vol = rets.std() * np.sqrt(252)
    sharpe = cagr / vol if vol > 0 else 0
    peak = cum_r.cummax()
    dd = (cum_r - peak) / peak
    mdd = dd.min()
    if verbose:
        print(f"\n  [{label}]")
        print(f"  Sharpe:     {sharpe:.2f}")
        print(f"  CAGR:       {cagr*100:+.1f}%")
        print(f"  MDD:        {mdd*100:.1f}%")
        print(f"  Vol:        {vol*100:.1f}%")
    return sharpe, cagr, mdd, vol

print("\n" + "=" * 70)
print("  Results Comparison")
print("=" * 70)

strategies = [
    ("SMT Original", ret_original),
    ("+TrendFilter", ret_trend),
    ("+InvVol Weight", ret_invvol),
    ("+Monthly Rebal", ret_monthly),
    ("All Improved", ret_all),
]

if spy_bt is not None:
    strategies.append(("SPY (BM)", spy_bt))

all_stats = {}
for name, rets in strategies:
    stats = calc_stats(rets, name)
    all_stats[name] = stats

# 比較テーブル
print("\n" + "-" * 70)
print(f"  {'Strategy':<22} {'Sharpe':>8} {'CAGR':>8} {'MDD':>8} {'Vol':>8}")
print("-" * 70)
for name in all_stats:
    sh, cagr, mdd, vol = all_stats[name]
    print(f"  {name:<22} {sh:>7.2f} {cagr*100:>+7.1f}% {mdd*100:>+7.1f}% {vol*100:>7.1f}%")
print("-" * 70)

# ============================================================
# 9. 年次リターン比較
# ============================================================
print(f"\n  [Annual Returns]")
print(f"  {'Year':>6}  {'Original':>10}  {'Improved':>10}  {'SPY':>10}  {'Orig-SPY':>10}  {'Impr-SPY':>10}")
print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

for year in range(2011, 2027):
    yr_mask_o = ret_original.index.year == year
    yr_mask_a = ret_all.index.year == year
    if yr_mask_o.sum() == 0:
        continue
    orig_yr = (1 + ret_original[yr_mask_o]).prod() - 1
    impr_yr = (1 + ret_all[yr_mask_a]).prod() - 1
    if spy_bt is not None:
        spy_yr = (1 + spy_bt[yr_mask_o]).prod() - 1
        print(f"  {year:>6}  {orig_yr*100:>+9.1f}%  {impr_yr*100:>+9.1f}%  "
              f"{spy_yr*100:>+9.1f}%  {(orig_yr-spy_yr)*100:>+9.1f}%  {(impr_yr-spy_yr)*100:>+9.1f}%")

# ============================================================
# 10. 下落局面分析
# ============================================================
print(f"\n  [Drawdown Periods]")
dd_periods = [
    ("COVID-19", "2020-02-01", "2020-03-31"),
    ("2022 Bear", "2022-01-01", "2022-10-31"),
    ("2018 Q4",  "2018-10-01", "2018-12-31"),
]

for pname, start, end in dd_periods:
    print(f"\n  {pname} ({start} ~ {end}):")
    for sname, rets in [("Original", ret_original), ("Improved", ret_all), ("SPY", spy_bt)]:
        if rets is None:
            continue
        mask = (rets.index >= pd.Timestamp(start)) & (rets.index <= pd.Timestamp(end))
        pr = rets[mask]
        if len(pr) < 2:
            continue
        cum_pr = (1 + pr).cumprod()
        period_ret = cum_pr.iloc[-1] - 1
        period_dd = ((cum_pr - cum_pr.cummax()) / cum_pr.cummax()).min()
        print(f"    {sname:<12} Return: {period_ret*100:>+7.1f}%  MaxDD: {period_dd*100:>+7.1f}%")

print("\n" + "=" * 70)
print("  Done.", flush=True)
print("=" * 70)
