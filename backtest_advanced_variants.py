"""
SMT米国モメンタム 未検証バリアント一括テスト
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5つの未検証アイデアを一括で比較:
  1. 直近1ヶ月除外（1-month reversal skip）
  2. 銘柄数変更（15, 21, 30）
  3. バケット傾斜（短期重視 / 長期重視）
  4. リスク調整モメンタム（リターン÷ボラ）
  5. 単一期間モメンタム（12ヶ月1本で21銘柄）

ベース: 改良版（トレンドフィルター + 逆ボラ加重 + 隔月リバランス + セクター分散）
"""
import gc
import io
import urllib.request
import yfinance as yf
import pandas as pd
import numpy as np
import sys

sys.stdout.reconfigure(encoding='utf-8')

# 固定パラメータ
TREND_SMA = 200
TREND_REDUCE = 0.5
VOL_WINDOW = 60
MAX_PER_SECTOR = 3
LB_SHORT = 126
LB_MID = 252
LB_LONG = 756
SKIP_PERIOD = 21  # 直近1ヶ月（リバーサルスキップ用）

BT_START = '2011-03-01'
BT_END = '2026-03-07'
DATA_START = '2008-01-01'

print("=" * 70)
print("  Advanced Variants Analysis")
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

# モメンタム事前計算
print("Computing momentum variants...", flush=True)

# 通常モメンタム
mom_short = close_df.pct_change(LB_SHORT, fill_method=None)
mom_mid = close_df.pct_change(LB_MID, fill_method=None)
mom_long = close_df.pct_change(LB_LONG, fill_method=None)

# 直近1ヶ月除外モメンタム: (t-LB ~ t-SKIP) のリターン
# = (price[t-SKIP] / price[t-LB]) - 1
close_skip = close_df.shift(SKIP_PERIOD)
mom_short_skip = (close_skip / close_df.shift(LB_SHORT) - 1)
mom_mid_skip = (close_skip / close_df.shift(LB_MID) - 1)
mom_long_skip = (close_skip / close_df.shift(LB_LONG) - 1)

# リスク調整モメンタム: リターン / ボラ
vol_for_adj = ret_df.rolling(VOL_WINDOW).std()
mom_short_radj = mom_short / vol_for_adj.replace(0, np.nan)
mom_mid_radj = mom_mid / vol_for_adj.replace(0, np.nan)
mom_long_radj = mom_long / vol_for_adj.replace(0, np.nan)

# 直近1ヶ月除外 + リスク調整
mom_short_skip_radj = mom_short_skip / vol_for_adj.replace(0, np.nan)
mom_mid_skip_radj = mom_mid_skip / vol_for_adj.replace(0, np.nan)
mom_long_skip_radj = mom_long_skip / vol_for_adj.replace(0, np.nan)

# 単一期間（12ヶ月）
mom_12m = close_df.pct_change(LB_MID, fill_method=None)
mom_12m_skip = (close_skip / close_df.shift(LB_MID) - 1)

print(f"Valid tickers: {len(close_df.columns)}", flush=True)

# リバランス日（隔月 = 偶数月末）
bt_dates = date_index[(date_index >= BT_START) & (date_index <= BT_END)]
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

print(f"Rebalance count: {len(rebal_exec_dates)} (bimonthly)", flush=True)

# numpy
ret_np = ret_df.values
gap_ret_np = gap_ret_df.values
intra_ret_np = intra_ret_df.values
col_to_idx = {c: i for i, c in enumerate(ret_df.columns)}

# ============================================================
# 汎用バックテストエンジン
# ============================================================
def run_backtest(mom_s_df, mom_m_df, mom_l_df, n_short, n_mid, n_long,
                 single_period_df=None):
    """
    mom_s/m/l_df: 各バケットのスコアDataFrame
    n_short/mid/long: 各バケットの銘柄数
    single_period_df: 単一期間モードの場合、このDFから全銘柄を選ぶ
    """
    total_stocks = n_short + n_mid + n_long if single_period_df is None else 21
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
            jidx = date_index.get_loc(judge_date)

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

            if single_period_df is not None:
                # 単一期間モード: 1つのDFから全銘柄を選ぶ
                scores = single_period_df.iloc[jidx].dropna().sort_values(ascending=False)
                count = 0
                for tk in scores.index:
                    if count >= total_stocks: break
                    if can_add(tk):
                        add_tk(tk)
                        count += 1
            else:
                # 3バケットモード
                ms = mom_s_df.iloc[jidx].dropna().sort_values(ascending=False)
                mm = mom_m_df.iloc[jidx].dropna().sort_values(ascending=False)
                ml = mom_l_df.iloc[jidx].dropna().sort_values(ascending=False)

                for bucket_df, n in [(ms, n_short), (mm, n_mid), (ml, n_long)]:
                    count = 0
                    for tk in bucket_df.index:
                        if count >= n: break
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
                total_w = sum(inv_vols.values())
                weights = {t: inv_vols[t] / total_w for t in selected}
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

    return np.array(portfolio_returns)

def calc_stats(rets_arr):
    cum = np.cumprod(1 + rets_arr)
    total = cum[-1] - 1
    years = len(rets_arr) / 252
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 else 0
    vol = np.std(rets_arr) * np.sqrt(252)
    sharpe = cagr / vol if vol > 0 else 0
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    mdd = dd.min()
    return sharpe, cagr, mdd, vol

# ============================================================
# バリアント定義
# ============================================================
variants = []

# --- ベースライン（改良版: 通常モメンタム, 7/7/7, 隔月） ---
variants.append(("Baseline (7/7/7)",
    lambda: run_backtest(mom_short, mom_mid, mom_long, 7, 7, 7)))

# --- 1. 直近1ヶ月除外 ---
variants.append(("1M Skip",
    lambda: run_backtest(mom_short_skip, mom_mid_skip, mom_long_skip, 7, 7, 7)))

# --- 2. 銘柄数変更 ---
variants.append(("Fewer: 5/5/5=15",
    lambda: run_backtest(mom_short, mom_mid, mom_long, 5, 5, 5)))
variants.append(("More: 10/10/10=30",
    lambda: run_backtest(mom_short, mom_mid, mom_long, 10, 10, 10)))

# --- 3. バケット傾斜 ---
variants.append(("Short-heavy: 10/7/4",
    lambda: run_backtest(mom_short, mom_mid, mom_long, 10, 7, 4)))
variants.append(("Long-heavy: 4/7/10",
    lambda: run_backtest(mom_short, mom_mid, mom_long, 4, 7, 10)))

# --- 4. リスク調整モメンタム ---
variants.append(("Risk-adj Momentum",
    lambda: run_backtest(mom_short_radj, mom_mid_radj, mom_long_radj, 7, 7, 7)))

# --- 5. 単一期間（12ヶ月） ---
variants.append(("Single 12M Top21",
    lambda: run_backtest(None, None, None, 0, 0, 0, single_period_df=mom_12m)))

# --- 組み合わせ: 1M Skip + リスク調整 ---
variants.append(("1M Skip + Risk-adj",
    lambda: run_backtest(mom_short_skip_radj, mom_mid_skip_radj, mom_long_skip_radj, 7, 7, 7)))

# --- 組み合わせ: 単一12ヶ月 + 1ヶ月除外 ---
variants.append(("Single 12M + 1M Skip",
    lambda: run_backtest(None, None, None, 0, 0, 0, single_period_df=mom_12m_skip)))

# ============================================================
# 実行
# ============================================================
print(f"\n=== Running {len(variants)} variants ===", flush=True)

results = []
for i, (name, run_fn) in enumerate(variants):
    print(f"  [{i+1:>2}/{len(variants)}] {name}...", flush=True)
    rets = run_fn()
    sharpe, cagr, mdd, vol = calc_stats(rets)
    results.append({
        'name': name, 'sharpe': sharpe, 'cagr': cagr, 'mdd': mdd, 'vol': vol,
        'rets': rets
    })
    print(f"         Sharpe={sharpe:.2f}  CAGR={cagr*100:+.1f}%  MDD={mdd*100:.1f}%", flush=True)

# SPY
spy_bt = spy_series.reindex(bt_dates).pct_change().fillna(0).values
spy_sharpe, spy_cagr, spy_mdd, spy_vol = calc_stats(spy_bt)

# ============================================================
# 結果出力
# ============================================================
# Sharpe順ソート
results_sorted = sorted(results, key=lambda x: x['sharpe'], reverse=True)

print("\n" + "=" * 85)
print("  RESULTS RANKED BY SHARPE")
print("=" * 85)
print(f"  {'Rank':>4}  {'Variant':<28} {'Sharpe':>7} {'CAGR':>8} {'MDD':>8} {'Vol':>7}  Note")
print("-" * 85)

for i, r in enumerate(results_sorted):
    note = ""
    if r['name'] == "Baseline (7/7/7)":
        note = "<< BASE"
    elif i == 0:
        note = "<< BEST"
    print(f"  {i+1:>4}  {r['name']:<28} {r['sharpe']:>6.2f} "
          f"{r['cagr']*100:>+7.1f}% {r['mdd']*100:>+7.1f}% {r['vol']*100:>6.1f}%  {note}")

print(f"  {'':>4}  {'SPY (BM)':<28} {spy_sharpe:>6.2f} "
      f"{spy_cagr*100:>+7.1f}% {spy_mdd*100:>+7.1f}% {spy_vol*100:>6.1f}%")
print("-" * 85)

# ベースラインとの差分
baseline = next(r for r in results if r['name'] == "Baseline (7/7/7)")
print(f"\n  [Difference from Baseline (Sharpe {baseline['sharpe']:.2f})]")
print(f"  {'Variant':<28} {'dSharpe':>8} {'dCAGR':>8} {'dMDD':>8}")
print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*8}")
for r in results_sorted:
    if r['name'] == "Baseline (7/7/7)":
        continue
    ds = r['sharpe'] - baseline['sharpe']
    dc = (r['cagr'] - baseline['cagr']) * 100
    dm = (r['mdd'] - baseline['mdd']) * 100
    sign_s = "+" if ds >= 0 else ""
    sign_c = "+" if dc >= 0 else ""
    sign_m = "+" if dm >= 0 else ""  # MDDは負なので+は改善
    print(f"  {r['name']:<28} {sign_s}{ds:>7.2f} {sign_c}{dc:>7.1f}% {sign_m}{dm:>7.1f}%")

# ============================================================
# カテゴリ別分析
# ============================================================
print(f"\n" + "=" * 85)
print("  CATEGORY ANALYSIS")
print("=" * 85)

categories = {
    "1. 1M Skip効果": ["Baseline (7/7/7)", "1M Skip"],
    "2. 銘柄数": ["Fewer: 5/5/5=15", "Baseline (7/7/7)", "More: 10/10/10=30"],
    "3. バケット傾斜": ["Short-heavy: 10/7/4", "Baseline (7/7/7)", "Long-heavy: 4/7/10"],
    "4. スコア方式": ["Baseline (7/7/7)", "Risk-adj Momentum", "1M Skip + Risk-adj"],
    "5. 単一 vs 3バケット": ["Single 12M Top21", "Single 12M + 1M Skip", "Baseline (7/7/7)"],
}

results_dict = {r['name']: r for r in results}
for cat_name, variant_names in categories.items():
    print(f"\n  [{cat_name}]")
    print(f"  {'Variant':<28} {'Sharpe':>7} {'CAGR':>8} {'MDD':>8} {'Vol':>7}")
    print(f"  {'-'*28} {'-'*7} {'-'*8} {'-'*8} {'-'*7}")
    for vn in variant_names:
        if vn in results_dict:
            r = results_dict[vn]
            print(f"  {r['name']:<28} {r['sharpe']:>6.2f} "
                  f"{r['cagr']*100:>+7.1f}% {r['mdd']*100:>+7.1f}% {r['vol']*100:>6.1f}%")

# ============================================================
# 年次リターン（ベースライン vs ベスト vs SPY）
# ============================================================
best = results_sorted[0]
base_rets = pd.Series(baseline['rets'], index=bt_dates[:len(baseline['rets'])])
best_rets = pd.Series(best['rets'], index=bt_dates[:len(best['rets'])])
spy_rets_s = pd.Series(spy_bt, index=bt_dates[:len(spy_bt)])

print(f"\n  [Annual Returns: Baseline vs Best({best['name']}) vs SPY]")
print(f"  {'Year':>6}  {'Baseline':>10}  {'Best':>10}  {'SPY':>10}")
print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}")
for year in range(2011, 2027):
    bm = base_rets.index.year == year
    btm = best_rets.index.year == year
    sm = spy_rets_s.index.year == year
    if bm.sum() == 0: continue
    br = (1 + base_rets[bm]).prod() - 1
    btr = (1 + best_rets[btm]).prod() - 1
    sr = (1 + spy_rets_s[sm]).prod() - 1
    print(f"  {year:>6}  {br*100:>+9.1f}%  {btr*100:>+9.1f}%  {sr*100:>+9.1f}%")

# 下落局面
print(f"\n  [Drawdown: Baseline vs Best({best['name']})]")
for pname, start, end in [("COVID-19", "2020-02-01", "2020-03-31"),
                           ("2022 Bear", "2022-01-01", "2022-10-31"),
                           ("2018 Q4", "2018-10-01", "2018-12-31")]:
    print(f"\n  {pname}:")
    for label, rets_s in [("Baseline", base_rets), (best['name'], best_rets), ("SPY", spy_rets_s)]:
        mask = (rets_s.index >= pd.Timestamp(start)) & (rets_s.index <= pd.Timestamp(end))
        pr = rets_s[mask]
        if len(pr) < 2: continue
        cum_pr = (1 + pr).cumprod()
        pret = cum_pr.iloc[-1] - 1
        pmdd = ((cum_pr - cum_pr.cummax()) / cum_pr.cummax()).min()
        print(f"    {label:<28} Return: {pret*100:>+7.1f}%  MDD: {pmdd*100:>+7.1f}%")

print("\n" + "=" * 85)
print("  Done.", flush=True)
print("=" * 85)
