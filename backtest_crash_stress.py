"""
暴落ストレステスト
━━━━━━━━━━━━━━━━
1. 通常版 (6M/12M/36M): 2011〜2026（15年）
2. 2バケット版 (6M/12M): 2009〜2026（17年）← リーマン後の回復局面を含む
3. 各暴落局面の詳細分析（日次ドローダウン推移、回復日数、最大DD）

改良版ベース（トレンドフィルター + 逆ボラ + 隔月 + セクター分散）
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

DATA_START = '2006-01-01'  # 36ヶ月LB用に余裕

print("=" * 70)
print("  Crash Stress Test")
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
        df = yf.download(' '.join(batch), start=DATA_START, end='2026-03-07',
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

# モメンタム
print("Computing momentum...", flush=True)
mom_126 = close_df.pct_change(126, fill_method=None)
mom_252 = close_df.pct_change(252, fill_method=None)
mom_756 = close_df.pct_change(756, fill_method=None)

ret_np = ret_df.values
gap_ret_np = gap_ret_df.values
intra_ret_np = intra_ret_df.values
col_to_idx = {c: i for i, c in enumerate(ret_df.columns)}

print(f"Valid tickers: {len(close_df.columns)}", flush=True)
print(f"Data range: {date_index[0].strftime('%Y-%m-%d')} ~ {date_index[-1].strftime('%Y-%m-%d')}", flush=True)

# ============================================================
# バックテストエンジン
# ============================================================
def make_rebal_exec(bt_start, bt_end):
    bt_dates_local = date_index[(date_index >= bt_start) & (date_index <= bt_end)]
    rebal_dates = []
    for dt in bt_dates_local:
        if dt.month % 2 == 0:
            month_dates = bt_dates_local[(bt_dates_local.month == dt.month) &
                                          (bt_dates_local.year == dt.year)]
            if len(month_dates) > 0 and month_dates[-1] not in rebal_dates:
                rebal_dates.append(month_dates[-1])
    rebal_dates = sorted(set(rebal_dates))
    rebal_exec = []
    for rd in rebal_dates:
        future = bt_dates_local[bt_dates_local > rd]
        if len(future) > 0:
            rebal_exec.append((rd, future[0]))
    return bt_dates_local, rebal_exec

def run_backtest(bt_start, bt_end, mom_dfs, ns, use_trend=True):
    """
    mom_dfs: list of (mom_df, n_picks) tuples
    """
    bt_dates_local, rebal_exec = make_rebal_exec(bt_start, bt_end)

    current_holdings = {}
    next_rebal_idx = 0
    portfolio_returns = []

    for i, date in enumerate(bt_dates_local):
        date_idx = date_index.get_loc(date)

        is_rebalance = False
        if next_rebal_idx < len(rebal_exec):
            judge_date, exec_date = rebal_exec[next_rebal_idx]
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

            for mom_df, n in mom_dfs:
                scores = mom_df.iloc[jidx].dropna().sort_values(ascending=False)
                count = 0
                for tk in scores.index:
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

    rets = pd.Series(portfolio_returns, index=bt_dates_local[:len(portfolio_returns)])
    return rets

def full_stats(rets, label):
    cum = (1 + rets).cumprod()
    total = cum.iloc[-1] - 1
    years = len(rets) / 252
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 else 0
    vol = rets.std() * np.sqrt(252)
    sharpe = cagr / vol if vol > 0 else 0
    peak = cum.cummax()
    dd = (cum - peak) / peak
    mdd = dd.min()
    # DD期間
    dd_duration = 0
    max_dd_duration = 0
    for v in dd.values:
        if v < 0:
            dd_duration += 1
            max_dd_duration = max(max_dd_duration, dd_duration)
        else:
            dd_duration = 0
    return {
        'label': label, 'sharpe': sharpe, 'cagr': cagr, 'mdd': mdd,
        'vol': vol, 'cum': cum, 'dd': dd, 'max_dd_duration': max_dd_duration,
        'rets': rets
    }

# ============================================================
# 4つの戦略を実行
# ============================================================
print("\n=== Running strategies ===", flush=True)

# A. 3バケット改良版 (2011~2026, 15年)
print("  [A] 3-bucket improved (2011-2026)...", flush=True)
rets_3b = run_backtest('2011-03-01', '2026-03-07',
                        [(mom_126, 7), (mom_252, 7), (mom_756, 7)], 21, use_trend=True)

# B. 3バケット トレンドフィルターなし (2011~2026)
print("  [B] 3-bucket NO trend filter (2011-2026)...", flush=True)
rets_3b_notf = run_backtest('2011-03-01', '2026-03-07',
                             [(mom_126, 7), (mom_252, 7), (mom_756, 7)], 21, use_trend=False)

# C. 2バケット改良版 (2009~2026, 17年) ← リーマン後を含む
print("  [C] 2-bucket improved (2009-2026)...", flush=True)
rets_2b = run_backtest('2009-06-01', '2026-03-07',
                        [(mom_126, 10), (mom_252, 11)], 21, use_trend=True)

# D. 2バケット トレンドフィルターなし (2009~2026)
print("  [D] 2-bucket NO trend filter (2009-2026)...", flush=True)
rets_2b_notf = run_backtest('2009-06-01', '2026-03-07',
                             [(mom_126, 10), (mom_252, 11)], 21, use_trend=False)

# SPY
spy_3b = spy_series.reindex(rets_3b.index).pct_change().fillna(0)
spy_2b = spy_series.reindex(rets_2b.index).pct_change().fillna(0)

stats = {
    '3B Improved': full_stats(rets_3b, '3B Improved'),
    '3B No TF': full_stats(rets_3b_notf, '3B No TF'),
    '2B Improved': full_stats(rets_2b, '2B Improved'),
    '2B No TF': full_stats(rets_2b_notf, '2B No TF'),
    'SPY (15yr)': full_stats(spy_3b, 'SPY (15yr)'),
    'SPY (17yr)': full_stats(spy_2b, 'SPY (17yr)'),
}

# ============================================================
# メイン成績
# ============================================================
print("\n" + "=" * 90)
print("  MAIN RESULTS")
print("=" * 90)
print(f"  {'Strategy':<18} {'Period':>14} {'Sharpe':>7} {'CAGR':>7} {'MDD':>8} {'Vol':>6} {'MaxDD日数':>10}")
print("-" * 90)
for name in ['3B Improved', '3B No TF', 'SPY (15yr)', '2B Improved', '2B No TF', 'SPY (17yr)']:
    s = stats[name]
    start = s['rets'].index[0].strftime('%Y/%m')
    end = s['rets'].index[-1].strftime('%Y/%m')
    years = len(s['rets']) / 252
    print(f"  {name:<18} {start}-{end} {s['sharpe']:>6.2f} {s['cagr']*100:>+6.1f}% "
          f"{s['mdd']*100:>+7.1f}% {s['vol']*100:>5.1f}% {s['max_dd_duration']:>8}d")
print("-" * 90)

# ============================================================
# 暴落局面 詳細分析
# ============================================================
crash_periods = [
    ("2011 US Debt Downgrade",   "2011-07-01", "2011-10-31", "S&P格下げ、欧州債務危機"),
    ("2015 China Shock",         "2015-08-01", "2015-09-30", "中国景気減速、人民元切り下げ"),
    ("2018 Q4 Fed Tightening",   "2018-10-01", "2018-12-31", "FRB利上げ + 貿易摩擦"),
    ("COVID-19 Crash",           "2020-02-19", "2020-03-23", "パンデミック急落（ピーク→底）"),
    ("COVID-19 Extended",        "2020-01-01", "2020-06-30", "パンデミック（含む回復）"),
    ("2022 Rate Hike Bear",      "2022-01-03", "2022-10-12", "FRB急速利上げ（ピーク→底）"),
    ("2022 Full Year",           "2022-01-01", "2022-12-31", "2022年通年"),
    ("2025 Tariff Shock",        "2025-02-01", "2026-03-07", "トランプ関税ショック"),
]

print("\n" + "=" * 90)
print("  CRASH PERIOD ANALYSIS")
print("=" * 90)

for crash_name, start, end, desc in crash_periods:
    print(f"\n  {crash_name}: {start} ~ {end}")
    print(f"  ({desc})")
    print(f"  {'Strategy':<18} {'Return':>9} {'MaxDD':>9} {'Best Day':>10} {'Worst Day':>10}")
    print(f"  {'-'*18} {'-'*9} {'-'*9} {'-'*10} {'-'*10}")

    for name in stats:
        s = stats[name]
        rets = s['rets']
        mask = (rets.index >= pd.Timestamp(start)) & (rets.index <= pd.Timestamp(end))
        pr = rets[mask]
        if len(pr) < 2:
            continue
        cum_pr = (1 + pr).cumprod()
        pret = cum_pr.iloc[-1] - 1
        pmdd = ((cum_pr - cum_pr.cummax()) / cum_pr.cummax()).min()
        best_day = pr.max()
        worst_day = pr.min()
        print(f"  {name:<18} {pret*100:>+8.1f}% {pmdd*100:>+8.1f}% "
              f"{best_day*100:>+9.1f}% {worst_day*100:>+9.1f}%")

# ============================================================
# ドローダウン一覧（深さ順）
# ============================================================
print("\n" + "=" * 90)
print("  ALL DRAWDOWNS > -10% (3B Improved)")
print("=" * 90)

dd_3b = stats['3B Improved']['dd']
cum_3b = stats['3B Improved']['cum']

# DDイベントを検出
in_dd = False
dd_start = None
dd_events = []
peak_val = 0

for i, (date, dd_val) in enumerate(dd_3b.items()):
    cum_val = cum_3b.iloc[i]
    if dd_val == 0:
        if in_dd and dd_start is not None:
            dd_events.append({
                'start': dd_start,
                'end': date,
                'trough': trough_date,
                'max_dd': max_dd_val,
                'duration_to_trough': (trough_date - dd_start).days,
                'duration_to_recovery': (date - dd_start).days,
            })
        in_dd = False
        dd_start = None
        peak_val = cum_val
    else:
        if not in_dd:
            in_dd = True
            dd_start = date
            max_dd_val = dd_val
            trough_date = date
        else:
            if dd_val < max_dd_val:
                max_dd_val = dd_val
                trough_date = date

# まだDD中
if in_dd and dd_start is not None:
    dd_events.append({
        'start': dd_start,
        'end': dd_3b.index[-1],
        'trough': trough_date,
        'max_dd': max_dd_val,
        'duration_to_trough': (trough_date - dd_start).days,
        'duration_to_recovery': None,  # 未回復
    })

# -10%以上のDDのみ表示、深さ順
sig_dd = [d for d in dd_events if d['max_dd'] <= -0.10]
sig_dd.sort(key=lambda x: x['max_dd'])

print(f"  {'#':>3}  {'Start':>12}  {'Trough':>12}  {'Recovery':>12}  "
      f"{'MaxDD':>8}  {'To Trough':>10}  {'To Recovery':>12}")
print("-" * 90)

for i, d in enumerate(sig_dd):
    recov = d['end'].strftime('%Y-%m-%d') if d['duration_to_recovery'] is not None else "未回復"
    recov_days = f"{d['duration_to_recovery']}d" if d['duration_to_recovery'] is not None else "進行中"
    print(f"  {i+1:>3}  {d['start'].strftime('%Y-%m-%d'):>12}  "
          f"{d['trough'].strftime('%Y-%m-%d'):>12}  {recov:>12}  "
          f"{d['max_dd']*100:>+7.1f}%  {d['duration_to_trough']:>8}d  {recov_days:>12}")

# ============================================================
# 2バケット版のDD一覧（2009〜含む）
# ============================================================
print(f"\n  ALL DRAWDOWNS > -10% (2B Improved, from 2009)")

dd_2b = stats['2B Improved']['dd']
cum_2b = stats['2B Improved']['cum']

in_dd = False
dd_start = None
dd_events_2b = []

for i, (date, dd_val) in enumerate(dd_2b.items()):
    cum_val = cum_2b.iloc[i]
    if dd_val == 0:
        if in_dd and dd_start is not None:
            dd_events_2b.append({
                'start': dd_start, 'end': date, 'trough': trough_date,
                'max_dd': max_dd_val,
                'duration_to_trough': (trough_date - dd_start).days,
                'duration_to_recovery': (date - dd_start).days,
            })
        in_dd = False
        dd_start = None
    else:
        if not in_dd:
            in_dd = True
            dd_start = date
            max_dd_val = dd_val
            trough_date = date
        else:
            if dd_val < max_dd_val:
                max_dd_val = dd_val
                trough_date = date

if in_dd and dd_start is not None:
    dd_events_2b.append({
        'start': dd_start, 'end': dd_2b.index[-1], 'trough': trough_date,
        'max_dd': max_dd_val,
        'duration_to_trough': (trough_date - dd_start).days,
        'duration_to_recovery': None,
    })

sig_dd_2b = [d for d in dd_events_2b if d['max_dd'] <= -0.10]
sig_dd_2b.sort(key=lambda x: x['max_dd'])

print(f"  {'#':>3}  {'Start':>12}  {'Trough':>12}  {'Recovery':>12}  "
      f"{'MaxDD':>8}  {'To Trough':>10}  {'To Recovery':>12}")
print("-" * 90)

for i, d in enumerate(sig_dd_2b):
    recov = d['end'].strftime('%Y-%m-%d') if d['duration_to_recovery'] is not None else "未回復"
    recov_days = f"{d['duration_to_recovery']}d" if d['duration_to_recovery'] is not None else "進行中"
    print(f"  {i+1:>3}  {d['start'].strftime('%Y-%m-%d'):>12}  "
          f"{d['trough'].strftime('%Y-%m-%d'):>12}  {recov:>12}  "
          f"{d['max_dd']*100:>+7.1f}%  {d['duration_to_trough']:>8}d  {recov_days:>12}")

# ============================================================
# トレンドフィルターの暴落防御効果
# ============================================================
print("\n" + "=" * 90)
print("  TREND FILTER EFFECT DURING CRASHES")
print("=" * 90)
print(f"  {'Crash':<28} {'With TF':>10} {'No TF':>10} {'Diff':>10}  {'TF効果'}")
print("-" * 90)

for crash_name, start, end, desc in crash_periods:
    # 3バケット版で比較
    mask_tf = (rets_3b.index >= pd.Timestamp(start)) & (rets_3b.index <= pd.Timestamp(end))
    mask_notf = (rets_3b_notf.index >= pd.Timestamp(start)) & (rets_3b_notf.index <= pd.Timestamp(end))

    pr_tf = rets_3b[mask_tf]
    pr_notf = rets_3b_notf[mask_notf]

    if len(pr_tf) < 2 or len(pr_notf) < 2:
        continue

    ret_tf = (1 + pr_tf).prod() - 1
    ret_notf = (1 + pr_notf).prod() - 1
    diff = ret_tf - ret_notf

    effect = "防御効果あり" if diff > 0.01 else ("逆効果" if diff < -0.01 else "ほぼ同じ")
    print(f"  {crash_name:<28} {ret_tf*100:>+9.1f}% {ret_notf*100:>+9.1f}% "
          f"{diff*100:>+9.1f}%  {effect}")

# ============================================================
# 年次リターン（2バケット版 2009〜）
# ============================================================
print("\n" + "=" * 90)
print("  ANNUAL RETURNS (2B Improved, from 2009)")
print("=" * 90)
print(f"  {'Year':>6}  {'2B Impr':>10}  {'2B NoTF':>10}  {'SPY':>10}  {'Excess':>10}")
print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

for year in range(2009, 2027):
    yr_mask = rets_2b.index.year == year
    yr_mask_notf = rets_2b_notf.index.year == year
    yr_mask_spy = spy_2b.index.year == year
    if yr_mask.sum() == 0:
        continue
    r_2b = (1 + rets_2b[yr_mask]).prod() - 1
    r_notf = (1 + rets_2b_notf[yr_mask_notf]).prod() - 1
    r_spy = (1 + spy_2b[yr_mask_spy]).prod() - 1
    excess = r_2b - r_spy
    print(f"  {year:>6}  {r_2b*100:>+9.1f}%  {r_notf*100:>+9.1f}%  "
          f"{r_spy*100:>+9.1f}%  {excess*100:>+9.1f}%")

print("\n" + "=" * 90)
print("  Done.", flush=True)
print("=" * 90)
