"""
真のヒストリカル・ユニバース・バックテスト
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WikipediaのS&P500変更履歴（追加・除外）を使って、
各時点の「実際のS&P500構成銘柄」を復元してバックテスト。

前版(backtest_historical_universe.py)との違い:
  前版: 現在のS&P500メンバーの追加日のみ使用 → 除外された銘柄が含まれない
  本版: 追加+除外の両方を追跡 → 除外された銘柄も在籍期間中は投資対象

比較:
  A) Naive: 現在の構成銘柄で全期間計算（従来のバックテスト）
  B) Historical v1: 追加日のみ考慮（前版）
  C) True Historical: 追加+除外を完全追跡（本版）
"""
import gc
import io
import urllib.request
import yfinance as yf
import pandas as pd
import numpy as np
import sys
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
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
print("  True Historical Universe Backtest")
print("  (S&P500 additions + removals fully tracked)")
print("=" * 70, flush=True)

# ============================================================
# 1. Wikipedia からS&P500の現在メンバー + 変更履歴を取得
# ============================================================
print("\n=== Fetching S&P500 data from Wikipedia ===", flush=True)

req = urllib.request.Request(
    'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
)
with urllib.request.urlopen(req) as resp:
    html_content = resp.read().decode('utf-8')

tables = pd.read_html(io.StringIO(html_content))

# --- テーブル1: 現在の構成銘柄 ---
current_table = tables[0]
current_tickers = current_table['Symbol'].str.replace('.', '-', regex=False).tolist()

# セクター情報
ticker_sector = {}
for col in current_table.columns:
    if 'GICS' in str(col) and 'Sector' in str(col):
        for _, row in current_table.iterrows():
            tk = str(row['Symbol']).replace('.', '-')
            ticker_sector[tk] = str(row[col])
        break

# Date added (現在メンバーの追加日)
ticker_date_added = {}
date_col = None
for col in current_table.columns:
    if 'date' in str(col).lower() and 'added' in str(col).lower():
        date_col = col
        break
if date_col:
    for _, row in current_table.iterrows():
        tk = str(row['Symbol']).replace('.', '-')
        try:
            dt = pd.to_datetime(str(row[date_col]).strip())
            ticker_date_added[tk] = dt
        except:
            pass

print(f"  Current S&P500: {len(current_tickers)} stocks", flush=True)

# --- テーブル2: 変更履歴 ---
changes_table = tables[1]
print(f"  Changes table: {len(changes_table)} rows", flush=True)

# カラム名のクリーンアップ（マルチレベルヘッダー対応）
cols = changes_table.columns
if hasattr(cols, 'get_level_values'):
    # MultiIndexの場合、フラット化
    flat_cols = []
    for c in cols:
        if isinstance(c, tuple):
            flat_cols.append(' '.join(str(x) for x in c if str(x) != 'nan' and 'Unnamed' not in str(x)).strip())
        else:
            flat_cols.append(str(c))
    changes_table.columns = flat_cols

print(f"  Columns: {list(changes_table.columns)}", flush=True)

# 変更イベントをパース
changes = []  # list of (date, added_ticker, removed_ticker)

# カラム名を特定
date_col_name = None
added_col_name = None
removed_col_name = None

for col in changes_table.columns:
    cl = str(col).lower()
    if 'date' in cl:
        date_col_name = col
    elif 'added' in cl and 'ticker' in cl:
        added_col_name = col
    elif 'removed' in cl and 'ticker' in cl:
        removed_col_name = col
    elif 'added' in cl and 'security' not in cl and added_col_name is None:
        added_col_name = col
    elif 'removed' in cl and 'security' not in cl and removed_col_name is None:
        removed_col_name = col

# フォールバック: インデックスベース
if date_col_name is None:
    date_col_name = changes_table.columns[0]
if added_col_name is None:
    added_col_name = changes_table.columns[1]
if removed_col_name is None:
    removed_col_name = changes_table.columns[3]

print(f"  Using columns: date='{date_col_name}', added='{added_col_name}', removed='{removed_col_name}'")

n_parsed = 0
n_failed = 0
all_removed_tickers = set()  # 過去に除外された全ティッカー

for _, row in changes_table.iterrows():
    try:
        date_str = str(row[date_col_name]).strip()
        if not date_str or date_str == 'nan':
            continue
        dt = pd.to_datetime(date_str)

        added = str(row[added_col_name]).strip()
        removed = str(row[removed_col_name]).strip()

        added = added.replace('.', '-') if added and added != 'nan' else None
        removed = removed.replace('.', '-') if removed and removed != 'nan' else None

        if added or removed:
            changes.append((dt, added, removed))
            if removed:
                all_removed_tickers.add(removed)
            n_parsed += 1
    except Exception as e:
        n_failed += 1

changes.sort(key=lambda x: x[0])  # 日付順にソート

print(f"  Parsed {n_parsed} change events ({n_failed} failed)")
print(f"  Removed tickers (ever): {len(all_removed_tickers)}")

# 現在のメンバーでもある除外ティッカー（再追加されたもの）を除く
truly_removed = all_removed_tickers - set(current_tickers)
print(f"  Truly removed (not in current S&P500): {len(truly_removed)}")

# ============================================================
# 2. 各日付のS&P500メンバーシップを復元
# ============================================================
print("\n=== Reconstructing historical membership ===", flush=True)

# 現在のメンバーから開始し、変更を逆追いして各時点のメンバーを復元
# ただし、実装は「前から順に追跡」の方が簡単

# アプローチ:
# 1. 最も古い変更より前の時点での「初期メンバー」を推定
#    → 現在メンバー + 全除外ティッカー - 全追加ティッカー（変更テーブルで追加されたもの）
# 2. 各変更イベントを時系列順に適用

# 変更テーブル内の追加ティッカー
added_in_changes = set()
for dt, added, removed in changes:
    if added:
        added_in_changes.add(added)

# 初期メンバーの推定（最初の変更イベント前に既にS&P500にいた銘柄）
# = 現在のメンバー - 変更テーブルで追加された銘柄 + 変更テーブルで除外された銘柄のうち追加されていないもの
initial_members = set(current_tickers)
for dt, added, removed in changes:
    if added and added in initial_members:
        # この銘柄は変更テーブルで追加されたので、初期メンバーではない
        # ただし、追加前に除外されて再追加された場合もあるので注意
        pass
    if removed and removed not in initial_members:
        # この銘柄は除外されて現在はメンバーではない
        # → 変更前はメンバーだったはず
        initial_members.add(removed)

# 最も古い変更前には変更で追加された銘柄は除く
# （ただし、追加日が変更テーブルの最古日より前の場合は含める）
earliest_change = changes[0][0] if changes else pd.Timestamp('2020-01-01')

# より正確な方法: 変更テーブルを逆追い
# 現在のメンバーから開始 → 変更を新しい順に逆適用
membership_at_date = {}  # date → set of tickers

# 変更を新しい順にソート
changes_desc = sorted(changes, key=lambda x: x[0], reverse=True)

# 現在のメンバーシップ
current_set = set(current_tickers)

# 各変更の「直前」の状態を記録
# 変更(date, added, removed)の逆適用: addedを除去、removedを追加
timeline = [(pd.Timestamp('2026-12-31'), set(current_set))]  # 現在

for dt, added, removed in changes_desc:
    if added and added in current_set:
        current_set.remove(added)
    if removed:
        current_set.add(removed)
    timeline.append((dt, set(current_set)))

timeline.sort(key=lambda x: x[0])

print(f"  Timeline entries: {len(timeline)}")
print(f"  Earliest: {timeline[0][0].strftime('%Y-%m-%d')} ({len(timeline[0][1])} members)")
print(f"  Latest:   {timeline[-1][0].strftime('%Y-%m-%d')} ({len(timeline[-1][1])} members)")

# BT期間の各年の概算メンバー数
for year in [2011, 2013, 2015, 2017, 2019, 2021, 2023, 2025]:
    target = pd.Timestamp(f'{year}-06-01')
    # targetに最も近い（以前の）タイムラインエントリを使用
    members = None
    for dt, s in timeline:
        if dt <= target:
            members = s
        else:
            break
    if members:
        print(f"    {year}: ~{len(members)} members")

# 全ティッカー（ダウンロード対象）
all_tickers_ever = set()
for dt, s in timeline:
    all_tickers_ever.update(s)
print(f"\n  Total unique tickers ever in S&P500: {len(all_tickers_ever)}")

# ============================================================
# 3. 株価ダウンロード（全ティッカー）
# ============================================================
print("\n=== Downloading price data ===", flush=True)
all_dl_tickers = sorted(all_tickers_ever | {'SPY'})
print(f"  Downloading {len(all_dl_tickers)} tickers...")

batch_size = 50
all_close = {}
all_open = {}
failed_tickers = []

for i in range(0, len(all_dl_tickers), batch_size):
    batch = all_dl_tickers[i:i+batch_size]
    try:
        df = yf.download(' '.join(batch), start=DATA_START, end=BT_END,
                         progress=False, threads=True)
        if df.empty:
            failed_tickers.extend(batch)
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
            # 取得できなかった銘柄を記録
            downloaded = set(str(col) for col in close.columns if not close[col].dropna().empty)
            for t in batch:
                if t not in downloaded and t != 'SPY':
                    failed_tickers.append(t)
        n_batches = (len(all_dl_tickers) + batch_size - 1) // batch_size
        print(f"  Batch {i//batch_size+1}/{n_batches}: {len(all_close)} stocks downloaded", flush=True)
    except Exception as e:
        failed_tickers.extend(batch)
        print(f"  Batch error: {e}", flush=True)

print(f"\nDownloaded: {len(all_close)} stocks")
print(f"Failed/no data: {len(set(failed_tickers))} tickers")

# 取得できなかった除外銘柄をリスト（上場廃止等で取得不可）
failed_removed = set(failed_tickers) & truly_removed
if failed_removed:
    sample = sorted(failed_removed)[:20]
    print(f"  Failed removed tickers (delisted etc): {len(failed_removed)}")
    print(f"    Sample: {', '.join(sample)}")

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

# === データ品質フィルター ===
# 上場廃止後にティッカーが再利用された銘柄等で、日次±50%超のリターンが頻発する
# これらは明らかに不正データなので除外
DAILY_RET_CAP = 0.50  # ±50%
EXTREME_THRESHOLD = 5  # 極端なリターンが5回以上 → 不正データと判定

extreme_counts = (ret_df.abs() > DAILY_RET_CAP).sum()
bad_tickers = set(extreme_counts[extreme_counts >= EXTREME_THRESHOLD].index)

if bad_tickers:
    print(f"\n  === Data Quality Filter ===")
    print(f"  Tickers with {EXTREME_THRESHOLD}+ days of >{DAILY_RET_CAP*100:.0f}% daily return: {len(bad_tickers)}")
    for tk in sorted(bad_tickers)[:20]:
        cnt = int(extreme_counts[tk])
        print(f"    {tk}: {cnt} extreme days")
    if len(bad_tickers) > 20:
        print(f"    ... and {len(bad_tickers) - 20} more")

    # これらの銘柄を全DataFrameから除外
    good_cols = [c for c in close_df.columns if c not in bad_tickers]
    close_df = close_df[good_cols]
    ret_df = ret_df[good_cols]
    gap_ret_df = gap_ret_df[good_cols]
    intra_ret_df = intra_ret_df[good_cols]
    print(f"  Remaining tickers: {len(close_df.columns)}")

# さらに、残った銘柄でも日次リターンを±50%でキャップ（安全策）
ret_df = ret_df.clip(-DAILY_RET_CAP, DAILY_RET_CAP)
gap_ret_df = gap_ret_df.clip(-DAILY_RET_CAP, DAILY_RET_CAP)
intra_ret_df = intra_ret_df.clip(-DAILY_RET_CAP, DAILY_RET_CAP)

vol_df = ret_df.rolling(VOL_WINDOW).std()

del all_close, all_open, open_df
gc.collect()

mom_short = close_df.pct_change(LB_SHORT, fill_method=None)
mom_mid = close_df.pct_change(LB_MID, fill_method=None)
mom_long = close_df.pct_change(LB_LONG, fill_method=None)

print(f"Valid tickers in price data: {len(close_df.columns)}", flush=True)

# ============================================================
# 4. ユニバースマスク作成
# ============================================================
print("\n=== Building Universe Masks ===", flush=True)

# (A) Naive mask: 現在のS&P500メンバー × 全期間True
naive_cols = [t for t in current_tickers if t in close_df.columns]
naive_mask = pd.DataFrame(False, index=date_index, columns=close_df.columns)
naive_mask[naive_cols] = True

# (B) Historical v1 mask: 追加日のみ考慮（前版の方法）
hist_v1_mask = pd.DataFrame(False, index=date_index, columns=close_df.columns)
for tk in close_df.columns:
    if tk in ticker_date_added:
        hist_v1_mask.loc[date_index >= ticker_date_added[tk], tk] = True
    elif tk in current_tickers:
        # 追加日不明の現在メンバー → 古い銘柄と見なして全期間True
        hist_v1_mask[tk] = True

# (C) True Historical mask: 各日付のS&P500メンバーシップに基づく
true_hist_mask = pd.DataFrame(False, index=date_index, columns=close_df.columns)

# タイムラインから各日付のメンバーシップを適用
print("  Building true historical mask from timeline...")
available_cols = set(close_df.columns)

for i in range(len(timeline)):
    start_dt = timeline[i][0]
    end_dt = timeline[i+1][0] if i + 1 < len(timeline) else pd.Timestamp('2030-01-01')
    members = timeline[i][1] & available_cols  # 株価データがあるもののみ

    mask = (date_index >= start_dt) & (date_index < end_dt)
    if mask.sum() > 0:
        for tk in members:
            true_hist_mask.loc[mask, tk] = True

# 各時点のユニバースサイズ比較
print(f"\n  Universe sizes (avg stocks per year):")
print(f"  {'Year':>6}  {'Naive':>8}  {'Hist v1':>8}  {'True Hist':>10}")
for year in [2011, 2013, 2015, 2017, 2019, 2021, 2023, 2025]:
    yr_mask = date_index.year == year
    if yr_mask.sum() > 0:
        n_naive = naive_mask.loc[yr_mask].sum(axis=1).mean()
        n_v1 = hist_v1_mask.loc[yr_mask].sum(axis=1).mean()
        n_true = true_hist_mask.loc[yr_mask].sum(axis=1).mean()
        print(f"  {year:>6}  {n_naive:>8.0f}  {n_v1:>8.0f}  {n_true:>10.0f}")

# ============================================================
# 5. バックテストエンジン
# ============================================================
bt_dates = date_index[(date_index >= BT_START) & (date_index <= BT_END)]

# 隔月リバランス（Improved用）
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

# 年4回リバランス（Original用）
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

            ms = ms[ms.index.isin(eligible)].sort_values(ascending=False)
            mm = mm[mm.index.isin(eligible)].sort_values(ascending=False)
            ml = ml[ml.index.isin(eligible)].sort_values(ascending=False)

            selected = []
            sector_count = {}

            def can_add(tk):
                if tk in selected:
                    return False
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
                    if count >= N_PER_BUCKET:
                        break
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
# 6. 全パターン実行
# ============================================================
print(f"\n=== Running backtests ===", flush=True)

configs = [
    # (name, mask, rebal, trend, invvol, sector_cap)
    ("Naive Original",      naive_mask,      quarterly_exec,   False, False, False),
    ("Hist-v1 Original",    hist_v1_mask,    quarterly_exec,   False, False, False),
    ("True-Hist Original",  true_hist_mask,  quarterly_exec,   False, False, False),
    ("Naive Improved",      naive_mask,      rebal_exec_dates, True,  True,  True),
    ("Hist-v1 Improved",    hist_v1_mask,    rebal_exec_dates, True,  True,  True),
    ("True-Hist Improved",  true_hist_mask,  rebal_exec_dates, True,  True,  True),
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
# 7. 結果出力
# ============================================================
print("\n" + "=" * 90)
print("  RESULTS: Naive vs Hist-v1 vs True-Historical")
print("=" * 90)
print(f"  {'Strategy':<22} {'Sharpe':>7} {'CAGR':>8} {'MDD':>8} {'Vol':>7}")
print("-" * 90)
for name in ['Naive Original', 'Hist-v1 Original', 'True-Hist Original',
             'Naive Improved', 'Hist-v1 Improved', 'True-Hist Improved', 'SPY']:
    r = results[name]
    print(f"  {name:<22} {r['sharpe']:>6.2f} {r['cagr']*100:>+7.1f}% "
          f"{r['mdd']*100:>+7.1f}% {r['vol']*100:>6.1f}%")
print("-" * 90)

# バイアスの定量化
print(f"\n  === Survivorship Bias Breakdown ===")
for strategy_type in ["Original", "Improved"]:
    naive_name = f"Naive {strategy_type}"
    v1_name = f"Hist-v1 {strategy_type}"
    true_name = f"True-Hist {strategy_type}"

    n = results[naive_name]
    v1 = results[v1_name]
    t = results[true_name]

    print(f"\n  [{strategy_type}]")
    print(f"  {'':>20} {'Sharpe':>8} {'CAGR':>8}")
    print(f"  {'Naive':<20} {n['sharpe']:>8.2f} {n['cagr']*100:>+7.1f}%")
    print(f"  {'Hist-v1':<20} {v1['sharpe']:>8.2f} {v1['cagr']*100:>+7.1f}%")
    print(f"  {'True-Historical':<20} {t['sharpe']:>8.2f} {t['cagr']*100:>+7.1f}%")
    print(f"  {'':>20} {'--------':>8} {'--------':>8}")
    print(f"  {'Total bias':<20} {n['sharpe']-t['sharpe']:>+7.2f} {(n['cagr']-t['cagr'])*100:>+7.1f}%")
    print(f"    うち 追加日バイアス {n['sharpe']-v1['sharpe']:>+7.2f} {(n['cagr']-v1['cagr'])*100:>+7.1f}%")
    print(f"    うち 除外銘柄バイアス {v1['sharpe']-t['sharpe']:>+7.2f} {(v1['cagr']-t['cagr'])*100:>+7.1f}%")

# ============================================================
# 8. 年次リターン比較
# ============================================================
print(f"\n  [Annual Returns]")
hdr = f"  {'Year':>6}"
short_names = {
    'Naive Original': 'NaiveO',
    'True-Hist Original': 'TrueO',
    'Naive Improved': 'NaiveI',
    'True-Hist Improved': 'TrueI',
    'SPY': 'SPY'
}
display_order = ['Naive Original', 'True-Hist Original', 'Naive Improved', 'True-Hist Improved', 'SPY']
for name in display_order:
    hdr += f"  {short_names[name]:>9}"
print(hdr)
print(f"  {'-'*6}" + f"  {'-'*9}" * len(display_order))

for year in range(2011, 2027):
    line = f"  {year:>6}"
    for name in display_order:
        rets = results[name]['rets']
        yr_mask = rets.index.year == year
        if yr_mask.sum() > 0:
            yr_ret = (1 + rets[yr_mask]).prod() - 1
            line += f"  {yr_ret*100:>+8.1f}%"
        else:
            line += f"  {'':>9}"
    print(line)

# ============================================================
# 9. 下落局面
# ============================================================
print(f"\n  [Crash Periods]")
crashes = [
    ("COVID-19", "2020-02-01", "2020-04-30"),
    ("2022 Bear", "2022-01-01", "2022-10-31"),
    ("2018 Q4", "2018-10-01", "2018-12-31"),
]

for cname, start, end in crashes:
    print(f"\n  {cname}:")
    for name in ['Naive Improved', 'True-Hist Improved', 'SPY']:
        rets = results[name]['rets']
        mask = (rets.index >= pd.Timestamp(start)) & (rets.index <= pd.Timestamp(end))
        pr = rets[mask]
        if len(pr) < 2:
            continue
        cum_pr = (1 + pr).cumprod()
        pret = cum_pr.iloc[-1] - 1
        pmdd = ((cum_pr - cum_pr.cummax()) / cum_pr.cummax()).min()
        short = name.replace("Improved", "Impr").replace("True-Hist", "TrueHist")
        print(f"    {short:<24} Return: {pret*100:>+7.1f}%  MDD: {pmdd*100:>+7.1f}%")

# ============================================================
# 10. 除外銘柄の分析
# ============================================================
print(f"\n  === Removed Stocks Analysis ===")

# True-Histにはあるが、Naiveにはない銘柄を特定
only_in_true = set(close_df.columns) - set(naive_cols)
in_true_and_data = only_in_true & set(close_df.columns)
print(f"  Stocks in True-Hist but not in Naive: {len(in_true_and_data)}")

# これらの銘柄のうち、BT期間中にS&P500にいた銘柄
removed_in_bt = []
for tk in in_true_and_data:
    # この銘柄がTrue-Histマスクで1になっている期間を確認
    tk_mask = true_hist_mask[tk] if tk in true_hist_mask.columns else None
    if tk_mask is not None:
        bt_period = tk_mask.loc[(date_index >= BT_START) & (date_index <= BT_END)]
        if bt_period.sum() > 0:
            # BT期間中にS&P500にいた
            first_date = bt_period[bt_period].index[0]
            last_date = bt_period[bt_period].index[-1]
            # この銘柄のBT期間中のリターン
            tk_rets = ret_df[tk].loc[first_date:last_date].dropna()
            if len(tk_rets) > 20:
                cum_ret = (1 + tk_rets).prod() - 1
                removed_in_bt.append({
                    'ticker': tk,
                    'first': first_date.strftime('%Y-%m-%d'),
                    'last': last_date.strftime('%Y-%m-%d'),
                    'days': len(tk_rets),
                    'total_return': cum_ret
                })

if removed_in_bt:
    removed_df = pd.DataFrame(removed_in_bt).sort_values('total_return')
    print(f"  Removed stocks active during BT: {len(removed_df)}")

    # 最もパフォーマンスが悪かった（= Naiveから除外されてバイアスの原因になった）銘柄
    print(f"\n  Worst performers (removed from S&P500 during BT period):")
    print(f"  {'Ticker':<8} {'In S&P':>12} {'Out S&P':>12} {'Days':>6} {'Return':>10}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*6} {'-'*10}")
    for _, row in removed_df.head(20).iterrows():
        print(f"  {row['ticker']:<8} {row['first']:>12} {row['last']:>12} "
              f"{row['days']:>6} {row['total_return']*100:>+9.1f}%")

    # 平均リターン
    avg_ret = removed_df['total_return'].mean()
    median_ret = removed_df['total_return'].median()
    print(f"\n  Average return of removed stocks: {avg_ret*100:+.1f}%")
    print(f"  Median return of removed stocks:  {median_ret*100:+.1f}%")
    print(f"  Negative return: {(removed_df['total_return'] < 0).sum()}/{len(removed_df)} "
          f"({(removed_df['total_return'] < 0).mean()*100:.0f}%)")

# ============================================================
# 11. 実運用期待値
# ============================================================
print(f"\n  === Realistic Performance Expectation ===")
true_imp = results['True-Hist Improved']
spy = results['SPY']
print(f"  True-Historical Improved: Sharpe {true_imp['sharpe']:.2f}, "
      f"CAGR {true_imp['cagr']*100:+.1f}%, MDD {true_imp['mdd']*100:.1f}%")
print(f"  SPY:                      Sharpe {spy['sharpe']:.2f}, "
      f"CAGR {spy['cagr']*100:+.1f}%, MDD {spy['mdd']*100:.1f}%")
print(f"  Alpha (CAGR):             {(true_imp['cagr']-spy['cagr'])*100:+.1f}%")
print(f"  Alpha (Sharpe):           {true_imp['sharpe']-spy['sharpe']:+.2f}")

# さらに取引コスト（年1-2%）を考慮
cost_low = 0.01
cost_high = 0.02
adj_cagr_low = true_imp['cagr'] - cost_low
adj_cagr_high = true_imp['cagr'] - cost_high
adj_sharpe_low = adj_cagr_low / true_imp['vol'] if true_imp['vol'] > 0 else 0
adj_sharpe_high = adj_cagr_high / true_imp['vol'] if true_imp['vol'] > 0 else 0

print(f"\n  After transaction costs (1-2%/yr):")
print(f"    Sharpe: {adj_sharpe_high:.2f} ~ {adj_sharpe_low:.2f}")
print(f"    CAGR:   {adj_cagr_high*100:+.1f}% ~ {adj_cagr_low*100:+.1f}%")
print(f"    vs SPY: {(adj_cagr_high-spy['cagr'])*100:+.1f}% ~ {(adj_cagr_low-spy['cagr'])*100:+.1f}%")

print("\n" + "=" * 90)
print("  Done.", flush=True)
print("=" * 90)
