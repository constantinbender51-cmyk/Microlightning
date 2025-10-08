import pandas as pd
import numpy as np
from itertools import product

# ----------  read once ---------------------------------------------------------
df = pd.read_csv('btc_daily.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

# ----------  engine wrapped in a function -------------------------------------
def run_macd(lev, stp_pct):
    """
    Replicates the exact logic you posted.
    Returns a dict with all key numbers for this (lev, stp_pct) pair.
    """
    # ---- MACD (proper warm-up) ----------------------------------------------
    ema12 = df['close'].ewm(span=12, min_periods=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, min_periods=26, adjust=False).mean()
    macd  = ema12 - ema26
    signal = macd.ewm(span=9, min_periods=9, adjust=False).mean()

    # 1/-1 on cross
    cross = np.where((macd > signal) & (macd.shift() <= signal.shift()),  1,
                    np.where((macd < signal) & (macd.shift() >= signal.shift()), -1, 0))
    pos = pd.Series(cross, index=df.index).replace(0, np.nan).ffill().fillna(0)

    # =====================  SINGLE RUN (WITH STOP-LOSS) ======================
    curve    = [10000]
    in_pos   = 0
    entry_p  = None
    entry_d  = None
    trades   = []
    stp      = False
    days_stp = 0
    stp_cnt  = 0
    stp_cnt_max = 0

    for i in range(1, len(df)):
        p_prev = df['close'].iloc[i-1]
        p_now  = df['close'].iloc[i]
        pos_i  = pos.iloc[i]

        # ----- stop-loss check -------------------------------------------------
        if (not stp) and in_pos != 0:
            hh = df['high'].iloc[i]
            ll = df['low'].iloc[i]
            if ((entry_p/hh-1)*in_pos >= stp_pct) or ((entry_p/ll-1)*in_pos >= stp_pct):
                stp = True
                stp_price = curve[-1] * (1 - stp_pct * lev)
                stp_cnt += 1
                stp_cnt_max = max(stp_cnt_max, stp_cnt)

        # ----- entry -----------------------------------------------------------
        if in_pos == 0 and pos_i != 0:
            in_pos  = pos_i
            entry_p = p_now
            entry_d = df['date'].iloc[i]
            stp     = False

        # ----- exit on opposite cross ------------------------------------------
        if in_pos != 0 and pos_i == -in_pos:
            ret = (p_now / entry_p - 1) * in_pos * lev
            if stp:
                trades.append((entry_d, df['date'].iloc[i], -stp_pct*lev))
            else:
                trades.append((entry_d, df['date'].iloc[i], ret))
                if ret >= 0:
                    stp_cnt = 0
                else:
                    stp_cnt += 1
                    stp_cnt_max = max(stp_cnt_max, stp_cnt)
            in_pos = 0
            stp    = False

        # ----- equity update ----------------------------------------------------
        if stp:
            curve.append(stp_price)
            days_stp += 1
        else:
            curve.append(curve[-1] * (1 + (p_now/p_prev - 1) * in_pos * lev))

    curve = pd.Series(curve, index=df.index)

    # ---------------------------  METRICS  ------------------------------------
    daily_ret = curve.pct_change().dropna()
    trades_ret = pd.Series([t[2] for t in trades])
    n_years = (df['date'].iloc[-1] - df['date'].iloc[0]).days / 365.25

    cagr = (curve.iloc[-1] / curve.iloc[0]) ** (1 / n_years) - 1
    vol  = daily_ret.std() * np.sqrt(252)
    sharpe = cagr / vol if vol else np.nan
    drawdown = curve / curve.cummax() - 1
    maxdd = drawdown.min()
    calmar = cagr / abs(maxdd) if maxdd else np.nan

    wins   = trades_ret[trades_ret > 0]
    losses = trades_ret[trades_ret < 0]
    win_rate = len(wins) / len(trades_ret) if trades_ret.size else 0
    avg_win  = wins.mean()   if len(wins)   else 0
    avg_loss = losses.mean() if len(losses) else 0
    payoff   = abs(avg_win / avg_loss) if avg_loss else np.nan
    profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() else np.nan
    expectancy = win_rate * avg_win - (1 - win_rate) * abs(avg_loss)
    kelly = expectancy / trades_ret.var() if trades_ret.var() > 0 else np.nan
    time_in_mkt = 1 - ((1 - (pos != 0).mean()) * len(df) + days_stp) / len(df)
    tail_ratio = (np.percentile(daily_ret, 95) /
                  abs(np.percentile(daily_ret, 5))) if daily_ret.size else np.nan
    trades_per_year = len(trades) / n_years
    lose_streak = (trades_ret < 0).astype(int)
    max_lose_streak = lose_streak.groupby(
                          lose_streak.diff().ne(0).cumsum()).sum().max()

    final_macd = curve.iloc[-1]
    final_hold = (df['close'].iloc[-1] / df['close'].iloc[0]) * 10000
    worst      = min(trades, key=lambda x: x[2])

    return dict(
        lev=lev, stp_pct=stp_pct,
        final=final_macd, hold=final_hold,
        cagr=cagr, vol=vol, sharpe=sharpe,
        maxdd=maxdd, calmar=calmar,
        trades_py=trades_per_year, win_rate=win_rate,
        avg_win=avg_win, avg_loss=avg_loss,
        payoff=payoff, pf=profit_factor,
        expectancy=expectancy, kelly=kelly,
        time_in_mkt=time_in_mkt, tail=tail_ratio,
        max_ls=max_lose_streak
    )

# ----------  parameter grid ----------------------------------------------------
leverages = range(1, 6)                 # 1 2 3 4 5
stop_pcts = np.arange(0.1, 8.1, 0.1)   # 0.1 0.2 … 8.0 %

records = []
for lev, stp in product(leverages, stop_pcts):
    print(f'running lev={lev}×  stp={stp:4.1f}% …')
    records.append(run_macd(lev, stp))

results = pd.DataFrame(records)

# ----------  quick inspection --------------------------------------------------
print('\n===== TOP 10 by Calmar =====')
print(results.sort_values('calmar', ascending=False).head(10))

# ----------  save to csv if you want ------------------------------------------
# results.to_csv('macd_scan_lev_stp.csv', index=False)
