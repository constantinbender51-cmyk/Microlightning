import pandas as pd
import numpy as np
import time
import numpy as np

# ------------------------------------------------ read data ---------------------
df = pd.read_csv('btc_daily.csv', parse_dates=['date'])

# ---- MACD (proper warm-up) ----------------------------------------------------
ema12 = df['close'].ewm(span=12, min_periods=12, adjust=False).mean()
ema26 = df['close'].ewm(span=26, min_periods=26, adjust=False).mean()
macd  = ema12 - ema26
signal = macd.ewm(span=9, min_periods=9, adjust=False).mean()

# 1/-1 on cross
cross = np.where((macd > signal) & (macd.shift() <= signal.shift()),  1,
                np.where((macd < signal) & (macd.shift() >= signal.shift()), -1, 0))
pos = pd.Series(cross, index=df.index).replace(0, np.nan).ffill().fillna(0)

# =========================  BACK-TEST FUNCTION ================================
def run_macd_with_stop(stop_prc: float):
    """
    Run MACD strategy with an intrabar stop-loss of stop_prc %.
    Returns dict of key metrics (one row of the final table).
    """
    curve = [10000]
    in_pos = 0
    entry_p = None
    entry_d = None
    trades = []

    for i in range(1, len(df)):
        p_prev = df['close'].iloc[i-1]
        p_now  = df['close'].iloc[i]
        pos_i  = pos.iloc[i]

        # enter
        if in_pos == 0 and pos_i != 0:
            in_pos, entry_p, entry_d = pos_i, p_now, df['date'].iloc[i]

        # ---------- intrabar stop ----------
        if in_pos != 0:
            if in_pos == 1:                       # long
                stop_price = entry_p * (1 - stop_prc/100)
                if df['low'].iloc[i] <= stop_price:
                    trades.append((entry_d, df['date'].iloc[i], -stop_prc/100))
                    in_pos = 0
            else:                                 # short
                stop_price = entry_p * (1 + stop_prc/100)
                if df['high'].iloc[i] >= stop_price:
                    trades.append((entry_d, df['date'].iloc[i], -stop_prc/100))
                    in_pos = 0

        # exit on opposite MACD cross
        if in_pos != 0 and pos_i == -in_pos:
            ret = (p_now / entry_p - 1) * in_pos
            trades.append((entry_d, df['date'].iloc[i], ret))
            in_pos = pos_i
            entry_p, entry_d = p_now, df['date'].iloc[i]

        # equity update
        curve.append(curve[-1] * (1 + (p_now/p_prev - 1) * in_pos))

    curve = pd.Series(curve, index=df.index)

    # ---------- statistics ----------
    daily_ret = curve.pct_change().dropna()
    trades_ret = pd.Series([t[2] for t in trades])
    n_years = (df['date'].iloc[-1] - df['date'].iloc[0]).days / 365.25

    cagr = (curve.iloc[-1] / curve.iloc[0]) ** (1 / n_years) - 1
    vol  = daily_ret.std() * np.sqrt(252)
    sharpe = cagr / vol if vol else np.nan
    maxdd  = (curve / curve.cummax() - 1).min()
    calmar = cagr / abs(maxdd) if maxdd else np.nan
    win_rate = (trades_ret > 0).mean() if trades_ret.size else 0
    expectancy = trades_ret.mean() if trades_ret.size else np.nan
    trades_per_year = len(trades) / n_years
    time_in_mkt = (pos != 0).mean()

    return {
        'stop_%'          : stop_prc,
        'final_equity'    : curve.iloc[-1],
        'CAGR_%'          : cagr * 100,
        'Sharpe'          : sharpe,
        'Calmar'          : calmar,
        'maxDD_%'         : maxdd * 100,
        'win_rate_%'      : win_rate * 100,
        'exp_per_trade_%' : expectancy * 100,
        'trades_per_yr'   : trades_per_year,
        'time_in_mkt_%'   : time_in_mkt * 100,
    }

# -------------------- DEFAULT 2 % RUN (for reference) -------------------------
curve_2 = run_macd_with_stop(2.0)
print('\n----- DEFAULT 2 % STOP -----')
print(f"Final equity:  {curve_2['final_equity']:,.0f}")
print(f"CAGR:          {curve_2['CAGR_%']:.2f}%")
print(f"Sharpe:        {curve_2['Sharpe']:.2f}")
print(f"Calmar:        {curve_2['Calmar']:.2f}")
print(f"Max DD:        {curve_2['maxDD_%']:.2f}%")

# ========================  SWEEP 0.5 % – 8.0 %  ==============================
stop_grid = np.round(np.arange(0.5, 8.05, 0.1), 2)   # 0.5, 0.6, …, 8.0 %
summary   = []

print('\nScanning stop levels …')
for s in stop_grid:
    summary.append(run_macd_with_stop(s))

summary_df = pd.DataFrame(summary).set_index('stop_%')

# -------------------- quick report --------------------
print('\nTop-5 by Sharpe')
print(summary_df.nlargest(5, 'Sharpe')[['CAGR_%', 'Sharpe', 'Calmar', 'maxDD_%']])

print('\nTop-5 by Calmar')
print(summary_df.nlargest(5, 'Calmar')[['CAGR_%', 'Sharpe', 'Calmar', 'maxDD_%']])

# -------------------- optional plot --------------------
try:
    import matplotlib.pyplot as plt
    summary_df[['Sharpe', 'Calmar']].plot(figsize=(8, 4), title='Stop-loss sweep')
    plt.xlabel('stop loss %')
    plt.show()
except ImportError:
    pass
