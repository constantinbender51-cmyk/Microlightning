import pandas as pd, numpy as np

df = pd.read_csv('btc_daily.csv', parse_dates=['date'])

# MACD
ema12 = df['close'].ewm(12, adjust=False).mean()
ema26 = df['close'].ewm(26, adjust=False).mean()
macd = ema12 - ema26
signal = macd.ewm(9, adjust=False).mean()

# position: 1/-1 on cross, 0 otherwise
pos = np.where((macd > signal) & (macd.shift() <= signal.shift()),  1,
              np.where((macd < signal) & (macd.shift() >= signal.shift()), -1, np.nan))
pos = pd.Series(pos, index=df.index).ffill().fillna(0)

# equity curve
curve = (1 + pos.shift() * df['close'].pct_change()).cumprod() * 10000

# ----- trades -----
trades = []          # (entry_date, exit_date, return)
in_pos = 0
for i, p in enumerate(pos):
    if in_pos == 0 and p != 0:                    # new position
        in_pos, entry_p, entry_d = p, df['close'].iloc[i], df['date'].iloc[i]
    elif in_pos != 0 and p == -in_pos:            # opposite cross → close
        ret = (df['close'].iloc[i] / entry_p - 1) * in_pos
        trades.append((entry_d, df['date'].iloc[i], ret))
        in_pos = p                                # flip direction
        entry_p, entry_d = df['close'].iloc[i], df['date'].iloc[i]

# metrics
worst = min(trades, key=lambda x: x[2]) if trades else ('-', '-', 0)
maxbal = curve.cummax()

print(f"MACD equity: €{curve.iloc[-1]:,.0f}")
print(f"B&H equity: €{(df['close'].iloc[-1]/df['close'].iloc[0]*10000):,.0f}")
print(f"Worst trade: {worst[2]*100:.1f}%  (exit {worst[1].strftime('%Y-%m-%d')})")
print(f"Max drawdown: {(curve/maxbal-1).min()*100:.1f}%")
