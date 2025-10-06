import pandas as pd, numpy as np

df = pd.read_csv('btc_daily.csv', parse_dates=['date'])

# MACD
ema12 = df['close'].ewm(12, adjust=False).mean()
ema26 = df['close'].ewm(26, adjust=False).mean()
macd = ema12 - ema26
signal = macd.ewm(9, adjust=False).mean()

# 1/-1 on MACD cross
pos = np.where((macd > signal) & (macd.shift() <= signal.shift()),  1,
              np.where((macd < signal) & (macd.shift() >= signal.shift()), -1, np.nan))
pos = pd.Series(pos, index=df.index).ffill().fillna(0)

# equity curve
ret = df['close'].pct_change()
curve = (1 + pos.shift() * ret).cumprod() * 10000

# trade list
trades = []
in_pos = 0
entry_p = entry_d = None
for i, p in enumerate(pos):
    if in_pos == 0 and p != 0:               # enter
        in_pos, entry_p, entry_d = p, df['close'].iloc[i], df['date'].iloc[i]
    elif in_pos != 0 and (p == -in_pos or i == len(pos)-1):  # exit
        ret = (df['close'].iloc[i] / entry_p - 1) * in_pos
        trades.append((entry_d, df['date'].iloc[i], ret))
        in_pos = 0 if i == len(pos)-1 else p
        entry_p = entry_d = None

worst_trade = min(trades, key=lambda x: x[2])
maxbal = curve.cummax()

print(f"MACD equity: €{curve.iloc[-1]:,.0f}")
print(f"B&H equity: €{(df['close'].iloc[-1]/df['close'].iloc[0]*10000):,.0f}")
print(f"Worst trade: {worst_trade[2]*100:.1f}% ({worst_trade[1].strftime('%Y-%m-%d')})")
print(f"Max drawdown: {(curve/maxbal-1).min()*100:.1f}%")
