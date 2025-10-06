import pandas as pd, numpy as np

df = pd.read_csv('btc_daily.csv', parse_dates=['date'])

ema12 = df['close'].ewm(span=12, adjust=False).mean()
ema26 = df['close'].ewm(span=26, adjust=False).mean()

# 1 = bullish cross (12 crosses above 26), -1 = bearish cross (12 crosses below 26)
cross = np.where((ema12 > ema26) & (ema12.shift() <= ema26.shift()),  1,
                np.where((ema12 < ema26) & (ema12.shift() >= ema26.shift()), -1, 0))

last5 = df.loc[cross != 0, ['date']].assign(signal=cross[cross != 0]).tail(5)
for _, r in last5.iterrows():
    print(r.date.strftime('%Y-%m-%d'), 'BUY' if r.signal == 1 else 'SELL')
