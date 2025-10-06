import pandas as pd, numpy as np

df = pd.read_csv('btc_daily.csv', parse_dates=['date'])

ema12 = df['close'].ewm(span=12, adjust=False).mean()
ema26 = df['close'].ewm(span=26, adjust=False).mean()
macd = ema12 - ema26
signal = macd.ewm(span=9, adjust=False).mean()

cross = np.where((macd > signal) & (macd.shift() <= signal.shift()),  1,
                np.where((macd < signal) & (macd.shift() >= signal.shift()), -1, 0))

last5 = df.loc[cross != 0, ['date']].assign(signal=cross[cross != 0]).tail(5)
for _, r in last5.iterrows():
    print(r.date.strftime('%Y-%m-%d'), 'BUY' if r.signal == 1 else 'SELL')
