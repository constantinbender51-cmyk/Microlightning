import pandas as pd, numpy as np, time

df = pd.read_csv('btc_daily.csv', parse_dates=['date'])

# Ensure EMAs start only after enough data
ema12 = df['close'].ewm(span=12, min_periods=12, adjust=False).mean()
ema26 = df['close'].ewm(span=26, min_periods=26, adjust=False).mean()
macd = ema12 - ema26
signal = macd.ewm(span=9, min_periods=9, adjust=False).mean()

# Crosses: 1 = bullish, -1 = bearish
cross = np.where((macd > signal) & (macd.shift() <= signal.shift()),  1,
                np.where((macd < signal) & (macd.shift() >= signal.shift()), -1, 0))

# Print only valid crosses
for i in range(len(df)):
    if cross[i]:
        print(f"{df.loc[i, 'date']:%Y-%m-%d}  {'BUY' if cross[i] == 1 else 'SELL'}  @ ${df.loc[i, 'close']:.2f}")
        time.sleep(0.01)
