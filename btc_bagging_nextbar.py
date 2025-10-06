# btc_bagging_nextbar.py
import pandas as pd
import ta
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. Load data ---------------------------------------------------------------
df = pd.read_csv('btc_daily.csv', parse_dates=['date']).sort_values('date')

# KEEP volume (and any other cols you may add later)
df = df[['date', 'open', 'high', 'low', 'close', 'volume']].dropna()

# 2. Build base features -----------------------------------------------------
df['ret'] = df['close'].pct_change()
df['EMA20']  = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
df['EMA200'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
df['ADX']    = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
df['ATR']    = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()

# --- newly added indicators ---
# 1. Accumulation/Distribution Index
df['ADI'] = ta.volume.AccDistIndexIndicator(
                high=df['high'], low=df['low'],
                close=df['close'], volume=df['volume']).acc_dist_index()

# 2. Money Flow Index
df['MFI'] = ta.volume.MFIIndicator(
                high=df['high'], low=df['low'],
                close=df['close'], volume=df['volume'],
                window=14).money_flow_index()

# 3. Bollinger Bands (middle, upper, lower) + %-band width
bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
df['BB_high']  = bb.bollinger_hband()
df['BB_low']   = bb.bollinger_lband()
df['BB_width'] = (df['BB_high'] - df['BB_low']) / df['close']

# 4. Keltner Channel Width
kc = ta.volatility.KeltnerChannel(
        high=df['high'], low=df['low'], close=df['close'],
        window=20, window_atr=10)
df['KC_high']  = kc.keltner_channel_hband()
df['KC_low']   = kc.keltner_channel_lband()
df['KC_width'] = (df['KC_high'] - df['KC_low']) / df['close']

# 5. Parabolic SAR
df['PSAR'] = ta.trend.PSARIndicator(
                high=df['high'], low=df['low'],
                close=df['close'],
                step=0.02, max_step=0.2).psar()
# optional: distance to PSAR in % of price
df['PSAR_dist'] = (df['PSAR'] - df['close']) / df['close']

# trend vs EMA20 (kept from original)
df['trend'] = df['close'].gt(df['EMA20']).map({True: 'UP', False: 'DOWN'})
grp = (df['trend'] != df['trend'].shift()).cumsum()
df['consolid'] = df.groupby(grp).cumcount() + 1
df.loc[df['trend'] == 'DOWN', 'consolid'] *= -1

# 3. Feature matrix ----------------------------------------------------------
FEATS = ['ret', 'close-EMA20', 'close-EMA200', 'ADX', 'ATR',
         'ADI', 'MFI', 'BB_width', 'KC_width', 'PSAR_dist', 'consolid']
df['close-EMA20']  = (df['close'] - df['EMA20']) / df['EMA20']
df['close-EMA200'] = (df['close'] - df['EMA200']) / df['EMA200']

# 3 lags of each feature
for feat in FEATS:
    for lag in range(1, 4):
        df[f'{feat}_lag{lag}'] = df[feat].shift(lag)

# target: next bar return
df['y'] = df['ret'].shift(-1)

# drop rows with NaNs created by lags / target
df = df.dropna()

# 4. Walk-forward split ------------------------------------------------------
# last 20 % for test
test_split = int(len(df) * 0.8)
train_df = df.iloc[:test_split].copy()
test_df  = df.iloc[test_split:].copy()

X_cols = [c for c in df.columns if c.startswith(tuple(FEATS))]

scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[X_cols])
y_train = train_df['y'].values

X_test  = scaler.transform(test_df[X_cols])
y_test  = test_df['y'].values

# 5. Model -------------------------------------------------------------------
base = DecisionTreeRegressor(max_depth=4, min_samples_leaf=20)
model = BaggingRegressor(
    estimator=base,
    n_estimators=500,
    max_samples=0.8,
    max_features=0.8,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# 6. Predict -----------------------------------------------------------------
pred_ret = model.predict(X_test)
pred_close = test_df['close'].values * (1 + pred_ret)

# 7. Quick evaluation --------------------------------------------------------
test_df = test_df.copy()
test_df['pred_ret'] = pred_ret
test_df['pred_close'] = pred_close

# directional accuracy
test_df['dir_real'] = np.sign(test_df['y'])
test_df['dir_pred'] = np.sign(test_df['pred_ret'])
dir_acc = (test_df['dir_real'] == test_df['dir_pred']).mean()
print(f"Directional accuracy on test: {dir_acc:.2%}")

# MAE of return forecast
mae = np.abs(test_df['y'] - test_df['pred_ret']).mean()
print(f"MAE next-bar return: {mae:.4f}")

# 8. Show last few predictions -----------------------------------------------
cols = ['date', 'close', 'y', 'pred_ret', 'pred_close']
print(test_df[cols].tail())
