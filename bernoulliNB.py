"""
BTC daily next-bar predictor
BernoulliNB with 7 binary features, rolling 252-day retrain, walk-forward
"""

import pandas as pd
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from tqdm import tqdm   # eye-candy progress bar

# ------------------------------------------------------------------
# 1. load daily candles (reuse your existing loader)
# ------------------------------------------------------------------
def load_daily(path='xbtusd_1h_8y.csv'):
    try:
        df = pd.read_csv(path)
        df['open_time'] = pd.to_datetime(df['open_time'], format='ISO8601')
        df.set_index('open_time', inplace=True)
        df.sort_index(inplace=True)
        daily = df.resample('D').agg({
            'open':  'first',
            'high':  'max',
            'low':   'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        return daily
    except Exception as e:
        print("Data load error:", e); return None

# ------------------------------------------------------------------
# 2. build 7 binary features
# ------------------------------------------------------------------
def make_features(daily):
    df = daily.copy()
    # 0/1 flags
    df['price_up'] = (df['close'] > df['open']).astype(int)
    df['vol_up']   = (df['volume'] > df['volume'].shift(1)).astype(int)
    for n in [2, 5, 10, 20, 50]:
        sma = df['close'].rolling(n).mean()
        df[f'sma{n}_up'] = (sma > sma.shift(1)).astype(int)
    return df

# ------------------------------------------------------------------
# 3. walk-forward engine
# ------------------------------------------------------------------
ROLL = 252          # training window length
BUY  = 0.55         # prob threshold long
SELL = 0.45         # prob threshold short

def walk_forward(df):
    feats = ['price_up', 'vol_up', 'sma2_up', 'sma5_up',
             'sma10_up', 'sma20_up', 'sma50_up']
    df = df.dropna()            # sma50 needs 50 prior bars
    preds = []                  # store prob_green & position
    model = BernoulliNB()

    for i in tqdm(range(ROLL, len(df)), desc='walk-forward'):
        # ---- train window ----
        X_train = df[feats].iloc[i-ROLL : i]
        y_train = df['price_up'].iloc[i-ROLL+1 : i+1]   # next-day labels

        # ---- fit ----
        model.fit(X_train, y_train)

        # ---- predict ----
        # NEW
        x_today = df[feats].iloc[i:i+1]         # keeps DataFrame + column names
        prob = model.predict_proba(x_today)[0, 1]


        # ---- position rule ----
        if prob > BUY:   pos =  1
        elif prob < SELL: pos = -1
        else:             pos =  0
        preds.append({'date': df.index[i], 'prob': prob, 'pos': pos})

    return pd.DataFrame(preds).set_index('date')

# ------------------------------------------------------------------
# 4. quick performance helper (intraday round-turn)
# ------------------------------------------------------------------
def compute_stats(preds, df):
    merge = preds.join(df[['open', 'close']], how='left')
    merge['ret'] = merge['pos'] * (merge['close'].shift(-1) - merge['open'].shift(-1))
    merge = merge.dropna(subset=['ret'])
    c = merge['ret'].cumsum()
    sharpe = merge['ret'].mean() / merge['ret'].std() * np.sqrt(365)
    maxdd  = (c.cummax() - c).max()
    print(f"Sharpe {sharpe:.2f}  MaxDD {maxdd:.2f}  FinalEq {c.iloc[-1]:.2f}")

# ------------------------------------------------------------------
# 5. one-click
# ------------------------------------------------------------------
if __name__ == "__main__":
    daily = load_daily()
    if daily is not None:
        data  = make_features(daily)
        preds = walk_forward(data)
        compute_stats(preds, data)
        preds.to_csv('nb_signals.csv')
        print("signals saved -> nb_signals.csv")
