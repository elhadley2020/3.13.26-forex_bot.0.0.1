#############################################
# ADVANCED 5-MIN OANDA BOT WITH VOL, ALPHA PRUNING, WALK-FWD ML
#############################################

import pandas as pd
import numpy as np
import time
import logging

import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders

from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestRegressor
import ta

#############################################
# CONFIG
#############################################

API_KEY = "YOUR_API_KEY"
ACCOUNT_ID = "YOUR_ACCOUNT"

PAIRS = ["EUR_USD","GBP_USD","USD_JPY","AUD_USD","USD_CAD"]
TIMEFRAME = "M5"
CANDLES = 500
BASE_RISK = 0.01  # 1% per trade
MAX_DRAWDOWN = 0.15
RETRAIN_INTERVAL = 288  # retrain every 288 candles (~1 day)

client = oandapyV20.API(access_token=API_KEY)

logging.basicConfig(filename="trading_5min_pro.log",
                    level=logging.INFO,
                    format="%(asctime)s %(message)s")

#############################################
# DATA LOADER
#############################################

def get_candles(pair):
    params = {"count":CANDLES,"granularity":TIMEFRAME,"price":"M"}
    r = instruments.InstrumentsCandles(instrument=pair, params=params)
    client.request(r)
    closes = [float(c["mid"]["c"]) for c in r.response["candles"]]
    df = pd.DataFrame(closes, columns=["close"])
    return df

#############################################
# FEATURE ENGINEERING
#############################################

def build_features(df):
    df["returns"] = np.log(df.close / df.close.shift())
    df["volatility"] = df["returns"].rolling(20).std()
    df["momentum"] = df.close - df.close.shift(10)
    df["trend"] = df.close.rolling(50).mean() - df.close.rolling(200).mean()
    df["rsi"] = ta.momentum.RSIIndicator(df.close).rsi()
    df.dropna(inplace=True)
    return df

#############################################
# REGIME DETECTION
#############################################

class RegimeModel:
    def __init__(self):
        self.hmm = GaussianHMM(n_components=3, covariance_type="full", n_iter=200)
        self.trend_clf = RandomForestRegressor(n_estimators=200, max_depth=5)

    def train(self, df):
        X_hmm = df[["returns","volatility"]]
        self.hmm.fit(X_hmm)

        X_trend = df[["momentum","trend","rsi"]].iloc[:-1]
        y_trend = df["returns"].shift(-1).iloc[:-1]
        self.trend_clf.fit(X_trend, y_trend)

    def predict(self, df):
        hmm_state = self.hmm.predict(df[["returns","volatility"]])[-1]
        trend_pred = self.trend_clf.predict(df[["momentum","trend","rsi"]].iloc[[-1]])[0]
        if hmm_state == 2:
            return "high_vol"
        if trend_pred > 0:
            return "bull"
        if trend_pred < 0:
            return "bear"
        return "range"

#############################################
# ALPHA FACTORY + PRUNING
#############################################

def generate_alphas(df):
    alphas = {}
    windows = [5,10,20,50]
    for w1 in windows:
        for w2 in windows:
            if w1 >= w2: continue
            name = f"ma_{w1}_{w2}"
            alphas[name] = df.close.rolling(w1).mean() - df.close.rolling(w2).mean()
    return alphas

def prune_alphas(alphas):
    df_alpha = pd.DataFrame(alphas)
    corr_matrix = df_alpha.corr().abs()
    to_remove = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i,j] > 0.8:
                to_remove.add(corr_matrix.columns[i])
    return {k:v for k,v in alphas.items() if k not in to_remove}

def alpha_score(alpha, returns):
    pnl = alpha.shift() * returns
    if pnl.std() == 0: return 0
    return pnl.mean() / pnl.std()

def select_alphas(alphas, returns):
    selected = {}
    for name, alpha in alphas.items():
        score = alpha_score(alpha, returns)
        if score > 1:
            selected[name] = alpha
    return selected

#############################################
# META-MODEL
#############################################

class MetaModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=200, max_depth=5)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict_weights(self, X):
        preds = self.model.predict(X)
        preds = np.maximum(preds,0)
        if preds.sum() == 0:
            return np.ones(len(preds))/len(preds)
        return preds/preds.sum()

#############################################
# SIGNAL COMBINATION + VOLATILITY TARGET
#############################################

def combine_signals(selected, weights=None):
    signals = np.array([alpha.iloc[-1] for alpha in selected.values()])
    if weights is None:
        weights = np.ones(len(signals))/len(signals)
    return np.dot(signals, weights)

def calculate_trade_size(volatility):
    # Trade size inversely proportional to volatility
    return max(1, int(BASE_RISK / volatility * 1000))  # scale factor 1000

#############################################
# ORDER EXECUTION
#############################################

def place_order(pair, units):
    order = {"order":{"instrument":pair,"units":str(units),"type":"MARKET",
                       "timeInForce":"FOK","positionFill":"DEFAULT"}}
    r = orders.OrderCreate(ACCOUNT_ID, data=order)
    client.request(r)
    logging.info(f"Placed order {pair} units={units}")

#############################################
# DRAW-DOWN MONITOR
#############################################

equity_peak = 0
def allowed_to_trade(equity):
    global equity_peak
    if equity > equity_peak: equity_peak = equity
    dd = (equity_peak - equity)/equity_peak
    return dd < MAX_DRAWDOWN

#############################################
# MAIN LOOP WITH WALK-FORWARD TRAINING
#############################################

regime_model = RegimeModel()
meta_model = MetaModel()
candle_counter = 0

while True:
    for pair in PAIRS:
        try:
            df = get_candles(pair)
            df = build_features(df)

            # Walk-forward training daily
            if candle_counter % RETRAIN_INTERVAL == 0:
                regime_model.train(df)
                # For meta-model training, use past returns as target
                alphas = generate_alphas(df)
                alphas = prune_alphas(alphas)
                selected = select_alphas(alphas, df["returns"])
                if len(selected) > 0:
                    X_meta = np.array([a.values for a in selected.values()]).T
                    y_meta = df["returns"].iloc[-X_meta.shape[0]:]
                    meta_model.train(X_meta, y_meta)

            regime = regime_model.predict(df)

            alphas = generate_alphas(df)
            alphas = prune_alphas(alphas)
            selected = select_alphas(alphas, df["returns"])

            X_meta = np.array([a.iloc[-1] for a in selected.values()]).reshape(1,-1)
            weights = meta_model.predict_weights(X_meta) if len(selected) > 0 else None

            final_signal = combine_signals(selected, weights)

            vol = df["volatility"].iloc[-1]
            size = calculate_trade_size(vol)

            if allowed_to_trade(1.0):
                if final_signal > 0:
                    place_order(pair, size)
                elif final_signal < 0:
                    place_order(pair, -size)

            logging.info(f"{pair} regime={regime} signal={final_signal} trade_size={size}")

        except Exception as e:
            logging.error(f"Error on {pair}: {e}")

    candle_counter += 1
    time.sleep(300)  # 5 minutes
