import pandas as pd
import numpy as np
import threading
import time

import oandapyV20
import oandapyV20.endpoints.orders as orders
from oandapyV20.endpoints.accounts import AccountDetails
from oandapyV20.endpoints.instruments import InstrumentsCandles

from hmmlearn.hmm import GaussianHMM

# -------------------------
# CONFIG
# -------------------------
API_KEY = "YOUR_API_KEY"
ACCOUNT_ID = "YOUR_ACCOUNT_ID"

client = oandapyV20.API(access_token=API_KEY)

INSTRUMENTS = [
    "EUR_USD", "GBP_USD", "AUD_USD", "USD_JPY", "USD_CHF"
]

RISK_PCT = 0.01  # 1% risk per trade

# -------------------------
# DATA FETCHING
# -------------------------
def fetch_candles(inst):
    params = {"granularity": "H1", "count": 200}
    r = InstrumentsCandles(instrument=inst, params=params)
    data = client.request(r)
    df = pd.DataFrame([{
        "time": c["time"],
        "open": float(c["mid"]["o"]),
        "high": float(c["mid"]["h"]),
        "low": float(c["mid"]["l"]),
        "close": float(c["mid"]["c"])
    } for c in data["candles"] if c["complete"]])
    return df

# -------------------------
# INDICATORS
# -------------------------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def compute_atr(df, period=14):
    tr1 = df.high - df.low
    tr2 = abs(df.high - df.close.shift())
    tr3 = abs(df.low - df.close.shift())
    tr = np.maximum.reduce([tr1, tr2, tr3])
    return pd.Series(tr).rolling(period).mean()

def compute_adx(df, period=14):
    plus_dm = df.high.diff()
    minus_dm = df.low.diff() * -1
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    tr = compute_atr(df, 1)
    plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / tr
    minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / tr
    dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    return dx.rolling(period).mean()

def compute_hurst(series):
    lags = range(2, 20)
    tau = [np.std(series[lag:] - series[:-lag]) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2

def compute_indicators(df):
    df["SMA20"] = df.close.rolling(20).mean()
    df["SMA50"] = df.close.rolling(50).mean()
    df["RSI"] = compute_rsi(df.close)
    df["ATR"] = compute_atr(df)
    df["ADX"] = compute_adx(df)
    df["STD"] = df.close.rolling(20).std()
    df["UpperBB"] = df.SMA20 + 2 * df.STD
    df["LowerBB"] = df.SMA20 - 2 * df.STD
    df["BB_width"] = df.UpperBB - df.LowerBB
    df["HURST"] = compute_hurst(df.close)
    return df

# -------------------------
# HMM 20-REGIME ENGINE
# -------------------------
hmm_model = GaussianHMM(n_components=20, covariance_type="full", n_iter=2000, random_state=42)

def build_hmm_features(df):
    returns = np.log(df.close).diff()
    volatility = df.close.pct_change().rolling(10).std()
    momentum = compute_rsi(df.close)
    bb_width = df.close.rolling(20).std() * 2
    slope = df.SMA20 - df.SMA50
    features = pd.concat([returns, volatility, momentum, bb_width, slope], axis=1).dropna()
    return features

def train_hmm(df):
    X = build_hmm_features(df)
    hmm_model.fit(X)
    return X

HMM_STRATEGY_MAP = {
    0:"mean_reversion",1:"mean_reversion",2:"volatility_fade",
    3:"trend_follow",4:"trend_follow",5:"breakout",
    6:"trend_follow",7:"trend_follow",8:"breakout",
    9:"mean_reversion",10:"mean_reversion",11:"volatility_fade",
    12:"trend_follow",13:"trend_follow",14:"breakout",
    15:"breakout",16:"risk_off",17:"risk_off",
    18:"reversal",19:"neutral"
}

def hmm_strategy(state):
    return HMM_STRATEGY_MAP.get(state,"neutral")

# -------------------------
# STRATEGY PROBABILITY FILTER
# -------------------------
def score_strategy(regime, df):
    last = df.iloc[-1]
    score = 0
    if regime == "mean_reversion":
        if last.RSI < 30 or last.RSI > 70: score = 0.8
    if regime in ["trend_follow", "breakout"]:
        if last.ADX > 25: score = 0.75
    if regime in ["volatility_fade", "reversal"]:
        score = 0.7
    if regime == "risk_off":
        score = 0
    return score

# -------------------------
# ACCOUNT & POSITION SIZING (Equity Compounding)
# -------------------------
def get_equity():
    r = AccountDetails(accountID=ACCOUNT_ID)
    data = client.request(r)
    return float(data["account"]["NAV"])

def position_size_compounded(atr, price, equity, risk_pct=RISK_PCT):
    risk_amount = equity * risk_pct
    units = int(risk_amount / atr)
    return units

# -------------------------
# ORDER EXECUTION
# -------------------------
def place_order(inst, units, sl, tp):
    data = {
        "order": {
            "instrument": inst,
            "units": str(units),
            "type": "MARKET",
            "positionFill": "DEFAULT",
            "stopLossOnFill":{"price":str(sl)},
            "takeProfitOnFill":{"price":str(tp)}
        }
    }
    r = orders.OrderCreate(accountID=ACCOUNT_ID, data=data)
    client.request(r)

# -------------------------
# CORRELATION FILTER
# -------------------------
def correlation_matrix(price_history):
    symbols = list(price_history.keys())
    corr = pd.DataFrame(index=symbols, columns=symbols)
    for i in symbols:
        for j in symbols:
            corr.loc[i,j] = price_history[i].close.corr(price_history[j].close)
    return corr

def correlation_block(inst, corr):
    for other in corr.columns:
        if other != inst:
            if abs(corr.loc[inst,other]) > 0.8:
                return True
    return False

# -------------------------
# TRADE ENGINE
# -------------------------
def trade_engine(inst, price_history):
    while True:
        df = fetch_candles(inst)
        df = compute_indicators(df)
        price_history[inst] = df

        # HMM regime detection
        try:
            state = hmm_model.predict(build_hmm_features(df))[-1]
            regime = hmm_strategy(state)
        except:
            regime = "neutral"

        # Probability filter
        prob = score_strategy(regime, df)
        if prob < 0.65 or regime == "risk_off":
            time.sleep(30)
            continue

        # Correlation check
        corr = correlation_matrix(price_history)
        if correlation_block(inst, corr):
            time.sleep(30)
            continue

        last = df.iloc[-1]
        atr = last.ATR
        price = last.close

        # Equity compounding
        equity = get_equity()
        units = position_size_compounded(atr, price, equity)

        stop = 1.2 * atr
        take = 2.5 * atr

        if "down" in regime:
            units = -units
            sl = price + stop
            tp = price - take
        else:
            sl = price - stop
            tp = price + take

        place_order(inst, units, sl, tp)
        print(inst, regime, state, prob, "units:", units, "equity:", equity)
        time.sleep(30)

# -------------------------
# PORTFOLIO RUNNER
# -------------------------
def run():
    price_history = {}
    # Train HMM for all instruments first
    for inst in INSTRUMENTS:
        df = fetch_candles(inst)
        compute_indicators(df)
        train_hmm(df)

    threads = []
    for inst in INSTRUMENTS:
        t = threading.Thread(target=trade_engine, args=(inst, price_history))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

if __name__ == "__main__":
    run()
