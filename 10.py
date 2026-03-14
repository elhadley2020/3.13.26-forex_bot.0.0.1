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

INSTRUMENTS = ["EUR_USD", "GBP_USD", "AUD_USD", "USD_JPY", "USD_CHF"]
BASE_RISK_PCT = 0.01  # 1% risk per trade
BASE_PROB_THRESHOLD = 0.75  # high-confidence trades
MAX_DRAWDOWN = 0.1  # 10% max drawdown cap

# -------------------------
# DATA FETCHING
# -------------------------
def fetch_candles(inst):
    params = {"granularity":"H1", "count":200}
    r = InstrumentsCandles(instrument=inst, params=params)
    data = client.request(r)
    df = pd.DataFrame([{
        "time":c["time"],
        "open":float(c["mid"]["o"]),
        "high":float(c["mid"]["h"]),
        "low":float(c["mid"]["l"]),
        "close":float(c["mid"]["c"])
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
    rs = avg_gain/(avg_loss + 1e-9)
    return 100-(100/(1+rs))

def compute_atr(df, period=14):
    tr1 = df.high - df.low
    tr2 = abs(df.high - df.close.shift())
    tr3 = abs(df.low - df.close.shift())
    tr = np.maximum.reduce([tr1,tr2,tr3])
    return pd.Series(tr).rolling(period).mean()

def compute_slope(df, short=20, long=50):
    return df.close.rolling(short).mean() - df.close.rolling(long).mean()

def compute_indicators(df):
    df["SMA20"] = df.close.rolling(20).mean()
    df["SMA50"] = df.close.rolling(50).mean()
    df["RSI"] = compute_rsi(df.close)
    df["ATR"] = compute_atr(df)
    df["Slope"] = compute_slope(df)
    df["STD"] = df.close.rolling(20).std()
    df["UpperBB"] = df.SMA20 + 2*df.STD
    df["LowerBB"] = df.SMA20 - 2*df.STD
    df["BB_width"] = df.UpperBB - df.LowerBB
    return df

# -------------------------
# HMM 20-REGIME ENGINE
# -------------------------
hmm_model = GaussianHMM(n_components=20, covariance_type="full", n_iter=2000, random_state=42)

def build_hmm_features(df):
    returns = np.log(df.close).diff()
    volatility = df.close.pct_change().rolling(10).std()
    momentum = compute_rsi(df.close)
    bb_width = df.BB_width
    slope = df.Slope
    features = pd.concat([returns, volatility, momentum, bb_width, slope], axis=1).dropna()
    return features

def train_hmm(df):
    X = build_hmm_features(df)
    hmm_model.fit(X)
    return X

HMM_STRATEGY_MAP = {i:"mean_reversion" if i in [0,1,9,10] else
                    "trend_follow" if i in [3,4,6,12,13] else
                    "breakout" if i in [5,8,14,15] else
                    "volatility_fade" if i in [2,11] else
                    "risk_off" if i in [16,17] else
                    "reversal" if i==18 else
                    "neutral" for i in range(20)}

def hmm_strategy(state):
    return HMM_STRATEGY_MAP.get(state,"neutral")

# -------------------------
# STRATEGY PROBABILITY FILTER WITH MULTI-STRATEGY CONSENSUS
# -------------------------
def score_strategy(regime, df):
    last = df.iloc[-1]
    scores = []
    # Mean-reversion high-win-rate
    if regime=="mean_reversion":
        scores.append(0.85 if last.RSI<40 else 0.85 if last.RSI>60 else 0.6)
    # Trend-follow low-volatility confirmation
    if regime=="trend_follow":
        scores.append(0.8 if last.ATR<df.ATR.mean() else 0.65)
    # Breakout volatility fade
    if regime=="breakout":
        scores.append(0.75)
    if regime=="volatility_fade":
        scores.append(0.7)
    if regime=="reversal":
        scores.append(0.7)
    if regime=="risk_off":
        scores.append(0)
    # Aggregate multi-strategy probability
    return max(scores)

# -------------------------
# EQUITY & POSITION SIZING WITH VOLATILITY ADJUSTMENT
# -------------------------
def get_equity():
    r = AccountDetails(accountID=ACCOUNT_ID)
    data = client.request(r)
    return float(data["account"]["NAV"])

def position_size(atr, price, equity, risk_pct=BASE_RISK_PCT):
    # scale risk inversely with volatility
    vol_factor = max(0.5, min(1.5, df.ATR.iloc[-1]/df.ATR.mean()))
    adjusted_risk = risk_pct / vol_factor
    risk_amount = equity * adjusted_risk
    units = int(risk_amount / atr)
    return units

# -------------------------
# ORDER EXECUTION
# -------------------------
def place_order(inst, units, sl, tp):
    data = {
        "order":{
            "instrument":inst,
            "units":str(units),
            "type":"MARKET",
            "positionFill":"DEFAULT",
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
        if other!=inst:
            if abs(corr.loc[inst,other])>0.8:
                return True
    return False

# -------------------------
# TRADE ENGINE WITH ALL ADVANCED FILTERS
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

        # Adaptive probability
        prob = score_strategy(regime, df)
        equity = get_equity()
        drawdown = 1 - (equity / max(equity, getattr(trade_engine,"peak_equity", equity)))
        trade_engine.peak_equity = max(getattr(trade_engine,"peak_equity", equity), equity)

        # Reduce probability if drawdown is high
        if drawdown>MAX_DRAWDOWN:
            prob *= 0.5

        if prob<BASE_PROB_THRESHOLD or regime=="risk_off":
            time.sleep(30)
            continue

        # Correlation control
        corr = correlation_matrix(price_history)
        if correlation_block(inst, corr):
            time.sleep(30)
            continue

        last = df.iloc[-1]
        atr = last.ATR
        price = last.close
        units = position_size(atr, price, equity)

        # Volatility-adjusted stops and take-profits
        stop = 1.0*atr
        take = 1.8*atr

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

if __name__=="__main__":
    run()
