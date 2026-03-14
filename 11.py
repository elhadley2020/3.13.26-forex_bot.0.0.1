import pandas as pd
import numpy as np
import threading
import time
from datetime import datetime

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
BASE_RISK_PCT = 0.01
BASE_PROB_THRESHOLD = 0.75
MAX_DRAWDOWN = 0.1

# -------------------------
# DATA FETCHING
# -------------------------
def fetch_candles(inst):
    params = {"granularity":"H1","count":200}
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
# STRATEGY PROBABILITY FILTER
# -------------------------
def score_strategy(regime, df):
    last = df.iloc[-1]
    scores=[]
    if regime=="mean_reversion":
        scores.append(0.85 if last.RSI<40 else 0.85 if last.RSI>60 else 0.6)
    if regime=="trend_follow":
        scores.append(0.8 if last.ATR<df.ATR.mean() else 0.65)
    if regime=="breakout":
        scores.append(0.75)
    if regime=="volatility_fade":
        scores.append(0.7)
    if regime=="reversal":
        scores.append(0.7)
    if regime=="risk_off":
        scores.append(0)
    return max(scores)

# -------------------------
# EQUITY & POSITION SIZING
# -------------------------
def get_equity():
    r = AccountDetails(accountID=ACCOUNT_ID)
    data = client.request(r)
    return float(data["account"]["NAV"])

def position_size(atr, price, equity, prob, df):
    scale = 1.0
    if prob>0.85: scale=1.5
    elif prob<0.8: scale=0.7
    vol_factor = max(0.5,min(1.5,df.ATR.iloc[-1]/df.ATR.mean()))
    adjusted_risk = BASE_RISK_PCT * scale / vol_factor
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
    r = orders.OrderCreate(accountID=ACCOUNT_ID,data=data)
    client.request(r)

# -------------------------
# CORRELATION FILTER
# -------------------------
def correlation_matrix(price_history):
    symbols=list(price_history.keys())
    corr=pd.DataFrame(index=symbols,columns=symbols)
    for i in symbols:
        for j in symbols:
            corr.loc[i,j]=price_history[i].close.corr(price_history[j].close)
    return corr

def correlation_block(inst,corr):
    for other in corr.columns:
        if other!=inst and abs(corr.loc[inst,other])>0.8:
            return True
    return False

# -------------------------
# SESSION / TIME FILTER
# -------------------------
def in_trading_session():
    hour=datetime.utcnow().hour
    return 7<=hour<=16  # London + NY overlap

# -------------------------
# TRADE ENGINE WITH TRAILING STOPS & PARTIAL PROFIT
# -------------------------
def trade_engine(inst, price_history):
    open_trades = {}  # track active trades per instrument
    while True:
        if not in_trading_session():
            time.sleep(60)
            continue

        df = fetch_candles(inst)
        df = compute_indicators(df)
        price_history[inst] = df

        try:
            state = hmm_model.predict(build_hmm_features(df))[-1]
            regime = hmm_strategy(state)
        except:
            regime = "neutral"

        prob = score_strategy(regime, df)
        equity = get_equity()
        drawdown = 1 - (equity / max(getattr(trade_engine, "peak_equity", equity), equity))
        trade_engine.peak_equity = max(getattr(trade_engine, "peak_equity", equity), equity)
        if drawdown > MAX_DRAWDOWN: prob *= 0.5
        if prob < BASE_PROB_THRESHOLD or regime == "risk_off":
            time.sleep(30)
            continue

        # correlation filter
        corr = correlation_matrix(price_history)
        if correlation_block(inst, corr):
            time.sleep(30)
            continue

        # cross-instrument confirmation
        aligned = 0
        for other in INSTRUMENTS:
            if other == inst: continue
            if price_history[other].Slope.iloc[-1] * df.Slope.iloc[-1] > 0:
                aligned += 1
        if aligned < 1:
            time.sleep(30)
            continue

        last = df.iloc[-1]
        atr = last.ATR
        price = last.close
        units = position_size(atr, price, equity, prob, df)

        # regime-specific stop/take
        stop = 1.0*atr if "mean_reversion" in regime else 1.5*atr
        take = 1.8*atr if "mean_reversion" in regime else 2.5*atr

        long_trade = True
        if "down" in regime:
            long_trade = False
            units = -units

        sl = price - stop if long_trade else price + stop
        tp = price + take if long_trade else price - take
        place_order(inst, units, sl, tp)

        # Save trade state for trailing/partial profit
        open_trades[inst] = {
            "units": units,
            "initial_tp": tp,
            "sl": sl,
            "long": long_trade,
            "partial_closed": False
        }

        # Monitor trailing stop / partial profit
        while inst in open_trades:
            trade = open_trades[inst]
            current_price = fetch_candles(inst).close.iloc[-1]

            # Partial profit at initial TP
            if not trade["partial_closed"]:
                if (trade["long"] and current_price >= trade["initial_tp"]) or \
                   (not trade["long"] and current_price <= trade["initial_tp"]):
                    partial_units = int(trade["units"] * 0.5)
                    new_units = trade["units"] - partial_units
                    place_order(inst, partial_units, current_price, current_price)
                    trade["units"] = new_units
                    trade["partial_closed"] = True
                    trade["sl"] = current_price - 0.8*atr if trade["long"] else current_price + 0.8*atr

            # Trailing stop adjustment
            if trade["long"]:
                new_sl = max(trade["sl"], current_price - 0.8*atr)
                trade["sl"] = new_sl
            else:
                new_sl = min(trade["sl"], current_price + 0.8*atr)
                trade["sl"] = new_sl

            # Check stop hit
            if (trade["long"] and current_price <= trade["sl"]) or \
               (not trade["long"] and current_price >= trade["sl"]):
                place_order(inst, trade["units"], current_price, current_price)
                del open_trades[inst]
                break

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
    threads=[]
    for inst in INSTRUMENTS:
        t=threading.Thread(target=trade_engine,args=(inst,price_history))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

if __name__=="__main__":
    run()
