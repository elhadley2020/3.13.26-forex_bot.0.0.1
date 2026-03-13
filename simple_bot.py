import oandapyV20
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.instruments as instruments

import pandas as pd
import numpy as np
import ta
import time

# =========================
# CONFIG
# =========================

API_KEY = "YOUR_API_KEY"
ACCOUNT_ID = "YOUR_ACCOUNT_ID"

INSTRUMENTS = [
    "EUR_USD",
    "GBP_USD",
    "USD_JPY",
    "AUD_USD"
]

GRANULARITY = "M5"
CANDLE_COUNT = 200

client = oandapyV20.API(access_token=API_KEY)

market_data = {i: pd.DataFrame() for i in INSTRUMENTS}

account_balance = 10000


# =========================
# DATA
# =========================

def get_candles(instrument):

    params = {
        "count": CANDLE_COUNT,
        "granularity": GRANULARITY,
        "price": "M"
    }

    r = instruments.InstrumentsCandles(instrument=instrument, params=params)
    client.request(r)

    data = []

    for c in r.response["candles"]:
        data.append({
            "time": c["time"],
            "open": float(c["mid"]["o"]),
            "high": float(c["mid"]["h"]),
            "low": float(c["mid"]["l"]),
            "close": float(c["mid"]["c"])
        })

    df = pd.DataFrame(data)
    return df


# =========================
# FEATURE ENGINEERING
# =========================

def compute_features(df):

    df["ema20"] = ta.trend.ema_indicator(df["close"], 20)
    df["ema50"] = ta.trend.ema_indicator(df["close"], 50)

    df["rsi"] = ta.momentum.rsi(df["close"], 14)

    df["atr"] = ta.volatility.average_true_range(
        df["high"], df["low"], df["close"], 14
    )

    bb = ta.volatility.BollingerBands(df["close"])

    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()

    df["bb_width"] = df["bb_upper"] - df["bb_lower"]

    return df


# =========================
# REGIME CLASSIFIER
# =========================

def classify_regime(df):

    ema20 = df["ema20"].iloc[-1]
    ema50 = df["ema50"].iloc[-1]

    atr = df["atr"].iloc[-1]
    atr_mean = df["atr"].rolling(50).mean().iloc[-1]

    bb_width = df["bb_width"].iloc[-1]
    bb_mean = df["bb_width"].rolling(50).mean().iloc[-1]

    trend_strength = abs(ema20 - ema50)

    if trend_strength > 0.001 and atr > atr_mean:
        return "strong_trend"

    if trend_strength > 0.0004:
        return "weak_trend"

    if bb_width < bb_mean * 0.7:
        return "compression"

    if atr > atr_mean * 1.5:
        return "expansion"

    if df["rsi"].iloc[-1] > 80 or df["rsi"].iloc[-1] < 20:
        return "reversal"

    return "range"


# =========================
# STRATEGIES
# =========================

def trend_strategy(df):

    if df["ema20"].iloc[-1] > df["ema50"].iloc[-1]:
        return 1

    if df["ema20"].iloc[-1] < df["ema50"].iloc[-1]:
        return -1

    return 0


def pullback_strategy(df):

    price = df["close"].iloc[-1]
    ema20 = df["ema20"].iloc[-1]

    if price > ema20 and df["rsi"].iloc[-1] < 40:
        return 1

    if price < ema20 and df["rsi"].iloc[-1] > 60:
        return -1

    return 0


def mean_reversion(df):

    rsi = df["rsi"].iloc[-1]

    if rsi < 30:
        return 1

    if rsi > 70:
        return -1

    return 0


def bb_mean_reversion(df):

    price = df["close"].iloc[-1]

    if price < df["bb_lower"].iloc[-1]:
        return 1

    if price > df["bb_upper"].iloc[-1]:
        return -1

    return 0


def volatility_breakout(df):

    high = df["high"].rolling(30).max().iloc[-2]
    low = df["low"].rolling(30).min().iloc[-2]

    price = df["close"].iloc[-1]

    if price > high:
        return 1

    if price < low:
        return -1

    return 0


# =========================
# REGIME STRATEGY MAP
# =========================

REGIME_STRATEGIES = {

    "strong_trend": [
        trend_strategy,
        pullback_strategy,
        volatility_breakout
    ],

    "weak_trend": [
        trend_strategy
    ],

    "range": [
        mean_reversion,
        bb_mean_reversion
    ],

    "compression": [
        volatility_breakout
    ],

    "expansion": [
        volatility_breakout
    ],

    "reversal": [
        mean_reversion
    ]
}


# =========================
# STRATEGY WEIGHTS
# =========================

STRATEGY_WEIGHTS = {

    "trend_strategy": 2,
    "pullback_strategy": 1.5,
    "mean_reversion": 1,
    "bb_mean_reversion": 1,
    "volatility_breakout": 2
}


# =========================
# VOTING ENGINE
# =========================

def voting_engine(df, regime):

    strategies = REGIME_STRATEGIES[regime]

    score = 0

    for strat in strategies:

        vote = strat(df)

        weight = STRATEGY_WEIGHTS.get(strat.__name__, 1)

        score += vote * weight

    if score > 1:
        return "BUY"

    if score < -1:
        return "SELL"

    return "HOLD"


# =========================
# RISK MANAGEMENT
# =========================

def position_size(balance, atr):

    risk_per_trade = 0.01

    stop_distance = atr * 2

    units = (balance * risk_per_trade) / stop_distance

    return int(units)


# =========================
# EXECUTION
# =========================

def place_order(instrument, units):

    data = {
        "order": {
            "instrument": instrument,
            "units": str(units),
            "type": "MARKET",
            "positionFill": "DEFAULT"
        }
    }

    r = orders.OrderCreate(ACCOUNT_ID, data=data)

    client.request(r)


# =========================
# SIGNAL PIPELINE
# =========================

def process_instrument(instrument):

    global market_data

    df = market_data[instrument]

    df = compute_features(df)

    regime = classify_regime(df)

    signal = voting_engine(df, regime)

    atr = df["atr"].iloc[-1]

    units = position_size(account_balance, atr)

    print(instrument, regime, signal)

    if signal == "BUY":
        place_order(instrument, units)

    if signal == "SELL":
        place_order(instrument, -units)


# =========================
# INITIAL DATA LOAD
# =========================

for pair in INSTRUMENTS:
    market_data[pair] = compute_features(get_candles(pair))


# =========================
# LIVE STREAM
# =========================

params = {
    "instruments": ",".join(INSTRUMENTS)
}

r = pricing.PricingStream(accountID=ACCOUNT_ID, params=params)

for tick in client.request(r):

    if tick["type"] != "PRICE":
        continue

    instrument = tick["instrument"]
    price = float(tick["bids"][0]["price"])

    df = market_data[instrument]

    new_row = {
        "time": time.time(),
        "open": price,
        "high": price,
        "low": price,
        "close": price
    }

    df = df.append(new_row, ignore_index=True)

    market_data[instrument] = df.tail(CANDLE_COUNT)

    process_instrument(instrument)
