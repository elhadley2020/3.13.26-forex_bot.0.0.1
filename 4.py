import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.positions as positions

import pandas as pd
import numpy as np
import ta
import time

# =========================
# CONFIG
# =========================
API_KEY = "YOUR_API_KEY"
ACCOUNT_ID = "YOUR_ACCOUNT_ID"

INSTRUMENTS = ["EUR_USD","GBP_USD","USD_JPY","AUD_USD"]
CANDLE_COUNT = 200

TARGET_VOL = 0.02        # target portfolio volatility
MAX_DRAWDOWN = 0.20      # max drawdown before halting trades

client = oandapyV20.API(access_token=API_KEY)
market_data = {i: pd.DataFrame() for i in INSTRUMENTS}
peak_equity = 0

# Track strategy performance for adaptive weighting
strategy_performance = {s: [] for s in [
    "trend_strategy","pullback_strategy","mean_reversion",
    "bb_mean_reversion","volatility_breakout","rsi_reversion",
    "momentum_strategy","breakout_mean_reversion","trend_following"
]}

# =========================
# DATA FUNCTIONS
# =========================
def get_candles(instrument, granularity, count=CANDLE_COUNT):
    params = {"count": count, "granularity": granularity, "price":"M"}
    r = instruments.InstrumentsCandles(instrument=instrument, params=params)
    client.request(r)
    data = []
    for c in r.response["candles"]:
        data.append({
            "open": float(c["mid"]["o"]),
            "high": float(c["mid"]["h"]),
            "low": float(c["mid"]["l"]),
            "close": float(c["mid"]["c"])
        })
    return pd.DataFrame(data)

def compute_features(df):
    df["ema20"] = ta.trend.ema_indicator(df["close"], 20)
    df["ema50"] = ta.trend.ema_indicator(df["close"], 50)
    df["rsi"] = ta.momentum.rsi(df["close"], 14)
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], 14)
    bb = ta.volatility.BollingerBands(df["close"])
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]
    return df

# =========================
# ACCOUNT & POSITIONS
# =========================
def get_account_balance():
    r = accounts.AccountSummary(ACCOUNT_ID)
    client.request(r)
    return float(r.response["account"]["NAV"])

def get_position(instrument):
    r = positions.PositionDetails(accountID=ACCOUNT_ID, instrument=instrument)
    client.request(r)
    pos = r.response["position"]
    return int(pos["long"]["units"]) - int(pos["short"]["units"])

# =========================
# STRATEGIES
# =========================
def trend_strategy(df):
    return 1 if df["ema20"].iloc[-1] > df["ema50"].iloc[-1] else -1 if df["ema20"].iloc[-1] < df["ema50"].iloc[-1] else 0

def pullback_strategy(df):
    price = df["close"].iloc[-1]; ema20 = df["ema20"].iloc[-1]; rsi = df["rsi"].iloc[-1]
    if price > ema20 and rsi < 40: return 1
    if price < ema20 and rsi > 60: return -1
    return 0

def mean_reversion(df):
    rsi = df["rsi"].iloc[-1]
    return 1 if rsi < 30 else -1 if rsi > 70 else 0

def bb_mean_reversion(df):
    price = df["close"].iloc[-1]
    if price < df["bb_lower"].iloc[-1]: return 1
    if price > df["bb_upper"].iloc[-1]: return -1
    return 0

def volatility_breakout(df):
    high = df["high"].rolling(30).max().iloc[-2]
    low = df["low"].rolling(30).min().iloc[-2]
    price = df["close"].iloc[-1]
    return 1 if price > high else -1 if price < low else 0

def rsi_reversion(df):
    if df["rsi"].iloc[-1] < 25: return 1
    if df["rsi"].iloc[-1] > 75: return -1
    return 0

def momentum_strategy(df):
    returns = df["close"].pct_change(3)
    if returns.iloc[-1] > 0.002: return 1
    if returns.iloc[-1] < -0.002: return -1
    return 0

def breakout_mean_reversion(df):
    high = df["high"].rolling(20).max().iloc[-2]
    low = df["low"].rolling(20).min().iloc[-2]
    price = df["close"].iloc[-1]
    if price > high * 1.01: return -1
    if price < low * 0.99: return 1
    return 0

def trend_following(df):
    slope = df["ema20"].iloc[-1] - df["ema20"].iloc[-5]
    if slope > 0: return 1
    if slope < 0: return -1
    return 0

# =========================
# REGIME DETECTION
# =========================
def classify_regime(df):
    ema20 = df["ema20"].iloc[-1]; ema50 = df["ema50"].iloc[-1]
    atr = df["atr"].iloc[-1]; atr_mean = df["atr"].rolling(50).mean().iloc[-1]
    bb_width = df["bb_width"].iloc[-1]; bb_mean = df["bb_width"].rolling(50).mean().iloc[-1]
    trend_strength = abs(ema20 - ema50)
    if trend_strength > 0.001 and atr > atr_mean: return "strong_trend"
    if trend_strength > 0.0004: return "weak_trend"
    if bb_width < bb_mean*0.7: return "compression"
    if atr > atr_mean*1.5: return "expansion"
    if df["rsi"].iloc[-1] > 80 or df["rsi"].iloc[-1] < 20: return "reversal"
    return "range"

def multi_timeframe_regime(instrument):
    df5 = compute_features(market_data[instrument])
    df1h = compute_features(get_candles(instrument,"H1"))
    dfd = compute_features(get_candles(instrument,"D"))
    r5 = classify_regime(df5)
    r1h = classify_regime(df1h)
    rd = classify_regime(dfd)
    if rd in ["strong_trend","weak_trend"]: return "strong_trend"
    if r1h == "compression": return "compression"
    if r5 == "range" and df5["rsi"].iloc[-1] > 70: return "reversal"
    if df5["ema20"].iloc[-1] - df5["ema50"].iloc[-1] > 0.002: return "momentum_surge"
    if abs(df5["ema20"].iloc[-1] - df5["ema50"].iloc[-1]) < 0.0003: return "sideways"
    return r5

REGIME_STRATEGIES = {
    "strong_trend": [trend_strategy,pullback_strategy,volatility_breakout,trend_following],
    "weak_trend": [trend_strategy,pullback_strategy],
    "range": [mean_reversion, bb_mean_reversion, rsi_reversion],
    "compression": [volatility_breakout, breakout_mean_reversion],
    "expansion": [volatility_breakout, trend_following],
    "reversal": [mean_reversion, rsi_reversion],
    "momentum_surge": [momentum_strategy, volatility_breakout],
    "sideways": [bb_mean_reversion, rsi_reversion]
}

# =========================
# ADAPTIVE STRATEGY WEIGHTS
# =========================
def update_strategy_perf(strategy, pnl):
    history = strategy_performance[strategy]
    history.append(pnl)
    if len(history) > 50: history.pop(0)

def get_strategy_weight(name):
    history = strategy_performance[name]
    if len(history) < 10: return 1
    wins = sum([1 for x in history if x > 0])
    win_ratio = wins / len(history)
    weight = 0.5 + 2.5 * win_ratio
    return min(max(weight, 0.5), 3)

# =========================
# POSITION SIZING
# =========================
def portfolio_volatility():
    vols = []
    for pair in INSTRUMENTS:
        df = market_data[pair]
        returns = np.log(df["close"]/df["close"].shift())
        vols.append(returns.std())
    return np.mean(vols)

def position_size(balance, atr):
    risk = balance*0.01
    port_vol = portfolio_volatility()
    vol_scale = TARGET_VOL/(port_vol+1e-6)
    stop_distance = atr*2
    units = (risk/stop_distance)*vol_scale
    return int(units)

# =========================
# ENHANCED VOTING ENGINE
# =========================
MIN_STRATEGY_CONFIRM = 2
MIN_CONFIDENCE = 2.0
EMA_SLOPE_THRESHOLD = 0.002
MIN_ATR = 0.0005

def enhanced_voting_engine(df, regime):
    strategies = REGIME_STRATEGIES.get(regime, [])
    votes = []
    weights = []

    for strat in strategies:
        vote = strat(df)
        weight = get_strategy_weight(strat.__name__)
        votes.append(vote)
        weights.append(weight)

    votes = np.array(votes)
    weights = np.array(weights)

    if np.sum(votes > 0) >= MIN_STRATEGY_CONFIRM:
        direction = 1
    elif np.sum(votes < 0) >= MIN_STRATEGY_CONFIRM:
        direction = -1
    else:
        direction = 0

    weighted_score = np.dot(votes, weights)

    ema_slope = df["ema20"].iloc[-1] - df["ema20"].iloc[-5]
    if abs(ema_slope) < EMA_SLOPE_THRESHOLD and direction != 0:
        direction = 0

    atr = df["atr"].iloc[-1]
    if atr < MIN_ATR:
        direction = 0

    if abs(weighted_score) < MIN_CONFIDENCE:
        return "HOLD"

    if direction == 1: return "BUY"
    if direction == -1: return "SELL"
    return "HOLD"

# =========================
# MAX DRAWDOWN
# =========================
def drawdown_allowed():
    global peak_equity
    balance = get_account_balance()
    if balance > peak_equity: peak_equity = balance
    drawdown = (peak_equity - balance)/peak_equity
    if drawdown > MAX_DRAWDOWN:
        print("MAX DRAWDOWN HIT — TRADING DISABLED")
        return False
    return True

# =========================
# SMART ORDER EXECUTION
# =========================
def get_price(instrument):
    params = {"instruments":instrument}
    r = pricing.PricingInfo(accountID=ACCOUNT_ID, params=params)
    client.request(r)
    p = r.response["prices"][0]
    bid = float(p["bids"][0]["price"])
    ask = float(p["asks"][0]["price"])
    return bid, ask

def place_order(instrument, units, atr):
    bid, ask = get_price(instrument)
    price = ask if units>0 else bid
    stop = atr*2
    tp = atr*3
    sl_price = price-stop if units>0 else price+stop
    tp_price = price+tp if units>0 else price-tp
    data = {
        "order":{
            "instrument":instrument,
            "units":str(units),
            "type":"LIMIT",
            "price":str(round(price,5)),
            "timeInForce":"FOK",
            "stopLossOnFill":{"price":str(round(sl_price,5))},
            "takeProfitOnFill":{"price":str(round(tp_price,5))}
        }
    }
    r = orders.OrderCreate(ACCOUNT_ID, data=data)
    client.request(r)

# =========================
# MARKET DATA UPDATE
# =========================
def update_market_data(instrument, price):
    df = market_data[instrument]
    new_row = {"open": price,"high": price,"low": price,"close": price}
    df = df.append(new_row, ignore_index=True)
    market_data[instrument] = df.tail(CANDLE_COUNT)

# =========================
# PROCESS INSTRUMENT
# =========================
active_trades = {}

def process_instrument_enhanced(instrument):
    if not drawdown_allowed(): return
    df = market_data[instrument]
    df = compute_features(df)
    regime = multi_timeframe_regime(instrument)
    signal = enhanced_voting_engine(df, regime)
    atr = df["atr"].iloc[-1]
    balance = get_account_balance()
    units = position_size(balance, atr)
    current_position = get_position(instrument)

    if instrument in active_trades and active_trades[instrument] != 0:
        if (signal == "BUY" and active_trades[instrument] > 0) or \
           (signal == "SELL" and active_trades[instrument] < 0):
            return

    if signal == "BUY" and current_position <= 0:
        place_order(instrument, units, atr)
        active_trades[instrument] = units
    elif signal == "SELL" and current_position >= 0:
        place_order(instrument, -units, atr)
        active_trades[instrument] = -units
    else:
        active_trades[instrument] = 0

    print(instrument, regime, signal, "Units:", units, "Position:", current_position)

# =========================
# INITIAL DATA LOAD
# =========================
for pair in INSTRUMENTS:
    market_data[pair] = compute_features(get_candles(pair,"M5"))

# =========================
# LIVE STREAM
# =========================
params = {"instruments": ",".join(INSTRUMENTS)}
r = pricing.PricingStream(accountID=ACCOUNT_ID, params=params)

for tick in client.request(r):
    if tick["type"] != "PRICE": continue
    instrument = tick["instrument"]
    price = float(tick["bids"][0]["price"])
    update_market_data(instrument, price)
    process_instrument_enhanced(instrument)
