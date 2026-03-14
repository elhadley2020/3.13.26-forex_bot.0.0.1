import pandas as pd
import numpy as np
import oandapyV20
from oandapyV20 import API
import oandapyV20.endpoints.orders as orders
from oandapyV20.endpoints.accounts import AccountDetails
from websocket import create_connection
import json
import time
import threading

# ----------------------------
# OANDA API Setup
# ----------------------------
API_KEY = "YOUR_OANDA_API_KEY"
ACCOUNT_ID = "YOUR_OANDA_ACCOUNT_ID"
INSTRUMENTS = ["EUR_USD", "GBP_USD", "USD_JPY"]
client = API(access_token=API_KEY)

# ----------------------------
# Parameters
# ----------------------------
RISK_PCT = 0.01  # Risk per trade (% of account equity)
ATR_PERIOD = 14
SMA_SHORT = 20
SMA_LONG = 50
BB_STD = 2

# ----------------------------
# Regime → Strategy Mapping
# ----------------------------
REGIME_STRATEGY_MAP = {
    "range_sideways": ["bollinger_mean_reversion", "range_scalping"],
    "low_volatility": ["vwap_reversion", "moving_average_pullback"],
    "overbought": ["rsi_mean_reversion_sell"],
    "oversold": ["rsi_mean_reversion_buy"],
    "stable_correlation": ["pairs_trading"],
    "liquidity_rich": ["market_making"],
    "post_news": ["news_spike_fade"],
    "trend_up_exhaustion": ["divergence_reversal", "trend_pullback_buy"],
    "trend_down_exhaustion": ["divergence_reversal", "trend_pullback_sell"],
    "volatility_spike": ["volatility_fade"]
}

# ----------------------------
# Fetch Candles
# ----------------------------
def fetch_candles(instrument, granularity="H1", count=100):
    from oandapyV20.endpoints.instruments import InstrumentsCandles
    params = {"granularity": granularity, "count": count}
    r = InstrumentsCandles(instrument=instrument, params=params)
    data = client.request(r)
    df = pd.DataFrame([{
        "time": c['time'],
        "open": float(c['mid']['o']),
        "high": float(c['mid']['h']),
        "low": float(c['mid']['l']),
        "close": float(c['mid']['c'])
    } for c in data['candles'] if c['complete']])
    return df

# ----------------------------
# Indicators
# ----------------------------
def compute_indicators(df):
    df['SMA20'] = df['close'].rolling(SMA_SHORT).mean()
    df['SMA50'] = df['close'].rolling(SMA_LONG).mean()
    df['Std20'] = df['close'].rolling(SMA_SHORT).std()
    df['UpperBB'] = df['SMA20'] + BB_STD * df['Std20']
    df['LowerBB'] = df['SMA20'] - BB_STD * df['Std20']
    df['RSI'] = compute_rsi(df['close'], 14)
    df['ATR'] = compute_atr(df, ATR_PERIOD)
    return df

def compute_rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_atr(df, period):
    df['H-L'] = df['high'] - df['low']
    df['H-Cp'] = abs(df['high'] - df['close'].shift(1))
    df['L-Cp'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L','H-Cp','L-Cp']].max(axis=1)
    atr = df['TR'].rolling(period).mean()
    return atr

# ----------------------------
# Regime Detection
# ----------------------------
def detect_regime(df):
    last = df.iloc[-1]
    if last['close'] < last['UpperBB'] and last['close'] > last['LowerBB']:
        return "range_sideways"
    elif last['ATR'] < df['ATR'].rolling(50).mean().iloc[-1]:
        return "low_volatility"
    elif last['RSI'] > 70:
        return "overbought"
    elif last['RSI'] < 30:
        return "oversold"
    elif last['SMA20'] > last['SMA50'] * 1.01:
        return "trend_up_exhaustion"
    elif last['SMA20'] < last['SMA50'] * 0.99:
        return "trend_down_exhaustion"
    elif last['ATR'] > 2 * df['ATR'].rolling(50).mean().iloc[-1]:
        return "volatility_spike"
    else:
        return "unknown"

# ----------------------------
# Strategy Scoring (0–1)
# ----------------------------
def score_strategy(strategy, df):
    last = df.iloc[-1]
    score = 0.0
    atr_mean = df['ATR'].rolling(50).mean().iloc[-1]

    if strategy == "bollinger_mean_reversion":
        score = max(0, 1 - abs((last['close'] - last['SMA20']) / last['Std20']))
    elif strategy == "range_scalping":
        score = 0.5
    elif strategy == "vwap_reversion":
        score = max(0, 1 - abs(last['close'] - last['SMA20']) / last['Std20'])
    elif strategy == "moving_average_pullback":
        score = max(0, 1 - abs(last['close'] - last['SMA20']) / last['SMA20'])
    elif strategy == "rsi_mean_reversion_buy":
        score = max(0, (30 - last['RSI']) / 30 if last['RSI'] < 30 else 0)
    elif strategy == "rsi_mean_reversion_sell":
        score = max(0, (last['RSI'] - 70) / 30 if last['RSI'] > 70 else 0)
    elif strategy == "pairs_trading":
        score = 0.6
    elif strategy == "market_making":
        score = 0.8
    elif strategy == "news_spike_fade":
        score = 0.7
    elif strategy == "divergence_reversal":
        score = 0.7 if last['RSI'] > 70 or last['RSI'] < 30 else 0.3
    elif strategy == "trend_pullback_buy":
        score = 0.7 if last['SMA20'] > last['SMA50'] else 0.3
    elif strategy == "trend_pullback_sell":
        score = 0.7 if last['SMA20'] < last['SMA50'] else 0.3
    elif strategy == "volatility_fade":
        score = 0.7 if last['ATR'] > 2*atr_mean else 0.3

    return score

# ----------------------------
# Strategy Selection
# ----------------------------
def select_strategy(df, regime):
    strategies = REGIME_STRATEGY_MAP.get(regime, [])
    if not strategies:
        return None
    scores = {s: score_strategy(s, df) for s in strategies}
    selected = max(scores, key=scores.get)
    if scores[selected] < 0.5:
        return None
    return selected

# ----------------------------
# Live Account Equity
# ----------------------------
def get_account_equity():
    r = AccountDetails(accountID=ACCOUNT_ID)
    resp = client.request(r)
    equity = float(resp['account']['NAV'])
    return equity

def calculate_units_live(risk_pct, atr):
    account_equity = get_account_equity()
    risk_amount = account_equity * risk_pct
    units = int(risk_amount / atr)
    return units

# ----------------------------
# Place Trade
# ----------------------------
def place_order(instrument, units, take_profit, stop_loss):
    data = {
        "order": {
            "instrument": instrument,
            "units": str(units),
            "type": "MARKET",
            "positionFill": "DEFAULT",
            "takeProfitOnFill": {"price": str(take_profit)},
            "stopLossOnFill": {"price": str(stop_loss)}
        }
    }
    r = orders.OrderCreate(accountID=ACCOUNT_ID, data=data)
    client.request(r)
    print(f"Order placed: {units} units of {instrument}")

# ----------------------------
# Concurrent Multi-Instrument Streaming
# ----------------------------
def stream_instrument_live(instrument):
    ws_url = f"wss://stream-fxpractice.oanda.com/v3/accounts/{ACCOUNT_ID}/pricing/stream?instruments={instrument}"
    ws = create_connection(ws_url, header=[f"Authorization: Bearer {API_KEY}"])
    
    while True:
        try:
            result = ws.recv()
            data = json.loads(result)
            if "bids" in data:
                price = float(data['bids'][0]['price'])
                df = fetch_candles(instrument)
                df = compute_indicators(df)
                regime = detect_regime(df)
                strategy = select_strategy(df, regime)

                if strategy:
                    atr = df['ATR'].iloc[-1]
                    units = calculate_units_live(RISK_PCT, atr)
                    take_profit = price + 2*atr
                    stop_loss = price - atr
                    if "sell" in strategy or "reversion_sell" in strategy:
                        units = -units
                        take_profit, stop_loss = price - 2*atr, price + atr

                    place_order(instrument, units, take_profit, stop_loss)
                    current_equity = get_account_equity()
                    print(f"{instrument} | Regime: {regime}, Strategy: {strategy}, Price: {price}, Equity: {current_equity:.2f}")
                else:
                    print(f"{instrument} | No strong trade signal. Regime: {regime}")

        except Exception as e:
            print(f"{instrument} Error:", e)
            time.sleep(1)

def start_concurrent_streams():
    threads = []
    for inst in INSTRUMENTS:
        t = threading.Thread(target=stream_instrument_live, args=(inst,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

# ----------------------------
# Run Bot
# ----------------------------
if __name__ == "__main__":
    start_concurrent_streams()
