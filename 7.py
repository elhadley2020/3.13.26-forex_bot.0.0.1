import pandas as pd
import numpy as np
import threading
import time

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

import oandapyV20
import oandapyV20.endpoints.orders as orders
from oandapyV20.endpoints.accounts import AccountDetails
from oandapyV20.endpoints.instruments import InstrumentsCandles

API_KEY = "YOUR_API_KEY"
ACCOUNT_ID = "YOUR_ACCOUNT_ID"

client = oandapyV20.API(access_token=API_KEY)

INSTRUMENTS = [
"EUR_USD",
"GBP_USD",
"AUD_USD",
"USD_JPY",
"USD_CHF",
"NZD_USD"
]

RISK_PCT = 0.01

model = RandomForestClassifier(n_estimators=200)

# -----------------------
# DATA
# -----------------------

def fetch_candles(inst):

    params = {"granularity":"H1","count":200}

    r = InstrumentsCandles(instrument=inst,params=params)

    data = client.request(r)

    df = pd.DataFrame([{
        "time":c["time"],
        "open":float(c["mid"]["o"]),
        "high":float(c["mid"]["h"]),
        "low":float(c["mid"]["l"]),
        "close":float(c["mid"]["c"])
    } for c in data["candles"] if c["complete"]])

    return df

# -----------------------
# INDICATORS
# -----------------------

def compute_rsi(series,period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain/(avg_loss+1e-9)

    return 100-(100/(1+rs))

def compute_atr(df,period=14):

    tr1 = df.high-df.low
    tr2 = abs(df.high-df.close.shift())
    tr3 = abs(df.low-df.close.shift())

    tr = np.maximum.reduce([tr1,tr2,tr3])

    return pd.Series(tr).rolling(period).mean()

def compute_adx(df,period=14):

    plus_dm = df.high.diff()
    minus_dm = df.low.diff()*-1

    plus_dm = np.where((plus_dm>minus_dm)&(plus_dm>0),plus_dm,0)
    minus_dm = np.where((minus_dm>plus_dm)&(minus_dm>0),minus_dm,0)

    tr = compute_atr(df,1)

    plus_di = 100*pd.Series(plus_dm).rolling(period).mean()/tr
    minus_di = 100*pd.Series(minus_dm).rolling(period).mean()/tr

    dx = abs(plus_di-minus_di)/(plus_di+minus_di)*100

    return dx.rolling(period).mean()

def compute_hurst(series):

    lags = range(2,20)
    tau = [np.std(series[lag:]-series[:-lag]) for lag in lags]

    poly = np.polyfit(np.log(lags),np.log(tau),1)

    return poly[0]*2

def compute_indicators(df):

    df["SMA20"]=df.close.rolling(20).mean()
    df["SMA50"]=df.close.rolling(50).mean()

    df["RSI"]=compute_rsi(df.close)

    df["ATR"]=compute_atr(df)

    df["ADX"]=compute_adx(df)

    df["STD"]=df.close.rolling(20).std()

    df["UpperBB"]=df.SMA20+2*df.STD
    df["LowerBB"]=df.SMA20-2*df.STD

    df["BB_width"]=df.UpperBB-df.LowerBB

    df["HURST"]=compute_hurst(df.close)

    return df

# -----------------------
# REGIME
# -----------------------

def detect_regime(df):

    last=df.iloc[-1]

    atr_avg=df.ATR.rolling(50).mean().iloc[-1]

    if last.ADX<20 and last.HURST<0.45:
        return "mean_reversion"

    if last.ADX>25 and last.SMA20>last.SMA50:
        return "trend_up"

    if last.ADX>25 and last.SMA20<last.SMA50:
        return "trend_down"

    if last.BB_width<df.BB_width.rolling(50).mean().iloc[-1]:
        return "compression"

    if last.ATR>1.8*atr_avg:
        return "volatility_expansion"

    return "neutral"

# -----------------------
# VOL CLUSTER
# -----------------------

def volatility_cluster(df):

    vol=df["ATR"].values.reshape(-1,1)

    model=KMeans(n_clusters=3)

    model.fit(vol)

    return model.labels_[-1]

# -----------------------
# FEATURES
# -----------------------

def build_features(df):

    last=df.iloc[-1]

    features=[

        last.ADX,
        last.RSI,
        last.ATR,
        last.BB_width,
        last.HURST,
        last.SMA20-last.SMA50

    ]

    return np.array(features).reshape(1,-1)

# -----------------------
# ML
# -----------------------

def predict_probability(features):

    try:
        p=model.predict_proba(features)[0][1]
    except:
        p=0.5

    return p

# -----------------------
# CORRELATION
# -----------------------

def correlation_matrix(price_history):

    symbols=list(price_history.keys())

    corr=pd.DataFrame(index=symbols,columns=symbols)

    for i in symbols:
        for j in symbols:

            corr.loc[i,j]=price_history[i].close.corr(price_history[j].close)

    return corr

def correlation_block(inst,corr):

    for other in corr.columns:

        if other!=inst:

            if abs(corr.loc[inst,other])>0.8:
                return True

    return False

# -----------------------
# ACCOUNT
# -----------------------

def get_equity():

    r=AccountDetails(accountID=ACCOUNT_ID)

    data=client.request(r)

    return float(data["account"]["NAV"])

def position_size(atr):

    equity=get_equity()

    risk=equity*RISK_PCT

    units=int(risk/atr)

    return units

# -----------------------
# EXECUTION
# -----------------------

def place_order(inst,units,sl,tp):

    data={
    "order":{
    "instrument":inst,
    "units":str(units),
    "type":"MARKET",
    "positionFill":"DEFAULT",
    "stopLossOnFill":{"price":str(sl)},
    "takeProfitOnFill":{"price":str(tp)}
    }}

    r=orders.OrderCreate(accountID=ACCOUNT_ID,data=data)

    client.request(r)

# -----------------------
# TRADE ENGINE
# -----------------------

def trade_engine(inst,price_history):

    while True:

        df=fetch_candles(inst)

        df=compute_indicators(df)

        price_history[inst]=df

        corr=correlation_matrix(price_history)

        regime=detect_regime(df)

        vol_regime=volatility_cluster(df)

        features=build_features(df)

        prob=predict_probability(features)

        if prob<0.7:
            time.sleep(30)
            continue

        if correlation_block(inst,corr):
            time.sleep(30)
            continue

        last=df.iloc[-1]

        atr=last.ATR

        units=position_size(atr)

        stop=1.2*atr
        tp=2.5*atr

        price=last.close

        if regime=="trend_down":

            units=-units

            sl=price+stop
            take=price-tp

        else:

            sl=price-stop
            take=price+tp

        place_order(inst,units,sl,take)

        print(inst,regime,prob)

        time.sleep(30)

# -----------------------
# PORTFOLIO
# -----------------------

def run():

    price_history={}

    threads=[]

    for inst in INSTRUMENTS:

        t=threading.Thread(target=trade_engine,args=(inst,price_history))

        t.start()

        threads.append(t)

    for t in threads:
        t.join()

if __name__=="__main__":

    run()
