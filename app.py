import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pyotp
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MCX & Crypto Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 700; }
    .signal-box {
        border-radius: 10px; padding: 14px 18px; margin-top: 4px;
        font-size: 1rem; font-weight: 600;
    }
    .buy     { background: #0d2b1a; color: #00e676; border: 1px solid #00e676; }
    .sell    { background: #2b0d0d; color: #ff5252; border: 1px solid #ff5252; }
    .neutral { background: #1e1b0d; color: #ffd700; border: 1px solid #ffd700; }
    .alert-banner {
        background: linear-gradient(135deg, #0d2b1a, #002a10);
        border: 2px solid #00e676;
        border-radius: 14px;
        padding: 20px 24px;
        margin: 12px 0;
    }
    .alert-banner h2 { color: #00e676; margin: 0 0 6px; font-size: 1.5rem; }
    .alert-banner p  { color: #c9d1d9; margin: 4px 0 0; font-size: 0.95rem; }
    .waiting-banner {
        background: #1c1f26; border: 1px solid #333; border-radius: 10px;
        padding: 16px 20px; margin: 8px 0;
    }
    .tf-card {
        background: #1c1f26; border-radius: 10px;
        padding: 14px 18px; margin-bottom: 10px;
        border-left: 4px solid;
    }
    .tf-pass { border-color: #00e676; }
    .tf-fail { border-color: #ff5252; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
BASE_URL   = "https://apiconnect.angelone.in"
ADX_STRONG = 25

INTERVAL_MAP = {
    "5 min":  ("FIVE_MINUTE",    30),
    "15 min": ("FIFTEEN_MINUTE", 60),
    "1 hour": ("ONE_HOUR",      180),
    "1 day":  ("ONE_DAY",       365),
}

INSTRUMENT_SEARCH = {
    "MCX Gold":        ("MCX", "GOLD"),
    "MCX Silver":      ("MCX", "SILVER"),
    "MCX Gold Mini":   ("MCX", "GOLDM"),
    "MCX Silver Mini": ("MCX", "SILVERMI"),
}

# ─────────────────────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────────────────────

def calc_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()


def calc_adx(df, period=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr   = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    up   = h.diff(); down = -l.diff()
    pdm  = pd.Series(np.where((up>down)&(up>0), up, 0.0), index=df.index)
    mdm  = pd.Series(np.where((down>up)&(down>0), down, 0.0), index=df.index)
    a    = 1/period
    tr14 = tr.ewm(alpha=a,adjust=False).mean()
    pdm14= pdm.ewm(alpha=a,adjust=False).mean()
    mdm14= mdm.ewm(alpha=a,adjust=False).mean()
    pdi  = 100*pdm14/tr14
    mdi  = 100*mdm14/tr14
    dx   = 100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    adx  = dx.ewm(alpha=a,adjust=False).mean()
    return adx.round(2), pdi.round(2), mdi.round(2)


def calc_supertrend(df, period=10, multiplier=3.0):
    h,l,c = df["high"].values, df["low"].values, df["close"].values
    atr = calc_atr(df["high"],df["low"],df["close"],period).values
    hl2 = (h+l)/2
    ub  = hl2+multiplier*atr;  lb = hl2-multiplier*atr
    fub = ub.copy();  flb = lb.copy()
    stl = np.zeros(len(df));   dr = np.ones(len(df),dtype=int)
    for i in range(1,len(df)):
        fub[i] = ub[i] if (ub[i]<fub[i-1] or c[i-1]>fub[i-1]) else fub[i-1]
        flb[i] = lb[i] if (lb[i]>flb[i-1] or c[i-1]<flb[i-1]) else flb[i-1]
        prev = stl[i-1] if i>1 else fub[0]
        if prev==fub[i-1]:
            if c[i]>fub[i]: dr[i]=-1; stl[i]=flb[i]
            else:            dr[i]= 1; stl[i]=fub[i]
        else:
            if c[i]<flb[i]: dr[i]= 1; stl[i]=fub[i]
            else:            dr[i]=-1; stl[i]=flb[i]
    return pd.Series(stl,index=df.index), pd.Series(dr,index=df.index)


def calc_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).round(2)


def calc_vwap(df):
    tp  = (df["high"] + df["low"] + df["close"]) / 3
    cvv = (tp * df["volume"]).cumsum()
    cv  = df["volume"].cumsum().replace(0, np.nan)
    return (cvv / cv).round(4)


def calc_pivots(df):
    h=df["high"].iloc[-2]; l=df["low"].iloc[-2]; c=df["close"].iloc[-2]
    pp=(h+l+c)/3
    return {"PP":pp,
            "R1":2*pp-l,"S1":2*pp-h,
            "R2":pp+(h-l),"S2":pp-(h-l),
            "R3":h+2*(pp-l),"S3":l-2*(h-pp)}


def calc_enhanced_signal(df, adx_period=14, st_period=10, st_mult=3.0, atr_period=14):
    """Full strategy: 7 conditions + ATR-based entry, stop-loss, and targets."""
    price = df["close"].iloc[-1]
    adx, pdi, mdi = calc_adx(df, adx_period)
    stl, dr       = calc_supertrend(df, st_period, st_mult)
    rsi           = calc_rsi(df["close"], 14)
    vwap          = calc_vwap(df)
    atr           = calc_atr(df["high"], df["low"], df["close"], atr_period)
    pivots        = calc_pivots(df)

    vol_avg = df["volume"].rolling(20).mean().iloc[-1]
    vol_now = df["volume"].iloc[-1]
    vol_spike = vol_now > (vol_avg * 1.5) if vol_avg > 0 else False

    di_bullish = pdi.iloc[-1] > mdi.iloc[-1]
    di_bearish = mdi.iloc[-1] > pdi.iloc[-1]
    pdi_cross  = di_bullish and (pdi.iloc[-2] <= mdi.iloc[-2])
    mdi_cross  = di_bearish and (mdi.iloc[-2] <= pdi.iloc[-2])

    rsi_val = rsi.iloc[-1]
    rsi_ok_buy  = 30 <= rsi_val <= 70
    rsi_ok_sell = 30 <= rsi_val <= 70
    atr_val     = atr.iloc[-1]
    vwap_val    = vwap.iloc[-1]

    buy_conditions = {
        "di_bullish":    di_bullish,
        "adx_strong":    adx.iloc[-1] > ADX_STRONG,
        "above_vwap":    price > vwap_val,
        "st_bullish":    dr.iloc[-1] == -1,
        "above_pivot":   price > pivots["PP"],
        "rsi_ok":        rsi_ok_buy,
        "vol_spike":     vol_spike,
    }
    sell_conditions = {
        "di_bearish":    di_bearish,
        "adx_strong":    adx.iloc[-1] > ADX_STRONG,
        "below_vwap":    price < vwap_val,
        "st_bearish":    dr.iloc[-1] == 1,
        "below_pivot":   price < pivots["PP"],
        "rsi_ok":        rsi_ok_sell,
        "vol_spike":     vol_spike,
    }

    buy_score  = sum(buy_conditions.values())
    sell_score = sum(sell_conditions.values())

    is_buy  = buy_score >= 5
    is_sell = sell_score >= 5

    if is_buy:
        entry    = price
        stop     = entry - 1.5 * atr_val
        target1  = entry + 1.5 * atr_val
        target2  = entry + 3.0 * atr_val
        target3  = entry + 4.5 * atr_val
        trailing = entry - 1.0 * atr_val
    elif is_sell:
        entry    = price
        stop     = entry + 1.5 * atr_val
        target1  = entry - 1.5 * atr_val
        target2  = entry - 3.0 * atr_val
        target3  = entry - 4.5 * atr_val
        trailing = entry + 1.0 * atr_val
    else:
        entry = stop = target1 = target2 = target3 = trailing = None

    return {
        "price": price, "pp": pivots["PP"], "pivots": pivots,
        "adx": adx, "pdi": pdi, "mdi": mdi,
        "adx_val": adx.iloc[-1], "pdi_val": pdi.iloc[-1], "mdi_val": mdi.iloc[-1],
        "stl": stl, "dr": dr,
        "rsi": rsi, "rsi_val": rsi_val,
        "vwap": vwap, "vwap_val": vwap_val,
        "atr": atr, "atr_val": atr_val,
        "vol_now": vol_now, "vol_avg": vol_avg, "vol_spike": vol_spike,
        "di_bullish": di_bullish, "di_bearish": di_bearish,
        "pdi_crossover": pdi_cross, "mdi_crossover": mdi_cross,
        "buy_conditions": buy_conditions, "sell_conditions": sell_conditions,
        "buy_score": buy_score, "sell_score": sell_score,
        "BUY": is_buy, "SELL": is_sell,
        "entry": entry, "stop": stop,
        "target1": target1, "target2": target2, "target3": target3,
        "trailing": trailing,
    }


def check_signal(df, adx_period=14):
    adx, pdi, mdi = calc_adx(df, adx_period)
    pivots = calc_pivots(df)
    price  = df["close"].iloc[-1]

    di_bullish = pdi.iloc[-1] > mdi.iloc[-1]
    di_bearish = mdi.iloc[-1] > pdi.iloc[-1]

    pdi_cross = di_bullish and (pdi.iloc[-2] <= mdi.iloc[-2])
    mdi_cross = di_bearish and (mdi.iloc[-2] <= pdi.iloc[-2])

    return {
        "price":        price,
        "pp":           pivots["PP"],
        "adx":          adx.iloc[-1],
        "pdi":          pdi.iloc[-1],
        "mdi":          mdi.iloc[-1],
        "above_pivot":  price > pivots["PP"],
        "below_pivot":  price < pivots["PP"],
        "adx_strong":   adx.iloc[-1] > ADX_STRONG,
        "di_bullish":   di_bullish,
        "di_bearish":   di_bearish,
        "pdi_crossover": pdi_cross,
        "mdi_crossover": mdi_cross,
        "BUY":  (price > pivots["PP"]) and (adx.iloc[-1] > ADX_STRONG) and di_bullish,
        "SELL": (price < pivots["PP"]) and (adx.iloc[-1] > ADX_STRONG) and di_bearish,
    }

# ─────────────────────────────────────────────────────────────
# ANGEL ONE API
# ─────────────────────────────────────────────────────────────

def _headers(api_key, jwt=None):
    h = {"Content-Type":"application/json","Accept":"application/json",
         "X-PrivateKey":api_key,"X-SourceID":"WEB",
         "X-ClientLocalIP":"127.0.0.1","X-ClientPublicIP":"127.0.0.1",
         "X-MACAddress":"fe:80:00:00:00:00","X-UserType":"USER"}
    if jwt: h["Authorization"]=f"Bearer {jwt}"
    return h


def angel_login(api_key, client_id, password, totp_secret):
    totp = pyotp.TOTP(totp_secret.strip().upper()).now()
    r    = requests.post(f"{BASE_URL}/rest/auth/angelbroking/user/v1/loginByPassword",
                         headers=_headers(api_key),
                         json={"clientcode":client_id,"password":password,"totp":totp},
                         timeout=15)
    d = r.json()
    if d.get("status"): return d["data"]["jwtToken"]
    raise Exception(d.get("message","Login failed"))


def search_scrip(api_key, jwt, exchange, keyword):
    r = requests.post(f"{BASE_URL}/rest/secure/angelbroking/order/v1/searchScrip",
                      headers=_headers(api_key,jwt),
                      json={"exchange":exchange,"searchscrip":keyword},timeout=15)
    d = r.json()
    return d.get("data",[]) if d.get("status") else []


def fetch_candles(api_key, jwt, exchange, token, interval, days_back):
    fmt = "%Y-%m-%d %H:%M"; now = datetime.now()
    r   = requests.post(
        f"{BASE_URL}/rest/secure/angelbroking/historical/v1/getCandleData",
        headers=_headers(api_key,jwt),
        json={"exchange":exchange,"symboltoken":token,"interval":interval,
              "fromdate":(now-timedelta(days=days_back)).strftime(fmt),
              "todate":now.strftime(fmt)},
        timeout=20)
    d = r.json()
    if d.get("status") and d.get("data"):
        df = pd.DataFrame(d["data"],columns=["datetime","open","high","low","close","volume"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df.set_index("datetime").sort_index().astype(float)
    raise Exception(d.get("message","No candle data"))

# ─────────────────────────────────────────────────────────────
# CRYPTO API  (Binance → Kraken fallback, no key required)
# Kraken is US-based — works on Streamlit Cloud without geo-blocks
# ─────────────────────────────────────────────────────────────

CRYPTO_SYMBOLS = {
    "BTC/USDT":  "BTCUSDT",
    "ETH/USDT":  "ETHUSDT",
    "SOL/USDT":  "SOLUSDT",
    "BNB/USDT":  "BNBUSDT",
}

BINANCE_INTERVALS = {
    "5 min":  ("5m",  500),
    "15 min": ("15m", 500),
    "1 hour": ("1h",  500),
    "1 day":  ("1d",  365),
}

KRAKEN_SYMBOL_MAP = {
    "BTCUSDT": "XBTUSDT", "ETHUSDT": "ETHUSDT",
    "SOLUSDT": "SOLUSDT", "BNBUSDT": "BNBUSDT",
}
KRAKEN_INTERVAL_MAP = {"5m": 5, "15m": 15, "1h": 60, "1d": 1440}


def _fetch_binance(symbol, interval, limit):
    r = requests.get("https://api.binance.com/api/v3/klines",
                     params={"symbol": symbol, "interval": interval, "limit": limit},
                     timeout=15)
    r.raise_for_status()
    raw = r.json()
    df = pd.DataFrame(raw, columns=[
        "datetime","open","high","low","close","volume",
        "close_time","qav","num_trades","tbbav","tbqav","ignore"])
    df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
    df = df.set_index("datetime")[["open","high","low","close","volume"]]
    return df.astype(float).sort_index()


def _fetch_kraken(symbol, interval, limit):
    kr_sym = KRAKEN_SYMBOL_MAP.get(symbol, symbol)
    kr_iv  = KRAKEN_INTERVAL_MAP.get(interval, 15)
    since  = int((datetime.now() - timedelta(minutes=kr_iv * limit)).timestamp())
    r = requests.get("https://api.kraken.com/0/public/OHLC",
                     params={"pair": kr_sym, "interval": kr_iv, "since": since},
                     timeout=15)
    r.raise_for_status()
    data = r.json()
    if data.get("error"):
        raise Exception(f"Kraken: {data['error']}")
    pair_key = list(data["result"].keys())
    pair_key = [k for k in pair_key if k != "last"][0]
    rows = data["result"][pair_key]
    df = pd.DataFrame(rows, columns=[
        "datetime","open","high","low","close","vwap","volume","count"])
    df["datetime"] = pd.to_datetime(df["datetime"].astype(int), unit="s")
    df = df.set_index("datetime")[["open","high","low","close","volume"]]
    return df.astype(float).sort_index()


def fetch_crypto_candles(symbol, interval, limit=500):
    """Try Binance first; fall back to Kraken if geo-blocked."""
    try:
        return _fetch_binance(symbol, interval, limit)
    except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError):
        return _fetch_kraken(symbol, interval, min(limit, 720))


def _ticker_binance(symbol):
    r = requests.get("https://api.binance.com/api/v3/ticker/24hr",
                     params={"symbol": symbol}, timeout=10)
    r.raise_for_status()
    return r.json()


def _ticker_kraken(symbol):
    kr_sym = KRAKEN_SYMBOL_MAP.get(symbol, symbol)
    r = requests.get("https://api.kraken.com/0/public/Ticker",
                     params={"pair": kr_sym}, timeout=10)
    r.raise_for_status()
    data = r.json()
    pair_key = [k for k in data.get("result", {}).keys()][0]
    t = data["result"][pair_key]
    return {
        "highPrice":   t["h"][1],
        "lowPrice":    t["l"][1],
        "quoteVolume": str(float(t["v"][1]) * float(t["p"][1])),
    }


def fetch_crypto_ticker(symbol):
    """Try Binance first; fall back to Kraken if geo-blocked."""
    try:
        return _ticker_binance(symbol)
    except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError):
        return _ticker_kraken(symbol)


# ─────────────────────────────────────────────────────────────
# EMAIL
# ─────────────────────────────────────────────────────────────

def send_email_alert(cfg, symbol, r15, r1h, signal_type="BUY"):
    try:
        price = r15["price"]
        is_buy = signal_type == "BUY"
        accent = "#00e676" if is_buy else "#ff5252"
        bg_grad = "linear-gradient(135deg,#0d2b1a,#002a10)" if is_buy else "linear-gradient(135deg,#2b0d0d,#2a0000)"
        emoji = "🟢" if is_buy else "🔴"
        di_label = "+DI > -DI (Blue > Green)" if is_buy else "-DI > +DI (Green > Blue)"
        price_cmp = "&gt;" if is_buy else "&lt;"

        msg   = MIMEMultipart("alternative")
        msg["Subject"] = f"{emoji} MCX {signal_type} Alert — {symbol} @ ₹{price:,.0f}"
        msg["From"]    = cfg["sender"]
        msg["To"]      = cfg["recipient"]
        now_str        = datetime.now().strftime("%d %b %Y  %H:%M:%S")
        html = f"""
        <html><body style="font-family:Arial,sans-serif;background:#0e1117;color:#c9d1d9;padding:20px">
        <div style="max-width:520px;margin:auto;background:#1c1f26;border-radius:14px;
                    border:2px solid {accent};overflow:hidden">
          <div style="background:{bg_grad};padding:20px 24px">
            <h1 style="color:{accent};margin:0">{emoji} {signal_type} SIGNAL TRIGGERED</h1>
            <p style="color:#888;margin:4px 0 0">{now_str}</p>
          </div>
          <div style="padding:20px 24px">
            <table style="width:100%;border-collapse:collapse">
              <tr><td style="padding:8px 0;color:#888">Instrument</td>
                  <td style="font-weight:700;color:#ffd700">{symbol}</td></tr>
              <tr><td style="padding:8px 0;color:#888">LTP</td>
                  <td style="font-weight:700">₹{price:,.0f}</td></tr>
              <tr><td style="padding:8px 0;color:#888">DI Condition</td>
                  <td style="font-weight:700;color:{accent}">{di_label}</td></tr>
            </table>
            <hr style="border-color:#333;margin:16px 0">
            <h3 style="color:{accent};margin:0 0 12px">✅ All 3 Conditions — Both Timeframes</h3>
            <table style="width:100%;border-collapse:collapse;font-size:0.9rem">
              <tr style="background:#0d2b1a">
                <th style="padding:8px;text-align:left">Timeframe</th>
                <th style="padding:8px;text-align:right">Price vs PP</th>
                <th style="padding:8px;text-align:right">ADX</th>
                <th style="padding:8px;text-align:right">+DI / -DI</th>
              </tr>
              <tr>
                <td style="padding:8px">15 Min</td>
                <td style="padding:8px;text-align:right;color:{accent}">₹{r15['price']:,.0f} {price_cmp} PP ₹{r15['pp']:,.0f}</td>
                <td style="padding:8px;text-align:right;color:{accent}">{r15['adx']:.1f}</td>
                <td style="padding:8px;text-align:right;color:{accent}">{r15['pdi']:.1f} / {r15['mdi']:.1f}</td>
              </tr>
              <tr style="background:#161a20">
                <td style="padding:8px">1 Hour</td>
                <td style="padding:8px;text-align:right;color:{accent}">₹{r1h['price']:,.0f} {price_cmp} PP ₹{r1h['pp']:,.0f}</td>
                <td style="padding:8px;text-align:right;color:{accent}">{r1h['adx']:.1f}</td>
                <td style="padding:8px;text-align:right;color:{accent}">{r1h['pdi']:.1f} / {r1h['mdi']:.1f}</td>
              </tr>
            </table>
            <p style="color:#888;font-size:0.8rem;margin-top:20px">⚠️ Automated technical alert. Not investment advice.</p>
          </div>
        </div></body></html>"""
        msg.attach(MIMEText(html,"html"))
        with smtplib.SMTP_SSL("smtp.gmail.com",465) as s:
            s.login(cfg["sender"],cfg["app_password"])
            s.send_message(msg)
        return True
    except Exception as e:
        return str(e)

# ─────────────────────────────────────────────────────────────
# CHART
# ─────────────────────────────────────────────────────────────

def build_chart(df, title, adx_p=14, st_p=10, st_m=3.0):
    adx,pdi,mdi   = calc_adx(df,adx_p)
    stl,dr        = calc_supertrend(df,st_p,st_m)
    pivots        = calc_pivots(df)
    bull = stl.copy(); bull[dr== 1]=np.nan
    bear = stl.copy(); bear[dr==-1]=np.nan
    fig  = make_subplots(rows=2,cols=1,shared_xaxes=True,
                         row_heights=[0.68,0.32],vertical_spacing=0.03,
                         subplot_titles=(title,"ADX (gold)  |  +DI (blue)  |  -DI (green)"))
    fig.add_trace(go.Candlestick(x=df.index,open=df.open,high=df.high,low=df.low,close=df.close,
                  name="Price",increasing_line_color="#26a69a",decreasing_line_color="#ef5350",
                  increasing_fillcolor="#26a69a",decreasing_fillcolor="#ef5350"),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=bull,mode="lines",name="ST Bullish",
                             line=dict(color="#00e676",width=2.2)),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=bear,mode="lines",name="ST Bearish",
                             line=dict(color="#ff1744",width=2.2)),row=1,col=1)
    colors={"PP":"#ffd700","R1":"#ff6b6b","R2":"#ff4444","R3":"#cc0000",
            "S1":"#69f0ae","S2":"#00e676","S3":"#00875a"}
    x0=df.index[max(-80,-len(df))]; x1=df.index[-1]
    for k,v in pivots.items():
        fig.add_shape(type="line",x0=x0,x1=x1,y0=v,y1=v,row=1,col=1,
                      line=dict(color=colors[k],width=1,dash="dot" if k!="PP" else "dash"))
        fig.add_annotation(x=x1,y=v,text=f"  {k} {v:,.0f}",showarrow=False,
                           xanchor="left",font=dict(color=colors[k],size=10),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=adx,name="ADX",line=dict(color="#ffd700",width=2)),row=2,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=pdi,name="+DI",line=dict(color="#2962FF",width=1.5)),row=2,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=mdi,name="-DI",line=dict(color="#4CAF50",width=1.5)),row=2,col=1)
    fig.add_hline(y=25,line_dash="dash",line_color="#444",row=2,col=1,
                  annotation_text="25",annotation_font_color="#888")
    fig.update_layout(height=680,template="plotly_dark",paper_bgcolor="#0e1117",
                      plot_bgcolor="#0e1117",xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h",y=1.02,x=0,font_size=11),
                      margin=dict(l=10,r=110,t=40,b=10),font=dict(color="#c9d1d9"))
    fig.update_xaxes(gridcolor="#1c1f26",showgrid=True)
    fig.update_yaxes(gridcolor="#1c1f26",showgrid=True)
    return fig, adx, pdi, mdi, stl, dr, pivots

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def signal_box(label, text, kind):
    st.markdown(f'<div style="margin-bottom:6px"><strong>{label}</strong></div>'
                f'<div class="signal-box {kind}">{text}</div>', unsafe_allow_html=True)


def tf_card(timeframe, res):
    is_buy  = res["BUY"]
    is_sell = res["SELL"]

    if is_buy:
        cls, icon, label = "tf-pass", "✅", "BUY conditions MET"
    elif is_sell:
        cls, icon, label = "tf-fail", "🔴", "SELL conditions MET"
    else:
        cls, icon, label = "tf-fail", "⏳", "No clear signal"

    cross_txt = ""
    if res.get("pdi_crossover"):
        cross_txt = '<div style="margin:5px 0;font-size:0.88rem">🔵 +DI just crossed above -DI (BUY crossover)</div>'
    elif res.get("mdi_crossover"):
        cross_txt = '<div style="margin:5px 0;font-size:0.88rem">🔴 -DI just crossed above +DI (SELL crossover)</div>'

    checks = [
        (f"Price ₹{res['price']:,.0f} {'>' if res['above_pivot'] else '<'} PP ₹{res['pp']:,.0f}",
         res["above_pivot"] if not is_sell else res["below_pivot"]),
        (f"ADX {res['adx']:.1f} {'> 25 (Strong)' if res['adx_strong'] else '< 25 (Weak)'}",
         res["adx_strong"]),
        (f"+DI {res['pdi']:.1f} {'>' if res['di_bullish'] else '<'} -DI {res['mdi']:.1f}",
         res["di_bullish"] if not is_sell else res["di_bearish"]),
    ]
    rows = "".join(
        f'<div style="margin:5px 0;font-size:0.88rem">{"🟢" if ok else "🔴"} {txt}</div>'
        for txt, ok in checks
    )
    st.markdown(f"""
    <div class="tf-card {cls}">
      <div style="font-size:0.75rem;color:#888;margin-bottom:4px">{timeframe}</div>
      <div style="font-size:1rem;font-weight:700;margin-bottom:8px">{icon} {label}</div>
      {cross_txt}{rows}
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🥇 MCX Analyzer")
    st.divider()
    st.markdown("### 🔐 Angel One Login")
    api_key     = st.text_input("API Key",     type="password")
    client_id   = st.text_input("Client ID",               placeholder="e.g. A123456")
    password    = st.text_input("Password",    type="password")
    totp_secret = st.text_input("TOTP Secret", type="password", placeholder="Base32 key")

    if st.button("🔑 Login", use_container_width=True, type="primary"):
        if not all([api_key,client_id,password,totp_secret]):
            st.error("All fields required")
        else:
            with st.spinner("Authenticating…"):
                try:
                    jwt = angel_login(api_key,client_id,password,totp_secret)
                    st.session_state.update({"jwt":jwt,"api_key":api_key})
                    st.success("✅ Logged in!")
                except Exception as e:
                    st.error(str(e))

    if "jwt" in st.session_state:
        st.divider()
        st.markdown("### 📊 Chart Settings")
        instrument = st.selectbox("Instrument", list(INSTRUMENT_SEARCH.keys()))
        timeframe  = st.selectbox("Chart Timeframe", list(INTERVAL_MAP.keys()), index=1)
        exchange, keyword = INSTRUMENT_SEARCH[instrument]

        if st.button("🔍 Find Contracts", use_container_width=True):
            with st.spinner("Searching…"):
                results = search_scrip(st.session_state.api_key,
                                       st.session_state.jwt, exchange, keyword)
                if results:
                    st.session_state["contracts"] = results
                    st.session_state.pop("symboltoken",None)
                else:
                    st.warning("No contracts found")

        if "contracts" in st.session_state:
            opts = {f"{s['tradingsymbol']}  (exp {s.get('expiry','—')})": s["symboltoken"]
                    for s in st.session_state["contracts"][:12]}
            sel = st.selectbox("Select Contract", list(opts.keys()))
            st.session_state["symboltoken"] = opts[sel]
            st.session_state["instrument"]  = instrument
            st.session_state["timeframe"]   = timeframe

        if "symboltoken" in st.session_state:
            st.divider()
            adx_period = st.slider("ADX Period",            7,30,14)
            st_period  = st.slider("SuperTrend Period",     5,20,10)
            st_mult    = st.slider("SuperTrend Multiplier",1.0,5.0,3.0,0.5)
            st.session_state["params"] = (adx_period, st_period, st_mult)

            if st.button("📈 Load Chart", use_container_width=True, type="primary"):
                st.session_state["load"] = True

        # ── ALERT SECTION ────────────────────────────────────
        st.divider()
        st.markdown("### 🔔 BUY / SELL Alert Settings")
        alert_on = st.toggle("Enable Alert Scanner", value=False)

        if alert_on:
            refresh_sec = st.selectbox("Scan interval",
                                       [60,120,300,600],
                                       format_func=lambda x: f"{x//60} min")
            st.markdown("**📧 Email Notification**")
            email_on = st.toggle("Send email on BUY / SELL signal", value=False)
            if email_on:
                sender_email    = st.text_input("Your Gmail")
                app_password    = st.text_input("Gmail App Password", type="password",
                    help="myaccount.google.com → Security → App Passwords")
                recipient_email = st.text_input("Alert recipient email")
                st.session_state["email_cfg"] = {
                    "sender": sender_email, "app_password": app_password,
                    "recipient": recipient_email,
                    "enabled": bool(sender_email and app_password and recipient_email),
                }
            st.session_state["alert_cfg"] = {"enabled":True,"refresh_sec":refresh_sec}
        else:
            st.session_state.pop("alert_cfg",None)

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

st.markdown("## 📊 MCX & Crypto Analyzer")
tab1, tab2, tab3 = st.tabs(["📈  MCX Chart", "🔔  MCX BUY / SELL Alert", "₿  Crypto (BTC / ETH)"])

# Guard flag — MCX tabs need login
mcx_ready = "jwt" in st.session_state

# ───────────────────────── TAB 1 — MCX CHART ─────────────
with tab1:
    if not mcx_ready:
        st.info("👈 Login with Angel One SmartAPI credentials in the sidebar to use MCX charts.")
        c1,c2,c3 = st.columns(3)
        with c1: st.markdown("### 📊 ADX"); st.write("Measures trend strength. ADX > 25 = strong.")
        with c2: st.markdown("### 🟢 SuperTrend"); st.write("Green = uptrend, Red = downtrend.")
        with c3: st.markdown("### 🔵 Pivot Points"); st.write("PP, R1–R3, S1–S3 key levels.")
    elif st.session_state.get("load"):
        with st.spinner("Fetching MCX data…"):
            try:
                adx_p,st_p,st_m = st.session_state.get("params",(14,10,3.0))
                exchange,_      = INSTRUMENT_SEARCH[st.session_state["instrument"]]
                interval,days   = INTERVAL_MAP[st.session_state["timeframe"]]
                df = fetch_candles(st.session_state["api_key"],st.session_state["jwt"],
                                   exchange,st.session_state["symboltoken"],interval,days)
                sym   = st.session_state["instrument"]
                title = f"{sym}  —  {st.session_state['timeframe']}"
                fig,adx,pdi,mdi,stl,dr,pivots = build_chart(df,title,adx_p,st_p,st_m)
                price=df.close.iloc[-1]; prev=df.close.iloc[-2]
                chg=price-prev; chgp=chg/prev*100
                m1,m2,m3,m4,m5 = st.columns(5)
                m1.metric("LTP",       f"₹{price:,.0f}",  f"{chg:+.0f} ({chgp:+.2f}%)")
                m2.metric("ADX",       f"{adx.iloc[-1]:.1f}", "Strong ✅" if adx.iloc[-1]>25 else "Weak ⚠️")
                m3.metric("+DI",       f"{pdi.iloc[-1]:.1f}")
                m4.metric("−DI",       f"{mdi.iloc[-1]:.1f}")
                m5.metric("SuperTrend","🟢 BUY" if dr.iloc[-1]==-1 else "🔴 SELL")
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("🧠 Signal Dashboard")
                s1,s2,s3 = st.columns(3)
                with s1:
                    if adx.iloc[-1]>25:
                        if pdi.iloc[-1]>mdi.iloc[-1]:
                            txt  = f"ADX {adx.iloc[-1]:.1f} — Strong 📈 Bullish (🔵+DI > 🟢-DI)"
                            kind = "buy"
                        else:
                            txt  = f"ADX {adx.iloc[-1]:.1f} — Strong 📉 Bearish (🟢-DI > 🔵+DI)"
                            kind = "sell"
                    else: txt,kind = f"ADX {adx.iloc[-1]:.1f} — Weak / Ranging","neutral"
                    signal_box("📊 ADX Trend",txt,kind)
                with s2:
                    if dr.iloc[-1]==-1: signal_box("🟢 SuperTrend","BULLISH — Above SuperTrend","buy")
                    else:               signal_box("🔴 SuperTrend","BEARISH — Below SuperTrend","sell")
                with s3:
                    pp=pivots["PP"]
                    if   price>pivots["R1"]: txt,kind=f"Above R1 ({pivots['R1']:,.0f}) — Strongly Bullish","buy"
                    elif price>pp:           txt,kind=f"Above PP ({pp:,.0f}) — Bullish","buy"
                    elif price<pivots["S1"]: txt,kind=f"Below S1 ({pivots['S1']:,.0f}) — Strongly Bearish","sell"
                    elif price<pp:           txt,kind=f"Below PP ({pp:,.0f}) — Bearish","sell"
                    else:                    txt,kind=f"Near PP ({pp:,.0f}) — Neutral","neutral"
                    signal_box("🔵 Pivot Position",txt,kind)

                st.subheader("📐 Pivot Levels")
                rows=[{"Level":k,"Price (₹)":f"{v:,.0f}","Distance":f"{((price-v)/v*100):+.2f}%",
                        "Position":"✅ Above" if price>=v else "❌ Below"}
                      for k,v in pivots.items()]
                st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
                with st.expander("📋 OHLCV (last 50 candles)"):
                    show=df.tail(50).copy(); show.index=show.index.strftime("%Y-%m-%d %H:%M")
                    st.dataframe(show.style.format("{:,.2f}"),use_container_width=True)
            except Exception as ex:
                st.error(f"❌ {ex}")
    else:
        st.info("👈 Select instrument → Find Contracts → Load Chart")

# ───────────────────── TAB 2 — MCX ALERT SCANNER ─────────
with tab2:
    if not mcx_ready:
        st.info("👈 Login with Angel One SmartAPI to use the MCX Alert Scanner.")
    else:
        st.markdown("## 🔔 Multi-Timeframe BUY / SELL Alert Scanner")

        ca, cb, cc = st.columns(3)
        ca.info("📍 **Price vs Pivot Point**\nAbove PP → BUY · Below PP → SELL")
        cb.info("💪 **ADX > 25 (Strong Trend)**\nOn both 15-min and 1-hour")
        cc.info("📈 **DI Crossover**\n🔵 +DI > 🟢 -DI → BUY\n🟢 -DI > 🔵 +DI → SELL")
        st.divider()

        if "alert_cfg" not in st.session_state:
            st.warning("👈 Enable **Alert Scanner** in the sidebar first.")
        elif "symboltoken" not in st.session_state:
            st.warning("👈 Select a contract in the sidebar first.")
        else:
            cfg        = st.session_state["alert_cfg"]
            api_key_s  = st.session_state["api_key"]
            jwt_s      = st.session_state["jwt"]
            email_cfg  = st.session_state.get("email_cfg", {})
            exchange, _ = INSTRUMENT_SEARCH[st.session_state["instrument"]]
            token      = st.session_state["symboltoken"]
            sym        = st.session_state["instrument"]
            adx_p      = st.session_state.get("params", (14, 10, 3.0))[0]

            left, right = st.columns([3, 2])

            with left:
                st.markdown("### 🔍 Live Scan")
                with st.spinner("Fetching 15-min & 1-hour data…"):
                    try:
                        df15 = fetch_candles(api_key_s, jwt_s, exchange, token, "FIFTEEN_MINUTE", 60)
                        df1h = fetch_candles(api_key_s, jwt_s, exchange, token, "ONE_HOUR", 180)
                        r15  = check_signal(df15, adx_p)
                        r1h  = check_signal(df1h, adx_p)

                        strong_buy  = r15["BUY"]  and r1h["BUY"]
                        strong_sell = r15["SELL"] and r1h["SELL"]
                        now_str = datetime.now().strftime("%d %b %Y  %H:%M:%S")

                        if strong_buy:
                            st.markdown(f"""
                            <div class="alert-banner">
                              <h2>🟢 STRONG BUY — Both Timeframes</h2>
                              <p>{sym} &nbsp;|&nbsp; LTP ₹{r15['price']:,.0f}
                                 &nbsp;|&nbsp; {now_str}</p>
                              <p style="margin-top:8px;font-size:0.9rem;color:#a8f0c6">
                                🔵 +DI above 🟢 -DI on <strong>both 15-min &amp; 1-hour</strong>.
                                Price above PP, ADX &gt; 25.
                              </p>
                            </div>""", unsafe_allow_html=True)
                            st.toast(f"🟢 STRONG BUY — {sym} @ ₹{r15['price']:,.0f}", icon="🟢")

                            if email_cfg.get("enabled"):
                                last = st.session_state.get("last_alert_ts", 0)
                                if (time.time() - last) > 60:
                                    res = send_email_alert(email_cfg, sym, r15, r1h, "BUY")
                                    st.session_state["last_alert_ts"] = time.time()
                                    if res is True: st.success("📧 Email sent!")
                                    else:           st.warning(f"Email failed: {res}")
                                else:
                                    st.caption("📧 Email cooldown active (1 min)")

                            hist = st.session_state.get("alert_history", [])
                            hist.insert(0, {"Time": now_str, "Symbol": sym,
                                            "LTP": f"₹{r15['price']:,.0f}",
                                            "ADX 15m": f"{r15['adx']:.1f}",
                                            "ADX 1h": f"{r1h['adx']:.1f}",
                                            "+DI > -DI 15m": "✅" if r15["di_bullish"] else "❌",
                                            "+DI > -DI 1h":  "✅" if r1h["di_bullish"] else "❌",
                                            "Signal": "🟢 STRONG BUY"})
                            st.session_state["alert_history"] = hist[:50]

                        elif strong_sell:
                            st.markdown(f"""
                            <div class="alert-banner" style="background:linear-gradient(135deg,#2b0d0d,#2a0000);
                                        border-color:#ff5252">
                              <h2 style="color:#ff5252">🔴 STRONG SELL — Both Timeframes</h2>
                              <p>{sym} &nbsp;|&nbsp; LTP ₹{r15['price']:,.0f}
                                 &nbsp;|&nbsp; {now_str}</p>
                              <p style="margin-top:8px;font-size:0.9rem;color:#ff8a80">
                                🟢 -DI above 🔵 +DI on <strong>both 15-min &amp; 1-hour</strong>.
                                Price below PP, ADX &gt; 25.
                              </p>
                            </div>""", unsafe_allow_html=True)
                            st.toast(f"🔴 STRONG SELL — {sym} @ ₹{r15['price']:,.0f}", icon="🔴")

                            if email_cfg.get("enabled"):
                                last = st.session_state.get("last_sell_ts", 0)
                                if (time.time() - last) > 60:
                                    res = send_email_alert(email_cfg, sym, r15, r1h, "SELL")
                                    st.session_state["last_sell_ts"] = time.time()
                                    if res is True: st.success("📧 Sell alert email sent!")
                                    else:           st.warning(f"Email failed: {res}")
                                else:
                                    st.caption("📧 Email cooldown active (1 min)")

                            hist = st.session_state.get("alert_history", [])
                            hist.insert(0, {"Time": now_str, "Symbol": sym,
                                            "LTP": f"₹{r15['price']:,.0f}",
                                            "ADX 15m": f"{r15['adx']:.1f}",
                                            "ADX 1h": f"{r1h['adx']:.1f}",
                                            "+DI > -DI 15m": "✅" if r15["di_bullish"] else "❌",
                                            "+DI > -DI 1h":  "✅" if r1h["di_bullish"] else "❌",
                                            "Signal": "🔴 STRONG SELL"})
                            st.session_state["alert_history"] = hist[:50]

                        else:
                            st.markdown(f"""
                            <div class="waiting-banner">
                              <span style="color:#ffd700;font-size:1.1rem;font-weight:600">
                                ⏳ Monitoring — No strong signal yet</span><br>
                              <span style="color:#888;font-size:0.88rem">
                                {sym} &nbsp;|&nbsp; LTP ₹{r15['price']:,.0f}
                                &nbsp;|&nbsp; {now_str}</span>
                            </div>""", unsafe_allow_html=True)

                        st.markdown("#### 15-Minute Conditions")
                        tf_card("15 Min", r15)
                        st.markdown("#### 1-Hour Conditions")
                        tf_card("1 Hour", r1h)

                        refresh = cfg["refresh_sec"]
                        st.caption(f"🔄 Next scan in {refresh//60} min — page will auto-refresh")
                        time.sleep(refresh)
                        st.rerun()

                    except Exception as e:
                        st.error(f"Scan error: {e}")

            with right:
                st.markdown("### 📋 Alert History")
                hist = st.session_state.get("alert_history", [])
                if hist:
                    st.dataframe(pd.DataFrame(hist), use_container_width=True, hide_index=True)
                    if st.button("🗑 Clear History"):
                        st.session_state["alert_history"] = []
                        st.rerun()
                else:
                    st.info("Alerts will appear here when triggered.")
                    st.markdown("""
                    **🟢 STRONG BUY fires when (both 15-min & 1-hour):**
                    - ✅ Price > PP
                    - ✅ ADX > 25
                    - ✅ 🔵 +DI > 🟢 -DI  (blue crosses green)

                    **🔴 STRONG SELL fires when (both 15-min & 1-hour):**
                    - ✅ Price < PP
                    - ✅ ADX > 25
                    - ✅ 🟢 -DI > 🔵 +DI  (green crosses blue)
                    """)

# ───────────────────── TAB 3 — CRYPTO STRATEGY TESTER ─────
with tab3:
    st.markdown("## ₿ Pro Strategy Tester — Binance Live Data")
    st.caption("🆓 No API key needed  ·  7-condition strategy  ·  ATR-based entry / stop / targets")

    # ── Controls ──────────────────────────────────────────
    ctrl1, ctrl2, ctrl3, ctrl4, ctrl5 = st.columns([2,1,1,1,1])
    with ctrl1:
        crypto_sym = st.selectbox("Coin", list(CRYPTO_SYMBOLS.keys()),
                                  label_visibility="collapsed", key="crypto_sym")
    with ctrl2:
        crypto_tf = st.selectbox("Timeframe", list(BINANCE_INTERVALS.keys()),
                                 index=1, label_visibility="collapsed", key="crypto_tf")
    with ctrl3:
        crypto_adx = st.number_input("ADX Period", 7, 30, 14,
                                     label_visibility="collapsed", key="crypto_adx")
    with ctrl4:
        crypto_stp = st.number_input("ST Period", 5, 20, 10,
                                     label_visibility="collapsed", key="crypto_stp")
    with ctrl5:
        crypto_stm = st.number_input("ST Mult", 1.0, 5.0, 3.0, 0.5,
                                     label_visibility="collapsed", key="crypto_stm")

    col_load, _ = st.columns([1, 5])
    with col_load:
        load_crypto = st.button("📈 Analyze", type="primary", key="load_crypto",
                                use_container_width=True)

    if load_crypto or st.session_state.get("crypto_loaded"):
        st.session_state["crypto_loaded"] = True
        binance_sym          = CRYPTO_SYMBOLS[crypto_sym]
        interval_code, limit = BINANCE_INTERVALS[crypto_tf]

        with st.spinner(f"Fetching {crypto_sym} from Binance…"):
            try:
                df_c   = fetch_crypto_candles(binance_sym, interval_code, limit)
                ticker = fetch_crypto_ticker(binance_sym)

                sig = calc_enhanced_signal(df_c, int(crypto_adx),
                                           int(crypto_stp), float(crypto_stm))

                price_c = sig["price"]
                prev_c  = df_c.close.iloc[-2]
                chg_c   = price_c - prev_c
                chgp_c  = chg_c / prev_c * 100
                high24  = float(ticker.get("highPrice", 0))
                low24   = float(ticker.get("lowPrice",  0))
                vol24   = float(ticker.get("quoteVolume", 0))

                # ── Metrics row ───────────────────────────
                mc1,mc2,mc3,mc4,mc5,mc6,mc7,mc8 = st.columns(8)
                mc1.metric("Price", f"${price_c:,.2f}",
                           f"{chg_c:+.2f} ({chgp_c:+.2f}%)")
                mc2.metric("VWAP",  f"${sig['vwap_val']:,.2f}",
                           "Above ✅" if price_c > sig["vwap_val"] else "Below ❌")
                mc3.metric("ADX", f"{sig['adx_val']:.1f}",
                           "Strong ✅" if sig["adx_val"] > 25 else "Weak ⚠️")
                mc4.metric("+DI / −DI",
                           f"{sig['pdi_val']:.1f} / {sig['mdi_val']:.1f}")
                mc5.metric("RSI", f"{sig['rsi_val']:.1f}",
                           "OB ⚠️" if sig["rsi_val"]>70 else "OS ⚠️" if sig["rsi_val"]<30 else "OK ✅")
                mc6.metric("SuperTrend",
                           "🟢 BUY" if sig["dr"].iloc[-1]==-1 else "🔴 SELL")
                mc7.metric("Volume",
                           f"{sig['vol_now']:,.0f}",
                           f"{'🔥 Spike' if sig['vol_spike'] else 'Normal'}")
                mc8.metric("ATR", f"${sig['atr_val']:,.2f}")

                # ═══════════════════════════════════════════
                # 3-ROW CHART: Price+VWAP+ST | ADX/DI | RSI
                # ═══════════════════════════════════════════
                adx_c  = sig["adx"];  pdi_c = sig["pdi"]; mdi_c = sig["mdi"]
                stl_c  = sig["stl"];  dr_c  = sig["dr"]
                rsi_c  = sig["rsi"];  vwap_c = sig["vwap"]
                pivots_c = sig["pivots"]

                bull_c = stl_c.copy(); bull_c[dr_c ==  1] = np.nan
                bear_c = stl_c.copy(); bear_c[dr_c == -1] = np.nan

                fig_c = make_subplots(
                    rows=3, cols=1, shared_xaxes=True,
                    row_heights=[0.55, 0.22, 0.23], vertical_spacing=0.025,
                    subplot_titles=(
                        f"{crypto_sym}  —  {crypto_tf}  (Binance)",
                        "ADX (gold)  |  +DI (blue)  |  -DI (green)",
                        "RSI (14)"
                    ),
                )

                # Row 1 — Candlestick + SuperTrend + VWAP + Pivots
                fig_c.add_trace(go.Candlestick(
                    x=df_c.index, open=df_c.open, high=df_c.high,
                    low=df_c.low, close=df_c.close, name="Price",
                    increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
                    increasing_fillcolor="#26a69a",  decreasing_fillcolor="#ef5350",
                ), row=1, col=1)
                fig_c.add_trace(go.Scatter(x=df_c.index, y=bull_c, mode="lines",
                    name="ST Bullish", line=dict(color="#00e676", width=2.2)), row=1, col=1)
                fig_c.add_trace(go.Scatter(x=df_c.index, y=bear_c, mode="lines",
                    name="ST Bearish", line=dict(color="#ff1744", width=2.2)), row=1, col=1)
                fig_c.add_trace(go.Scatter(x=df_c.index, y=vwap_c, mode="lines",
                    name="VWAP", line=dict(color="#E040FB", width=2, dash="dot")), row=1, col=1)

                # Entry / SL / Target lines
                px0 = df_c.index[max(-60, -len(df_c))]; px1 = df_c.index[-1]
                if sig["entry"] is not None:
                    for lbl, val, clr, dsh in [
                        ("Entry", sig["entry"], "#FFFFFF",  "solid"),
                        ("Stop",  sig["stop"],  "#FF1744",  "dash"),
                        ("T1 (1:1)", sig["target1"], "#00E676", "dot"),
                        ("T2 (1:2)", sig["target2"], "#69F0AE", "dot"),
                        ("T3 (1:3)", sig["target3"], "#B9F6CA", "dot"),
                        ("Trail",    sig["trailing"],"#FFD740", "dashdot"),
                    ]:
                        fig_c.add_shape(type="line", x0=px0, x1=px1,
                                        y0=val, y1=val, row=1, col=1,
                                        line=dict(color=clr, width=1.2, dash=dsh))
                        fig_c.add_annotation(x=px1, y=val,
                                             text=f"  {lbl} ${val:,.2f}",
                                             showarrow=False, xanchor="left",
                                             font=dict(color=clr, size=9),
                                             row=1, col=1)

                # Pivot lines
                pcol = {"PP":"#ffd700","R1":"#ff6b6b","R2":"#ff4444","R3":"#cc0000",
                        "S1":"#69f0ae","S2":"#00e676","S3":"#00875a"}
                for pk, pv in pivots_c.items():
                    fig_c.add_shape(type="line", x0=px0, x1=px1, y0=pv, y1=pv,
                                    row=1, col=1,
                                    line=dict(color=pcol[pk], width=1,
                                              dash="dot" if pk!="PP" else "dash"))
                    fig_c.add_annotation(x=px1, y=pv,
                                         text=f"  {pk} ${pv:,.2f}",
                                         showarrow=False, xanchor="left",
                                         font=dict(color=pcol[pk], size=10),
                                         row=1, col=1)

                # Row 2 — ADX / +DI / -DI
                fig_c.add_trace(go.Scatter(x=df_c.index, y=adx_c, name="ADX",
                    line=dict(color="#ffd700", width=2)), row=2, col=1)
                fig_c.add_trace(go.Scatter(x=df_c.index, y=pdi_c, name="+DI",
                    line=dict(color="#2962FF", width=1.5)), row=2, col=1)
                fig_c.add_trace(go.Scatter(x=df_c.index, y=mdi_c, name="-DI",
                    line=dict(color="#4CAF50", width=1.5)), row=2, col=1)
                fig_c.add_hline(y=25, line_dash="dash", line_color="#444",
                                row=2, col=1, annotation_text="25",
                                annotation_font_color="#888")

                # Row 3 — RSI
                fig_c.add_trace(go.Scatter(x=df_c.index, y=rsi_c, name="RSI",
                    line=dict(color="#CE93D8", width=2)), row=3, col=1)
                fig_c.add_hline(y=70, line_dash="dash", line_color="#ff5252",
                                row=3, col=1, annotation_text="70 OB",
                                annotation_font_color="#ff5252")
                fig_c.add_hline(y=30, line_dash="dash", line_color="#00e676",
                                row=3, col=1, annotation_text="30 OS",
                                annotation_font_color="#00e676")
                fig_c.add_hline(y=50, line_dash="dot", line_color="#555",
                                row=3, col=1)
                fig_c.add_hrect(y0=30, y1=70, row=3, col=1,
                                fillcolor="#4CAF50", opacity=0.06,
                                line_width=0)

                fig_c.update_layout(
                    height=850, template="plotly_dark",
                    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                    xaxis_rangeslider_visible=False,
                    legend=dict(orientation="h", y=1.02, x=0, font_size=11),
                    margin=dict(l=10, r=140, t=40, b=10),
                    font=dict(color="#c9d1d9"),
                )
                fig_c.update_xaxes(gridcolor="#1c1f26", showgrid=True)
                fig_c.update_yaxes(gridcolor="#1c1f26", showgrid=True)
                st.plotly_chart(fig_c, use_container_width=True)

                # ═══════════════════════════════════════════
                # STRATEGY DASHBOARD
                # ═══════════════════════════════════════════
                st.subheader("🧠 7-Condition Strategy Dashboard")

                # ── Signal checklist cards ─────────────────
                buy_conds  = sig["buy_conditions"]
                sell_conds = sig["sell_conditions"]

                buy_labels = {
                    "di_bullish":  "🔵 +DI > 🟢 -DI  (Blue crosses Green)",
                    "adx_strong":  "💪 ADX > 25  (Strong trend)",
                    "above_vwap":  "📈 Price > VWAP",
                    "st_bullish":  "🟢 SuperTrend Bullish",
                    "above_pivot": "📍 Price > Pivot Point",
                    "rsi_ok":      f"📊 RSI in safe zone (30-70) — current {sig['rsi_val']:.1f}",
                    "vol_spike":   f"🔥 Volume spike > 1.5× avg — {sig['vol_now']:,.0f} vs {sig['vol_avg']:,.0f}",
                }
                sell_labels = {
                    "di_bearish":  "🟢 -DI > 🔵 +DI  (Green crosses Blue)",
                    "adx_strong":  "💪 ADX > 25  (Strong trend)",
                    "below_vwap":  "📉 Price < VWAP",
                    "st_bearish":  "🔴 SuperTrend Bearish",
                    "below_pivot": "📍 Price < Pivot Point",
                    "rsi_ok":      f"📊 RSI in safe zone (30-70) — current {sig['rsi_val']:.1f}",
                    "vol_spike":   f"🔥 Volume spike > 1.5× avg — {sig['vol_now']:,.0f} vs {sig['vol_avg']:,.0f}",
                }

                col_buy, col_sell = st.columns(2)

                with col_buy:
                    buy_s  = sig["buy_score"]
                    accent = "#00e676" if buy_s >= 5 else "#ffd700" if buy_s >= 3 else "#ff5252"
                    rows_h = "".join(
                        f'<div style="margin:4px 0">{"🟢" if v else "🔴"} {buy_labels[k]}</div>'
                        for k, v in buy_conds.items()
                    )
                    st.markdown(f"""
                    <div style="background:#1c1f26;border-radius:12px;padding:18px;
                                border:2px solid {accent}">
                      <div style="font-size:1.3rem;font-weight:800;color:{accent}">
                        {"🟢 BUY SIGNAL" if buy_s>=5 else "⏳ BUY — Waiting"} ({buy_s}/7)
                      </div>
                      <div style="margin-top:10px;font-size:0.85rem">{rows_h}</div>
                    </div>""", unsafe_allow_html=True)

                with col_sell:
                    sell_s = sig["sell_score"]
                    accent = "#ff5252" if sell_s >= 5 else "#ffd700" if sell_s >= 3 else "#555"
                    rows_h = "".join(
                        f'<div style="margin:4px 0">{"🟢" if v else "🔴"} {sell_labels[k]}</div>'
                        for k, v in sell_conds.items()
                    )
                    st.markdown(f"""
                    <div style="background:#1c1f26;border-radius:12px;padding:18px;
                                border:2px solid {accent}">
                      <div style="font-size:1.3rem;font-weight:800;color:{accent}">
                        {"🔴 SELL SIGNAL" if sell_s>=5 else "⏳ SELL — Waiting"} ({sell_s}/7)
                      </div>
                      <div style="margin-top:10px;font-size:0.85rem">{rows_h}</div>
                    </div>""", unsafe_allow_html=True)

                # ── Entry / Exit / Risk-Reward Panel ──────
                st.subheader("🎯 Entry, Stop-Loss & Targets (ATR-based)")

                if sig["entry"] is not None:
                    side     = "BUY" if sig["BUY"] else "SELL"
                    side_clr = "#00e676" if sig["BUY"] else "#ff5252"
                    risk     = abs(sig["entry"] - sig["stop"])
                    rr1      = abs(sig["target1"] - sig["entry"]) / risk if risk else 0
                    rr2      = abs(sig["target2"] - sig["entry"]) / risk if risk else 0
                    rr3      = abs(sig["target3"] - sig["entry"]) / risk if risk else 0

                    ec1, ec2, ec3 = st.columns(3)
                    with ec1:
                        st.markdown(f"""
                        <div style="background:#1c1f26;border-radius:10px;padding:16px;
                                    border-left:4px solid {side_clr}">
                          <div style="color:{side_clr};font-weight:800;font-size:1.2rem;margin-bottom:10px">
                            {side} ENTRY
                          </div>
                          <div style="font-size:0.9rem">
                            <strong>Entry:</strong> ${sig['entry']:,.2f}<br>
                            <strong>ATR (14):</strong> ${sig['atr_val']:,.2f}<br>
                            <strong>VWAP:</strong> ${sig['vwap_val']:,.2f}
                          </div>
                        </div>""", unsafe_allow_html=True)

                    with ec2:
                        st.markdown(f"""
                        <div style="background:#1c1f26;border-radius:10px;padding:16px;
                                    border-left:4px solid #FF1744">
                          <div style="color:#FF1744;font-weight:800;font-size:1.2rem;margin-bottom:10px">
                            STOP LOSS & TRAILING
                          </div>
                          <div style="font-size:0.9rem">
                            <strong>Stop Loss:</strong> ${sig['stop']:,.2f}
                              <span style="color:#888">(1.5× ATR)</span><br>
                            <strong>Risk:</strong> ${risk:,.2f}
                              <span style="color:#888">({risk/sig['entry']*100:.2f}%)</span><br>
                            <strong>Trailing Stop:</strong> ${sig['trailing']:,.2f}
                              <span style="color:#888">(1× ATR)</span>
                          </div>
                        </div>""", unsafe_allow_html=True)

                    with ec3:
                        st.markdown(f"""
                        <div style="background:#1c1f26;border-radius:10px;padding:16px;
                                    border-left:4px solid #00E676">
                          <div style="color:#00E676;font-weight:800;font-size:1.2rem;margin-bottom:10px">
                            TARGETS
                          </div>
                          <div style="font-size:0.9rem">
                            <strong>T1:</strong> ${sig['target1']:,.2f}
                              <span style="color:#888">(R:R {rr1:.1f}:1)</span><br>
                            <strong>T2:</strong> ${sig['target2']:,.2f}
                              <span style="color:#888">(R:R {rr2:.1f}:1)</span><br>
                            <strong>T3:</strong> ${sig['target3']:,.2f}
                              <span style="color:#888">(R:R {rr3:.1f}:1)</span>
                          </div>
                        </div>""", unsafe_allow_html=True)

                    st.markdown(f"""
                    <div style="margin-top:12px;background:#161a20;border-radius:8px;padding:12px 16px;
                                font-size:0.85rem;color:#888">
                      <strong style="color:#ffd700">Pro Tip:</strong>
                      Book 50% at T1, move SL to entry (breakeven). Book 30% at T2, trail the rest to T3.
                      Exit immediately if RSI crosses {"75" if sig["BUY"] else "25"} or
                      {"🟢-DI crosses above 🔵+DI" if sig["BUY"] else "🔵+DI crosses above 🟢-DI"}.
                    </div>""", unsafe_allow_html=True)

                else:
                    st.info("⏳ No active signal — need at least 5/7 conditions to trigger entry. "
                            "Waiting for alignment across ADX, DI crossover, VWAP, SuperTrend, "
                            "Pivot, RSI, and Volume.")

                # ── Individual indicator cards ────────────
                st.subheader("📊 Indicator Breakdown")
                sg1, sg2, sg3, sg4 = st.columns(4)

                with sg1:
                    if sig["adx_val"] > 25:
                        if sig["di_bullish"]:
                            txt = f"ADX {sig['adx_val']:.1f} — Strong 📈 (🔵+DI > 🟢-DI)"
                            kind = "buy"
                        else:
                            txt = f"ADX {sig['adx_val']:.1f} — Strong 📉 (🟢-DI > 🔵+DI)"
                            kind = "sell"
                    else:
                        txt, kind = f"ADX {sig['adx_val']:.1f} — Weak / Ranging", "neutral"
                    signal_box("📊 ADX + DI", txt, kind)

                with sg2:
                    if sig["dr"].iloc[-1] == -1:
                        signal_box("🟢 SuperTrend", "BULLISH — Above ST line", "buy")
                    else:
                        signal_box("🔴 SuperTrend", "BEARISH — Below ST line", "sell")

                with sg3:
                    if price_c > sig["vwap_val"]:
                        signal_box("💜 VWAP", f"Above VWAP ${sig['vwap_val']:,.2f}", "buy")
                    else:
                        signal_box("💜 VWAP", f"Below VWAP ${sig['vwap_val']:,.2f}", "sell")

                with sg4:
                    rv = sig["rsi_val"]
                    if rv > 70:
                        signal_box("📈 RSI", f"RSI {rv:.1f} — OVERBOUGHT ⚠️", "sell")
                    elif rv < 30:
                        signal_box("📉 RSI", f"RSI {rv:.1f} — OVERSOLD ⚠️", "buy")
                    else:
                        signal_box("📊 RSI", f"RSI {rv:.1f} — Neutral zone ✅", "neutral")

                # ── Pivot table + raw data ────────────────
                col_piv, col_vol = st.columns(2)
                with col_piv:
                    st.markdown("**📐 Pivot Levels**")
                    prows = [{"Level": k,
                              "Price ($)": f"${v:,.2f}",
                              "Distance": f"{((price_c-v)/v*100):+.2f}%",
                              "Position": "✅ Above" if price_c>=v else "❌ Below"}
                             for k, v in pivots_c.items()]
                    st.dataframe(pd.DataFrame(prows),
                                 use_container_width=True, hide_index=True)

                with col_vol:
                    st.markdown("**📊 Volume Analysis**")
                    vol_ratio = sig["vol_now"] / sig["vol_avg"] if sig["vol_avg"] > 0 else 0
                    st.markdown(f"""
                    <div style="background:#1c1f26;border-radius:10px;padding:16px">
                      <div>Current Volume: <strong>{sig['vol_now']:,.0f}</strong></div>
                      <div>20-bar Average: <strong>{sig['vol_avg']:,.0f}</strong></div>
                      <div>Ratio: <strong style="color:{'#00e676' if vol_ratio>1.5 else '#ffd700' if vol_ratio>1 else '#ff5252'}">{vol_ratio:.2f}×</strong>
                        {'🔥 Spike!' if vol_ratio>1.5 else '📈 Above avg' if vol_ratio>1 else '📉 Below avg'}</div>
                      <div style="margin-top:8px;color:#888;font-size:0.8rem">
                        Volume &gt; 1.5× average confirms strong breakout momentum.
                      </div>
                    </div>""", unsafe_allow_html=True)

                with st.expander(f"📋 {crypto_sym} OHLCV — last 50 candles"):
                    show_c = df_c.tail(50).copy()
                    show_c.index = show_c.index.strftime("%Y-%m-%d %H:%M")
                    st.dataframe(show_c.style.format("{:,.4f}"),
                                 use_container_width=True)

                st.caption(
                    f"🕒 Last updated: {datetime.now().strftime('%H:%M:%S')}  |  "
                    f"Source: Binance/Kraken  |  Volume 24h: ${vol24/1e6:.1f}M  |  "
                    f"⚠️ Not financial advice"
                )

            except Exception as ex:
                st.error(f"❌ Data error: {ex}")
    else:
        st.info("👆 Select a coin & timeframe, then click **Analyze**")
        st.markdown("""
        ### 🎯 Enhanced 7-Condition Strategy

        | # | Condition | BUY | SELL |
        |---|-----------|-----|------|
        | 1 | DI Crossover | 🔵 +DI > 🟢 -DI | 🟢 -DI > 🔵 +DI |
        | 2 | ADX Strength | ADX > 25 | ADX > 25 |
        | 3 | VWAP Position | Price > VWAP | Price < VWAP |
        | 4 | SuperTrend | Bullish (green) | Bearish (red) |
        | 5 | Pivot Point | Price > PP | Price < PP |
        | 6 | RSI Filter | 30 < RSI < 70 | 30 < RSI < 70 |
        | 7 | Volume | Vol > 1.5× avg | Vol > 1.5× avg |

        **Entry triggers at 5+ conditions. ATR-based stop-loss & 3 targets.**

        | Coin | What it tracks |
        |------|---------------|
        | BTC/USDT | Bitcoin — largest crypto |
        | ETH/USDT | Ethereum — DeFi leader |
        | SOL/USDT | Solana |
        | BNB/USDT | Binance Coin |
        """)
