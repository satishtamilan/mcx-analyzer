#!/usr/bin/env python3
"""
Automated BTC V3 Sniper trader via Alpaca paper trading.
Runs continuously, scans every hour on candle close, places orders automatically.

Usage:
  python alpaca_trader.py                        # one-shot scan
  python alpaca_trader.py --loop                 # scan every 1h (default)
  python alpaca_trader.py --loop --interval 3600 # custom interval in seconds

Environment variables:
  ALPACA_API_KEY        — Alpaca paper trading API key
  ALPACA_SECRET_KEY     — Alpaca paper trading secret key
  RISK_PCT              — risk per trade as % of equity (default 2)
  SENDER_EMAIL          — (optional) Gmail for trade notifications
  GMAIL_APP_PASSWORD    — (optional) Gmail App Password
  RECIPIENT_EMAIL       — (optional) alert recipient
"""

import os, sys, time, argparse, logging, smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import requests
import pandas as pd
import numpy as np

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("alpaca_trader")

# ── Indicators ────────────────────────────────────────────

def calc_atr(high, low, close, period=14):
    tr = pd.concat([high - low, (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def calc_adx(df, period=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    up = h.diff(); down = -l.diff()
    pdm = pd.Series(np.where((up>down)&(up>0), up, 0.0), index=df.index)
    mdm = pd.Series(np.where((down>up)&(down>0), down, 0.0), index=df.index)
    a = 1/period
    tr14 = tr.ewm(alpha=a, adjust=False).mean()
    pdm14 = pdm.ewm(alpha=a, adjust=False).mean()
    mdm14 = mdm.ewm(alpha=a, adjust=False).mean()
    pdi = 100*pdm14/tr14
    mdi = 100*mdm14/tr14
    dx = 100*(pdi-mdi).abs()/(pdi+mdi).replace(0, np.nan)
    adx = dx.ewm(alpha=a, adjust=False).mean()
    return adx.round(2), pdi.round(2), mdi.round(2)

def calc_supertrend(df, period=10, multiplier=3.0):
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    atr = calc_atr(df["high"], df["low"], df["close"], period).values
    hl2 = (h+l)/2
    ub = hl2 + multiplier*atr; lb = hl2 - multiplier*atr
    fub = ub.copy(); flb = lb.copy()
    stl = np.zeros(len(df)); dr = np.ones(len(df), dtype=int)
    for i in range(1, len(df)):
        fub[i] = ub[i] if (ub[i]<fub[i-1] or c[i-1]>fub[i-1]) else fub[i-1]
        flb[i] = lb[i] if (lb[i]>flb[i-1] or c[i-1]<flb[i-1]) else flb[i-1]
        prev = stl[i-1] if i > 1 else fub[0]
        if prev == fub[i-1]:
            if c[i] > fub[i]: dr[i] = -1; stl[i] = flb[i]
            else:              dr[i] = 1;  stl[i] = fub[i]
        else:
            if c[i] < flb[i]: dr[i] = 1;  stl[i] = fub[i]
            else:              dr[i] = -1; stl[i] = flb[i]
    return pd.Series(stl, index=df.index), pd.Series(dr, index=df.index)

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_g = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_l = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_g / avg_l.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).round(2)

def calc_vwap(df):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    cvv = (tp * df["volume"]).cumsum()
    cv = df["volume"].cumsum().replace(0, np.nan)
    return (cvv / cv).round(4)

def calc_bbands(series, period=20, std_mult=2.0):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return sma, sma + std_mult * std, sma - std_mult * std

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

# ── Crypto data fetch (Binance → Kraken fallback) ────────

KRAKEN_SYMBOL_MAP = {"BTCUSDT": "XBTUSDT", "ETHUSDT": "ETHUSDT",
                     "SOLUSDT": "SOLUSDT", "BNBUSDT": "BNBUSDT"}
KRAKEN_INTERVAL_MAP = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}

def _fetch_binance(symbol, interval, limit):
    r = requests.get("https://api.binance.com/api/v3/klines",
                     params={"symbol": symbol, "interval": interval, "limit": limit},
                     timeout=15)
    r.raise_for_status()
    df = pd.DataFrame(r.json(), columns=[
        "datetime","open","high","low","close","volume",
        "ct","qav","nt","tbbav","tbqav","ignore"])
    df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
    df = df.set_index("datetime")[["open","high","low","close","volume"]]
    return df.astype(float).sort_index()

def _fetch_kraken(symbol, interval, limit):
    from datetime import timedelta
    kr_sym = KRAKEN_SYMBOL_MAP.get(symbol, symbol)
    kr_iv = KRAKEN_INTERVAL_MAP.get(interval, 60)
    since = int((datetime.now() - timedelta(minutes=kr_iv * limit)).timestamp())
    r = requests.get("https://api.kraken.com/0/public/OHLC",
                     params={"pair": kr_sym, "interval": kr_iv, "since": since},
                     timeout=15)
    r.raise_for_status()
    data = r.json()
    if data.get("error"):
        raise Exception(f"Kraken: {data['error']}")
    pair_key = [k for k in data["result"].keys() if k != "last"][0]
    rows = data["result"][pair_key]
    df = pd.DataFrame(rows, columns=[
        "datetime","open","high","low","close","vwap","volume","count"])
    df["datetime"] = pd.to_datetime(df["datetime"].astype(int), unit="s")
    df = df.set_index("datetime")[["open","high","low","close","volume"]]
    return df.astype(float).sort_index()

def fetch_candles(symbol="BTCUSDT", interval="1h", limit=1000):
    try:
        return _fetch_binance(symbol, interval, limit)
    except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError):
        log.info("Binance unavailable, switching to Kraken…")
        return _fetch_kraken(symbol, interval, min(limit, 720))

# ── V3 Signal Detection ──────────────────────────────────

def detect_v3_signal(df):
    """Run V3 BTC Sniper logic on the last bar. Returns signal dict or None."""
    ema9 = calc_ema(df["close"], 9)
    ema21 = calc_ema(df["close"], 21)
    ema50 = calc_ema(df["close"], 50)
    ema200 = calc_ema(df["close"], 200)
    adx, pdi, mdi = calc_adx(df, 14)
    _, dr = calc_supertrend(df, 10, 3.0)
    rsi = calc_rsi(df["close"], 14)
    atr = calc_atr(df["high"], df["low"], df["close"], 14)
    vwap = calc_vwap(df)
    _, bb_upper, bb_lower = calc_bbands(df["close"], 20, 2.0)
    _, _, macd_hist = calc_macd(df["close"])

    i = len(df) - 1
    if i < 210:
        return None

    price = df["close"].iloc[i]
    hi = df["high"].iloc[i]
    lo = df["low"].iloc[i]
    opn = df["open"].iloc[i]
    atr_v = atr.iloc[i]

    if atr_v <= 0 or np.isnan(atr_v):
        return None

    macro_bull = ema50.iloc[i] > ema200.iloc[i] and ema200.iloc[i] > ema200.iloc[i-5]
    macro_bear = ema50.iloc[i] < ema200.iloc[i] and ema200.iloc[i] < ema200.iloc[i-5]

    if not macro_bull and not macro_bear:
        return None

    cr = hi - lo
    close_pos_b = (price - lo) / cr if cr > 0 else 0
    close_pos_s = (hi - price) / cr if cr > 0 else 0

    buy_rev = (price > opn and df["close"].iloc[i-1] < df["open"].iloc[i-1]
               and macro_bull and close_pos_b > 0.55)
    sell_rev = (price < opn and df["close"].iloc[i-1] > df["open"].iloc[i-1]
                and macro_bear and close_pos_s > 0.55)

    if not buy_rev and not sell_rev:
        return None

    prev_lo = df["low"].iloc[i-1]
    prev_hi = df["high"].iloc[i-1]
    tol = atr_v * 0.3

    conf_buy = sum([prev_lo <= bb_lower.iloc[i-1] + tol,
                    prev_lo <= vwap.iloc[i-1] + tol,
                    prev_lo <= ema21.iloc[i-1] + tol,
                    prev_lo <= ema50.iloc[i-1] + tol])
    conf_sell = sum([prev_hi >= bb_upper.iloc[i-1] - tol,
                     prev_hi >= vwap.iloc[i-1] - tol,
                     prev_hi >= ema21.iloc[i-1] - tol,
                     prev_hi >= ema50.iloc[i-1] - tol])

    rsi_low = any(rsi.iloc[i-j] < 40 for j in range(1, 4))
    rsi_high = any(rsi.iloc[i-j] > 60 for j in range(1, 4))
    macd_up = macd_hist.iloc[i] > macd_hist.iloc[i-1]
    macd_dn = macd_hist.iloc[i] < macd_hist.iloc[i-1]
    adx_ok = adx.iloc[i] > 20
    vol_avg = df["volume"].iloc[max(0,i-20):i].mean()
    vol_ok = df["volume"].iloc[i] > vol_avg * 0.8 if vol_avg > 0 else False
    cb = abs(price - opn)
    strong_b = cb > cr * 0.4 if cr > 0 else False

    if buy_rev:
        sc = sum([conf_buy >= 2, rsi_low, rsi.iloc[i] > rsi.iloc[i-1],
                  macd_up, adx_ok, pdi.iloc[i] > mdi.iloc[i],
                  dr.iloc[i] == -1, vol_ok, strong_b])
        if sc >= 6:
            return {
                "side": "BUY", "score": sc, "price": price, "atr": atr_v,
                "stop": price - 1.2 * atr_v,
                "t1": price + 0.6 * atr_v,
                "t2": price + 1.5 * atr_v,
                "trend": "Bullish", "adx": round(adx.iloc[i], 1),
                "rsi": round(rsi.iloc[i], 1),
            }

    if sell_rev:
        sc = sum([conf_sell >= 2, rsi_high, rsi.iloc[i] < rsi.iloc[i-1],
                  macd_dn, adx_ok, mdi.iloc[i] > pdi.iloc[i],
                  dr.iloc[i] == 1, vol_ok, strong_b])
        if sc >= 6:
            return {
                "side": "SELL", "score": sc, "price": price, "atr": atr_v,
                "stop": price + 1.2 * atr_v,
                "t1": price - 0.6 * atr_v,
                "t2": price - 1.5 * atr_v,
                "trend": "Bearish", "adx": round(adx.iloc[i], 1),
                "rsi": round(rsi.iloc[i], 1),
            }

    return None

# ── Trade State (persists across ticks) ───────────────────

trade_state = {
    "entry_atr": None,
    "t1_hit": False,
    "stop": None,
    "t1": None,
    "t2": None,
    "peak": None,
    "trough": None,
    "side": None,
}

def init_trade_state(entry, atr_v, side):
    """Initialize trade state when a new position is opened."""
    trade_state["entry_atr"] = atr_v
    trade_state["t1_hit"] = False
    trade_state["side"] = side
    trade_state["peak"] = entry
    trade_state["trough"] = entry
    if side == "LONG":
        trade_state["stop"] = entry - 1.2 * atr_v
        trade_state["t1"] = entry + 0.6 * atr_v
        trade_state["t2"] = entry + 1.5 * atr_v
    else:
        trade_state["stop"] = entry + 1.2 * atr_v
        trade_state["t1"] = entry - 0.6 * atr_v
        trade_state["t2"] = entry - 1.5 * atr_v

def reset_trade_state():
    for k in trade_state:
        trade_state[k] = None
    trade_state["t1_hit"] = False

# ── Trade Management (runs every minute) ──────────────────

def manage_position(client):
    """Check open BTC position with live price, manage SL/T1/T2/trail."""
    positions = client.get_all_positions()
    btc_pos = None
    for p in positions:
        if p.symbol in ("BTC/USD", "BTCUSD"):
            btc_pos = p
            break

    if btc_pos is None:
        if trade_state["side"] is not None:
            reset_trade_state()
        return None

    entry = float(btc_pos.avg_entry_price)
    current = float(btc_pos.current_price)
    qty = float(btc_pos.qty)
    side = btc_pos.side.value.upper()
    unr_pnl = float(btc_pos.unrealized_pl)

    if trade_state["side"] is None:
        log.info("Detected existing position without trade state — initializing from last ATR")
        try:
            df_quick = fetch_candles("BTCUSDT", "1h", 50)
            atr_v = calc_atr(df_quick["high"], df_quick["low"], df_quick["close"], 14).iloc[-1]
        except Exception:
            atr_v = abs(entry * 0.015)
        init_trade_state(entry, atr_v, side)

    atr_v = trade_state["entry_atr"]

    if side == "LONG":
        trade_state["peak"] = max(trade_state.get("peak", entry), current)

        if current <= trade_state["stop"]:
            pnl_type = "TRAIL_STOP" if trade_state["t1_hit"] else "SL_HIT"
            log.warning(f"{pnl_type} — closing LONG at ${current:,.2f} (stop: ${trade_state['stop']:,.2f})")
            client.close_position("BTC/USD")
            result = {"action": pnl_type, "pnl": unr_pnl, "price": current}
            reset_trade_state()
            return result

        if current >= trade_state["t2"]:
            log.info(f"T2 HIT — closing LONG at ${current:,.2f}")
            client.close_position("BTC/USD")
            result = {"action": "T2_HIT", "pnl": unr_pnl, "price": current}
            reset_trade_state()
            return result

        if current >= trade_state["t1"] and not trade_state["t1_hit"]:
            trade_state["t1_hit"] = True
            trade_state["stop"] = entry + 0.25 * atr_v
            log.info(f"T1 HIT — stop moved to ${trade_state['stop']:,.2f} (locked profit)")

        if trade_state["t1_hit"]:
            trail = trade_state["peak"] - 0.35 * atr_v
            if trail > trade_state["stop"]:
                trade_state["stop"] = trail
                log.info(f"Trail updated: ${trail:,.2f} (peak: ${trade_state['peak']:,.2f})")

    else:
        trade_state["trough"] = min(trade_state.get("trough", entry), current)

        if current >= trade_state["stop"]:
            pnl_type = "TRAIL_STOP" if trade_state["t1_hit"] else "SL_HIT"
            log.warning(f"{pnl_type} — closing SHORT at ${current:,.2f} (stop: ${trade_state['stop']:,.2f})")
            client.close_position("BTC/USD")
            result = {"action": pnl_type, "pnl": unr_pnl, "price": current}
            reset_trade_state()
            return result

        if current <= trade_state["t2"]:
            log.info(f"T2 HIT — closing SHORT at ${current:,.2f}")
            client.close_position("BTC/USD")
            result = {"action": "T2_HIT", "pnl": unr_pnl, "price": current}
            reset_trade_state()
            return result

        if current <= trade_state["t1"] and not trade_state["t1_hit"]:
            trade_state["t1_hit"] = True
            trade_state["stop"] = entry - 0.25 * atr_v
            log.info(f"T1 HIT — stop moved to ${trade_state['stop']:,.2f} (locked profit)")

        if trade_state["t1_hit"]:
            trail = trade_state["trough"] + 0.35 * atr_v
            if trail < trade_state["stop"]:
                trade_state["stop"] = trail
                log.info(f"Trail updated: ${trail:,.2f} (trough: ${trade_state['trough']:,.2f})")

    t1_lbl = "✅" if trade_state["t1_hit"] else "⏳"
    log.info(f"  {side} {qty:.6f} BTC | Entry: ${entry:,.2f} | Now: ${current:,.2f} | "
             f"P&L: ${unr_pnl:,.2f} | Stop: ${trade_state['stop']:,.2f} | T1:{t1_lbl}")
    return {"action": "HOLDING", "pnl": unr_pnl, "price": current}

# ── Email Notification ────────────────────────────────────

def send_email(subject, body):
    sender = os.getenv("SENDER_EMAIL")
    password = os.getenv("GMAIL_APP_PASSWORD")
    recipient = os.getenv("RECIPIENT_EMAIL")
    if not all([sender, password, recipient]):
        return

    try:
        msg = MIMEMultipart()
        msg["From"] = sender
        msg["To"] = recipient
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.send_message(msg)
        log.info(f"Email sent: {subject}")
    except Exception as e:
        log.error(f"Email failed: {e}")

# ── Position Check (every minute) ─────────────────────────

def tick_position(client):
    """Quick position check using Alpaca live price. No candle fetch needed."""
    try:
        result = manage_position(client)
        if result and result["action"] in ("SL_HIT", "T2_HIT", "TRAIL_STOP"):
            emoji = {"SL_HIT": "🛑", "T2_HIT": "🎯", "TRAIL_STOP": "🔒"}[result["action"]]
            send_email(
                f"{emoji} BTC Trade Closed — {result['action']}",
                f"<h2>{result['action']}</h2>"
                f"<p>Price: ${result['price']:,.2f}<br>"
                f"P&L: ${result['pnl']:,.2f}</p>"
            )
            return result
    except Exception as e:
        log.error(f"Position check error: {e}")
    return None

# ── Signal Scan (every hour on candle close) ──────────────

def scan_for_signal(client, risk_pct):
    """Fetch 1h candles, run V3 detection, place order if signal found."""
    log.info("=" * 60)
    log.info("HOURLY SCAN — checking for V3 entry signal…")

    positions = client.get_all_positions()
    has_btc = any(p.symbol in ("BTC/USD", "BTCUSD") for p in positions)
    if has_btc:
        log.info("Already in a position — skipping signal scan.")
        return

    try:
        df = fetch_candles("BTCUSDT", "1h", 1000)
    except Exception as e:
        log.error(f"Failed to fetch candles: {e}")
        return

    log.info(f"Fetched {len(df)} candles — "
             f"{df.index[0].strftime('%Y-%m-%d %H:%M')} → "
             f"{df.index[-1].strftime('%Y-%m-%d %H:%M')}")

    signal = detect_v3_signal(df)

    if signal is None:
        log.info("No V3 signal. Trend: %s | ADX: %.1f | RSI: %.1f",
                 "Bull" if df["close"].iloc[-1] > calc_ema(df["close"], 200).iloc[-1] else "Bear",
                 calc_adx(df, 14)[0].iloc[-1],
                 calc_rsi(df["close"], 14).iloc[-1])
        return

    log.info(f"🎯 V3 SIGNAL: {signal['side']} | Score: {signal['score']}/9 | "
             f"Price: ${signal['price']:,.2f}")

    try:
        acct = client.get_account()
        equity = float(acct.equity)
        risk_amt = equity * risk_pct / 100
        sl_dist = 1.2 * signal["atr"]
        qty_btc = round(risk_amt / sl_dist, 6)
        qty_btc = max(qty_btc, 0.0001)

        order_side = OrderSide.BUY if signal["side"] == "BUY" else OrderSide.SELL
        client.submit_order(MarketOrderRequest(
            symbol="BTC/USD",
            qty=qty_btc,
            side=order_side,
            time_in_force=TimeInForce.GTC,
        ))

        alpaca_side = "LONG" if signal["side"] == "BUY" else "SHORT"
        init_trade_state(signal["price"], signal["atr"], alpaca_side)

        log.info(f"✅ ORDER PLACED: {signal['side']} {qty_btc:.6f} BTC")
        log.info(f"   Stop: ${trade_state['stop']:,.2f} | T1: ${trade_state['t1']:,.2f} | T2: ${trade_state['t2']:,.2f}")

        send_email(
            f"🎯 BTC {signal['side']} Order Placed — V3 Sniper",
            f"<h2 style='color:{'green' if signal['side']=='BUY' else 'red'}'>"
            f"{signal['side']} BTC</h2>"
            f"<table>"
            f"<tr><td><b>Price:</b></td><td>${signal['price']:,.2f}</td></tr>"
            f"<tr><td><b>Qty:</b></td><td>{qty_btc:.6f} BTC (${qty_btc*signal['price']:,.2f})</td></tr>"
            f"<tr><td><b>Score:</b></td><td>{signal['score']}/9</td></tr>"
            f"<tr><td><b>Stop Loss:</b></td><td>${signal['stop']:,.2f}</td></tr>"
            f"<tr><td><b>T1:</b></td><td>${signal['t1']:,.2f}</td></tr>"
            f"<tr><td><b>T2:</b></td><td>${signal['t2']:,.2f}</td></tr>"
            f"<tr><td><b>Trend:</b></td><td>{signal['trend']}</td></tr>"
            f"<tr><td><b>ADX:</b></td><td>{signal['adx']}</td></tr>"
            f"<tr><td><b>RSI:</b></td><td>{signal['rsi']}</td></tr>"
            f"</table>"
        )

    except Exception as e:
        log.error(f"Order failed: {e}")
        send_email("❌ BTC Order Failed", f"<p>Error: {e}</p>")

# ── Main Loop ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="V3 BTC Sniper — Alpaca Auto Trader")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    args = parser.parse_args()

    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY")
    risk_pct = float(os.getenv("RISK_PCT", "2"))

    if not api_key or not api_secret:
        log.error("Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
        sys.exit(1)

    client = TradingClient(api_key, api_secret, paper=True)

    try:
        acct = client.get_account()
        log.info("Connected to Alpaca Paper Trading")
        log.info(f"  Equity: ${float(acct.equity):,.2f}")
        log.info(f"  Cash: ${float(acct.cash):,.2f}")
        log.info(f"  Risk per trade: {risk_pct}%")
    except Exception as e:
        log.error(f"Alpaca connection failed: {e}")
        sys.exit(1)

    if not args.loop:
        tick_position(client)
        minute = datetime.now().minute
        if minute < 10:
            scan_for_signal(client, risk_pct)
        else:
            positions = client.get_all_positions()
            has_btc = any(p.symbol in ("BTC/USD", "BTCUSD") for p in positions)
            if has_btc:
                log.info("Position monitored. Next signal scan at the top of the hour.")
            else:
                log.info(f"No position, no signal scan (minute={minute}, scans at :00-:09). Waiting.")
        return

    TICK_INTERVAL = 60
    SIGNAL_INTERVAL = 3600
    last_signal_scan = 0

    log.info("Starting auto-trader:")
    log.info("  Position check: every 60 seconds")
    log.info("  Signal scan: every 1 hour (on candle close)")
    log.info("-" * 60)

    while True:
        now = time.time()

        tick_position(client)

        if now - last_signal_scan >= SIGNAL_INTERVAL:
            scan_for_signal(client, risk_pct)
            last_signal_scan = now

        time.sleep(TICK_INTERVAL)


if __name__ == "__main__":
    main()
