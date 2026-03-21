#!/usr/bin/env python3
"""
Background alert scanner — runs independently of the Streamlit UI.
Checks BTC, ETH (and other coins) on Binance every N minutes,
sends email when 5+/7 strategy conditions are met.

Usage:
  python scanner.py                          # one-shot scan
  python scanner.py --loop --interval 900    # scan every 15 min

Environment variables (or .env file):
  SENDER_EMAIL      — your Gmail address
  GMAIL_APP_PASSWORD — Gmail App Password (not your normal password)
  RECIPIENT_EMAIL   — where to send alerts
  COINS             — comma-separated, default "BTCUSDT,ETHUSDT"
  TIMEFRAME         — Binance interval, default "15m"
  MIN_SCORE         — minimum conditions to trigger (default 5)
"""

import os, sys, time, json, argparse, smtplib, logging
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import requests
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("scanner")

ADX_STRONG = 25
BINANCE_BASE = "https://api.binance.com"

# ── Indicators (same logic as app.py) ─────────────────────

def calc_atr(high, low, close, period=14):
    tr = pd.concat([high - low, (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def calc_adx(df, period=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr  = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    up  = h.diff(); down = -l.diff()
    pdm = pd.Series(np.where((up>down)&(up>0), up, 0.0), index=df.index)
    mdm = pd.Series(np.where((down>up)&(down>0), down, 0.0), index=df.index)
    a = 1/period
    tr14  = tr.ewm(alpha=a, adjust=False).mean()
    pdm14 = pdm.ewm(alpha=a, adjust=False).mean()
    mdm14 = mdm.ewm(alpha=a, adjust=False).mean()
    pdi   = 100*pdm14/tr14
    mdi   = 100*mdm14/tr14
    dx    = 100*(pdi-mdi).abs()/(pdi+mdi).replace(0, np.nan)
    adx   = dx.ewm(alpha=a, adjust=False).mean()
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
            else:              dr[i] =  1; stl[i] = fub[i]
        else:
            if c[i] < flb[i]: dr[i] =  1; stl[i] = fub[i]
            else:              dr[i] = -1; stl[i] = flb[i]
    return pd.Series(stl, index=df.index), pd.Series(dr, index=df.index)

def calc_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_l = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_g / avg_l.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).round(2)

def calc_vwap(df):
    tp  = (df["high"] + df["low"] + df["close"]) / 3
    cvv = (tp * df["volume"]).cumsum()
    cv  = df["volume"].cumsum().replace(0, np.nan)
    return (cvv / cv).round(4)

def calc_pivots(df):
    h = df["high"].iloc[-2]; l = df["low"].iloc[-2]; c = df["close"].iloc[-2]
    pp = (h+l+c)/3
    return {"PP": pp, "R1": 2*pp-l, "S1": 2*pp-h,
            "R2": pp+(h-l), "S2": pp-(h-l),
            "R3": h+2*(pp-l), "S3": l-2*(h-pp)}

# ── Crypto fetch (Binance → Bybit fallback) ──────────────

BYBIT_INTERVAL_MAP = {"5m": "5", "15m": "15", "1h": "60", "4h": "240", "1d": "D"}

def _fetch_binance(symbol, interval, limit):
    r = requests.get(f"{BINANCE_BASE}/api/v3/klines",
                     params={"symbol": symbol, "interval": interval, "limit": limit},
                     timeout=15)
    r.raise_for_status()
    df = pd.DataFrame(r.json(), columns=[
        "datetime","open","high","low","close","volume",
        "ct","qav","nt","tbbav","tbqav","ignore"])
    df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
    df = df.set_index("datetime")[["open","high","low","close","volume"]]
    return df.astype(float).sort_index()

def _fetch_bybit(symbol, interval, limit):
    bybit_iv = BYBIT_INTERVAL_MAP.get(interval, "15")
    r = requests.get("https://api.bybit.com/v5/market/kline",
                     params={"category": "spot", "symbol": symbol,
                             "interval": bybit_iv, "limit": limit},
                     timeout=15)
    r.raise_for_status()
    rows = r.json().get("result", {}).get("list", [])
    if not rows:
        raise Exception("No data from Bybit")
    df = pd.DataFrame(rows, columns=[
        "datetime", "open", "high", "low", "close", "volume", "turnover"])
    df["datetime"] = pd.to_datetime(df["datetime"].astype(int), unit="ms")
    df = df.set_index("datetime")[["open","high","low","close","volume"]]
    return df.astype(float).sort_index()

def fetch_candles(symbol, interval="15m", limit=500):
    """Try Binance first; fall back to Bybit if geo-blocked (HTTP 451/403)."""
    try:
        return _fetch_binance(symbol, interval, limit)
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code in (451, 403):
            log.info(f"  Binance blocked (HTTP {e.response.status_code}), using Bybit…")
            return _fetch_bybit(symbol, interval, limit)
        raise

# ── Enhanced signal ───────────────────────────────────────

def analyse(df, adx_period=14, st_period=10, st_mult=3.0):
    price = df["close"].iloc[-1]
    adx, pdi, mdi = calc_adx(df, adx_period)
    _, dr         = calc_supertrend(df, st_period, st_mult)
    rsi           = calc_rsi(df["close"], 14)
    vwap          = calc_vwap(df)
    atr           = calc_atr(df["high"], df["low"], df["close"], 14)
    pivots        = calc_pivots(df)

    vol_avg   = df["volume"].rolling(20).mean().iloc[-1]
    vol_now   = df["volume"].iloc[-1]
    vol_spike = vol_now > (vol_avg * 1.5) if vol_avg > 0 else False

    di_bullish = pdi.iloc[-1] > mdi.iloc[-1]
    di_bearish = mdi.iloc[-1] > pdi.iloc[-1]
    rsi_val    = rsi.iloc[-1]
    atr_val    = atr.iloc[-1]
    vwap_val   = vwap.iloc[-1]

    buy_conds = {
        "+DI > -DI (Blue > Green)": di_bullish,
        "ADX > 25":                 adx.iloc[-1] > ADX_STRONG,
        "Price > VWAP":             price > vwap_val,
        "SuperTrend Bullish":       dr.iloc[-1] == -1,
        "Price > Pivot PP":         price > pivots["PP"],
        f"RSI safe zone ({rsi_val:.1f})": 30 <= rsi_val <= 70,
        "Volume spike":             vol_spike,
    }
    sell_conds = {
        "-DI > +DI (Green > Blue)": di_bearish,
        "ADX > 25":                 adx.iloc[-1] > ADX_STRONG,
        "Price < VWAP":             price < vwap_val,
        "SuperTrend Bearish":       dr.iloc[-1] == 1,
        "Price < Pivot PP":         price < pivots["PP"],
        f"RSI safe zone ({rsi_val:.1f})": 30 <= rsi_val <= 70,
        "Volume spike":             vol_spike,
    }

    buy_score  = sum(buy_conds.values())
    sell_score = sum(sell_conds.values())

    return {
        "price": price, "adx": adx.iloc[-1], "pdi": pdi.iloc[-1], "mdi": mdi.iloc[-1],
        "rsi": rsi_val, "vwap": vwap_val, "atr": atr_val,
        "vol_now": vol_now, "vol_avg": vol_avg,
        "buy_conds": buy_conds, "sell_conds": sell_conds,
        "buy_score": buy_score, "sell_score": sell_score,
        "entry": price,
        "stop_buy":  price - 1.5*atr_val,  "stop_sell":  price + 1.5*atr_val,
        "t1_buy":    price + 1.5*atr_val,  "t1_sell":    price - 1.5*atr_val,
        "t2_buy":    price + 3.0*atr_val,  "t2_sell":    price - 3.0*atr_val,
        "t3_buy":    price + 4.5*atr_val,  "t3_sell":    price - 4.5*atr_val,
    }

# ── Email ─────────────────────────────────────────────────

def send_alert(sender, password, recipient, symbol, sig, side, timeframe):
    is_buy = side == "BUY"
    accent = "#00e676" if is_buy else "#ff5252"
    emoji  = "🟢" if is_buy else "🔴"
    score  = sig["buy_score"] if is_buy else sig["sell_score"]
    conds  = sig["buy_conds"] if is_buy else sig["sell_conds"]
    stop   = sig["stop_buy"]  if is_buy else sig["stop_sell"]
    t1     = sig["t1_buy"]    if is_buy else sig["t1_sell"]
    t2     = sig["t2_buy"]    if is_buy else sig["t2_sell"]
    t3     = sig["t3_buy"]    if is_buy else sig["t3_sell"]

    now_str = datetime.now().strftime("%d %b %Y  %H:%M:%S")
    cond_rows = "".join(
        f'<tr><td style="padding:6px 8px">{"✅" if v else "❌"} {k}</td></tr>'
        for k, v in conds.items()
    )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"{emoji} {side} Alert — {symbol} @ ${sig['price']:,.2f} ({score}/7)"
    msg["From"]    = sender
    msg["To"]      = recipient
    html = f"""
    <html><body style="font-family:Arial,sans-serif;background:#0e1117;color:#c9d1d9;padding:20px">
    <div style="max-width:560px;margin:auto;background:#1c1f26;border-radius:14px;
                border:2px solid {accent};overflow:hidden">
      <div style="background:{'#0d2b1a' if is_buy else '#2b0d0d'};padding:20px 24px">
        <h1 style="color:{accent};margin:0">{emoji} {side} SIGNAL — {symbol}</h1>
        <p style="color:#888;margin:4px 0 0">{now_str}  ·  {timeframe}  ·  Score {score}/7</p>
      </div>
      <div style="padding:20px 24px">
        <table style="width:100%;border-collapse:collapse;margin-bottom:16px">
          <tr><td style="padding:6px 0;color:#888">Price</td>
              <td style="font-weight:700">${sig['price']:,.2f}</td></tr>
          <tr><td style="padding:6px 0;color:#888">ADX</td>
              <td>{sig['adx']:.1f}</td></tr>
          <tr><td style="padding:6px 0;color:#888">+DI / -DI</td>
              <td>{sig['pdi']:.1f} / {sig['mdi']:.1f}</td></tr>
          <tr><td style="padding:6px 0;color:#888">RSI</td>
              <td>{sig['rsi']:.1f}</td></tr>
          <tr><td style="padding:6px 0;color:#888">VWAP</td>
              <td>${sig['vwap']:,.2f}</td></tr>
        </table>
        <h3 style="color:{accent};margin:0 0 8px">Strategy Conditions</h3>
        <table style="width:100%;font-size:0.9rem">{cond_rows}</table>
        <hr style="border-color:#333;margin:16px 0">
        <h3 style="color:#ffd700;margin:0 0 8px">🎯 Entry & Exit Levels</h3>
        <table style="width:100%;border-collapse:collapse;font-size:0.9rem">
          <tr><td style="padding:6px 0"><strong>Entry</strong></td>
              <td>${sig['entry']:,.2f}</td></tr>
          <tr><td style="padding:6px 0;color:#ff5252"><strong>Stop Loss</strong></td>
              <td style="color:#ff5252">${stop:,.2f} (1.5× ATR)</td></tr>
          <tr><td style="padding:6px 0;color:#00e676"><strong>Target 1</strong></td>
              <td style="color:#00e676">${t1:,.2f} (1:1 R:R)</td></tr>
          <tr><td style="padding:6px 0;color:#69f0ae"><strong>Target 2</strong></td>
              <td style="color:#69f0ae">${t2:,.2f} (1:2 R:R)</td></tr>
          <tr><td style="padding:6px 0;color:#b9f6ca"><strong>Target 3</strong></td>
              <td style="color:#b9f6ca">${t3:,.2f} (1:3 R:R)</td></tr>
        </table>
        <p style="color:#888;font-size:0.78rem;margin-top:20px">
          ⚠️ Automated technical alert. Not financial advice.
          Book 50% at T1, move SL to entry, trail rest.</p>
      </div>
    </div></body></html>"""
    msg.attach(MIMEText(html, "html"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(sender, password)
        s.send_message(msg)
    log.info(f"  📧 Email sent to {recipient}")

# ── Main scanner ──────────────────────────────────────────

def scan_once(coins, timeframe, min_score, sender, password, recipient):
    for symbol in coins:
        try:
            log.info(f"Scanning {symbol} on {timeframe}…")
            df  = fetch_candles(symbol, timeframe)
            sig = analyse(df)

            buy_s  = sig["buy_score"]
            sell_s = sig["sell_score"]
            log.info(f"  {symbol}: price=${sig['price']:,.2f}  "
                     f"BUY={buy_s}/7  SELL={sell_s}/7  "
                     f"ADX={sig['adx']:.1f}  RSI={sig['rsi']:.1f}")

            if buy_s >= min_score:
                log.info(f"  🟢 BUY signal triggered ({buy_s}/7)")
                if sender and password and recipient:
                    send_alert(sender, password, recipient, symbol, sig, "BUY", timeframe)
                else:
                    log.warning("  Email not configured — skipping send")

            elif sell_s >= min_score:
                log.info(f"  🔴 SELL signal triggered ({sell_s}/7)")
                if sender and password and recipient:
                    send_alert(sender, password, recipient, symbol, sig, "SELL", timeframe)
                else:
                    log.warning("  Email not configured — skipping send")

            else:
                log.info(f"  ⏳ No signal — waiting")

        except Exception as e:
            log.error(f"  Error scanning {symbol}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Crypto alert scanner")
    parser.add_argument("--loop", action="store_true",
                        help="Run continuously")
    parser.add_argument("--interval", type=int, default=900,
                        help="Seconds between scans (default 900 = 15 min)")
    parser.add_argument("--coins", default=None,
                        help="Comma-separated Binance symbols")
    parser.add_argument("--timeframe", default=None,
                        help="Binance interval (5m, 15m, 1h, 4h, 1d)")
    parser.add_argument("--min-score", type=int, default=None,
                        help="Minimum conditions to trigger alert (default 5)")
    args = parser.parse_args()

    sender    = os.getenv("SENDER_EMAIL", "")
    password  = os.getenv("GMAIL_APP_PASSWORD", "")
    recipient = os.getenv("RECIPIENT_EMAIL", "")
    coins_str = args.coins or os.getenv("COINS", "BTCUSDT,ETHUSDT")
    timeframe = args.timeframe or os.getenv("TIMEFRAME", "15m")
    min_score = args.min_score or int(os.getenv("MIN_SCORE", "5"))
    coins     = [c.strip() for c in coins_str.split(",")]

    log.info("=" * 50)
    log.info(f"MCX/Crypto Alert Scanner")
    log.info(f"Coins: {coins}  |  Timeframe: {timeframe}  |  Min score: {min_score}/7")
    log.info(f"Email: {'configured ✅' if (sender and password and recipient) else 'NOT configured ❌'}")
    log.info("=" * 50)

    if args.loop:
        log.info(f"Loop mode — scanning every {args.interval}s ({args.interval//60} min)")
        while True:
            scan_once(coins, timeframe, min_score, sender, password, recipient)
            log.info(f"Sleeping {args.interval}s until next scan…\n")
            time.sleep(args.interval)
    else:
        scan_once(coins, timeframe, min_score, sender, password, recipient)


if __name__ == "__main__":
    main()
