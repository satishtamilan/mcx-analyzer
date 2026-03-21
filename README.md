# 📊 MCX & Crypto Analyzer — Pro Strategy Tester

Live candlestick dashboard with a **7-condition trading strategy** for MCX Gold/Silver and Crypto (BTC, ETH).  
Includes **VWAP, RSI, ADX, SuperTrend, Pivot Points, Volume** — with ATR-based entry, stop-loss, and targets.

---

## 🚀 Run Locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open: http://localhost:8501

---

## ☁️ Deploy — 3 Options

### Option 1: Streamlit Community Cloud (Free — easiest)

Hosts the **dashboard UI** for free. Best for viewing charts and signals from any device.

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → `app.py`
4. Click **Deploy**

Your app is live at: `https://yourname-appname.streamlit.app`

> Note: The scanner/email alerts only run when someone has the page open.  
> For 24/7 background alerts, use Option 2 or 3 below.

---

### Option 2: GitHub Actions (Free — 24/7 email alerts)

Runs `scanner.py` every 15 minutes for free using GitHub Actions.  
No server needed — GitHub handles it. Free tier = 2000 min/month (enough for 15-min scans).

**Setup:**

1. Push this repo to GitHub

2. Go to your repo → **Settings** → **Secrets and variables** → **Actions**

3. Add these **Secrets** (click "New repository secret"):

   | Secret | Value |
   |--------|-------|
   | `SENDER_EMAIL` | your-gmail@gmail.com |
   | `GMAIL_APP_PASSWORD` | Gmail App Password (see below) |
   | `RECIPIENT_EMAIL` | where to receive alerts |

4. (Optional) Add these **Variables** to customise:

   | Variable | Default | Description |
   |----------|---------|-------------|
   | `COINS` | `BTCUSDT,ETHUSDT` | Comma-separated Binance symbols |
   | `TIMEFRAME` | `15m` | Binance interval (5m, 15m, 1h, 4h) |
   | `MIN_SCORE` | `5` | Min conditions out of 7 to trigger |

5. The scanner runs automatically. You can also trigger manually:  
   **Actions** tab → **Crypto Alert Scanner** → **Run workflow**

**How to get a Gmail App Password:**
1. Go to [myaccount.google.com](https://myaccount.google.com)
2. Security → 2-Step Verification (must be ON)
3. Search "App Passwords" → Create one for "Mail"
4. Copy the 16-character password (use this, not your normal password)

---

### Option 3: Docker (Self-hosted — full control)

Runs both the dashboard UI and a background scanner on any server (VPS, Raspberry Pi, etc).

```bash
# 1. Copy the env template and fill in your email details
cp .env.example .env
nano .env

# 2. Start everything
docker compose up -d

# Dashboard at http://your-server:8501
# Scanner runs every 15 min in the background
```

**Deploy to Railway / Render (free tier):**
- Push to GitHub
- Connect repo on [railway.app](https://railway.app) or [render.com](https://render.com)
- Set environment variables from `.env.example`
- Railway/Render auto-detects the Dockerfile

---

## 🎯 7-Condition Strategy

| # | Condition | BUY | SELL |
|---|-----------|-----|------|
| 1 | DI Crossover | 🔵 +DI > 🟢 -DI | 🟢 -DI > 🔵 +DI |
| 2 | ADX Strength | ADX > 25 | ADX > 25 |
| 3 | VWAP Position | Price > VWAP | Price < VWAP |
| 4 | SuperTrend | Bullish (green) | Bearish (red) |
| 5 | Pivot Point | Price > PP | Price < PP |
| 6 | RSI Filter | 30 < RSI < 70 | 30 < RSI < 70 |
| 7 | Volume | Vol > 1.5× avg | Vol > 1.5× avg |

**Signal triggers at 5+ conditions.** ATR-based stop-loss & 3 profit targets.

| Level | Calculation | Risk:Reward |
|-------|-------------|-------------|
| Stop Loss | 1.5× ATR | — |
| Target 1 | 1.5× ATR | 1:1 |
| Target 2 | 3.0× ATR | 1:2 |
| Target 3 | 4.5× ATR | 1:3 |

---

## 📧 Scanner CLI

Run the scanner standalone (locally or on any server):

```bash
# One-shot scan
python scanner.py

# Loop every 15 min
python scanner.py --loop --interval 900

# Custom coins and timeframe
python scanner.py --loop --coins BTCUSDT,ETHUSDT,SOLUSDT --timeframe 1h

# Need 6/7 conditions instead of 5
python scanner.py --loop --min-score 6
```

Environment variables: `SENDER_EMAIL`, `GMAIL_APP_PASSWORD`, `RECIPIENT_EMAIL`

---

## 🔐 Angel One Login (MCX tabs)

| Field | Where to find it |
|---|---|
| **API Key** | Angel One → SmartAPI → Your App → API Key |
| **Client ID** | Your Angel One login ID (e.g. A123456) |
| **Password** | Your Angel One trading password |
| **TOTP Secret** | Angel One → SmartAPI → TOTP Secret (base32 string) |

---

## 📊 Indicators

| Indicator | Default | What it shows |
|---|---|---|
| **ADX** | Period 14 | Trend strength. ADX > 25 = strong |
| **+DI / -DI** | Period 14 | Blue (+DI) vs Green (-DI) crossover |
| **SuperTrend** | Period 10, Mult 3 | Green = bullish, Red = bearish |
| **VWAP** | Intraday | Institutional fair value line |
| **RSI** | Period 14 | Overbought (>70) / Oversold (<30) |
| **Pivot Points** | Classic | PP, R1–R3, S1–S3 |
| **ATR** | Period 14 | Volatility for stop/target sizing |

---

## 🪙 Supported Markets

**MCX (requires Angel One login):** Gold, Silver, Gold Mini, Silver Mini

**Crypto (free, no login):** BTC, ETH, BNB, SOL, PAXG (tokenised gold)
