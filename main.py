import os
import json
import time
import traceback
from datetime import datetime, timedelta
import requests
import pandas as pd
from flask import Flask
from groq import Groq
from dotenv import load_dotenv
import re
import random

load_dotenv()

OKX_BASE_URL = "https://www.okx.com"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY", None)

client = Groq(api_key=GROQ_API_KEY)
app = Flask(__name__)
notified_in_memory = {}

# CoinGecko 全コインリスト取得と symbol→id マッピング
symbol2cg_id = {}
def load_coingecko_mapping():
    try:
        headers = {}
        if COINGECKO_API_KEY:
            headers["x-cg-pro-api-key"] = COINGECKO_API_KEY

        resp = requests.get("https://api.coingecko.com/api/v3/coins/list", headers=headers, timeout=5)
        resp.raise_for_status()
        for c in resp.json():
            symbol2cg_id[c["symbol"].lower()] = c["id"]
    except Exception as e:
        print("CoinGecko mapping error:", e)

load_coingecko_mapping()

def send_error_to_telegram(error_message):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": f"⚠️ エラー発生:\n```\n{error_message}\n```",
                "parse_mode": "Markdown"
            }
        )
    except:
        pass

def fetch_ohlcv_coingecko(coin_id, vs_currency="usd", days="max"):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
    params = {"vs_currency": vs_currency, "days": days}
    headers = {}
    if COINGECKO_API_KEY:
        headers["x-cg-pro-api-key"] = COINGECKO_API_KEY
    res = requests.get(url, params=params, headers=headers, timeout=5)
    res.raise_for_status()
    df = pd.DataFrame(res.json(), columns=["timestamp", "open", "high", "low", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["volume"] = None
    return df

def fetch_ohlcv_okx(symbol, limit=100):
    url = f"{OKX_BASE_URL}/api/v5/market/candles?instId={symbol}&bar=15m&limit={limit}"
    res = requests.get(url, timeout=1)
    res.raise_for_status()
    data = res.json().get("data")
    if not data or len(data) < 30:
        return None
    df = pd.DataFrame(data, columns=range(len(data[0])))
    df = df.rename(columns={0: "timestamp", 1: "open", 2: "high", 3: "low", 4: "close", 5: "volume"})
    df = df.iloc[::-1].copy()
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df

def calculate_indicators(df):
    df["ema12"] = df["close"].ewm(span=12).mean()
    df["ema26"] = df["close"].ewm(span=26).mean()
    df["macd"] = df["ema12"] - df["ema26"]
    df["signal"] = df["macd"].ewm(span=9).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(window=14).mean() / loss.rolling(window=14).mean()
    df["rsi"] = 100 - (100 / (1 + rs))
    df["ma25"] = df["close"].rolling(window=25).mean()
    df["disparity"] = (df["close"] - df["ma25"]) / df["ma25"] * 100
    df["vol_avg5"] = df["volume"].rolling(window=5).mean()
    return df

def passes_filters(df, direction):
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    rsi_cond = latest["rsi"] >= 55
    macd_cross = prev["macd"] > prev["signal"] and latest["macd"] < latest["signal"]
    disparity_cond = latest["disparity"] > 1.0
    volume_cond = (latest["volume"] is not None) and latest["volume"] > latest["vol_avg5"] * 1.1
    ath = df["high"].max()
    is_ath_broken = latest["high"] >= ath
    drop_rate = (ath - latest["close"]) / ath * 100
    drop_cond = drop_rate >= 3.0
    reversing = latest["close"] < latest["open"] and pre
