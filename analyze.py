import os
import json
import requests
from datetime import datetime
import numpy as np
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NOTIFIED_FILE = "notified_pairs.json"

def calculate_rsi(prices, period=14):
    deltas = np.diff(prices)
    seed = deltas[:period]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi = 100 - 100 / (1 + rs)
    return round(rsi, 2)

def calculate_macd(prices):
    ema12 = pd.Series(prices).ewm(span=12).mean()
    ema26 = pd.Series(prices).ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    recent_cross = macd.iloc[-2] > signal.iloc[-2] and macd.iloc[-1] < signal.iloc[-1]
    return "dead" if recent_cross else "none"

def calculate_ma_gap(prices):
    ma = np.mean(prices[-25:])
    current = prices[-1]
    return round(((current - ma) / ma) * 100, 2)

def is_volume_spike(volumes):
    avg = np.mean(volumes[:-5])
    return volumes[-1] > avg * 1.5

def log(msg):
    print(msg, flush=True)

def fetch_ohlcv(symbol):
    url = f"https://www.okx.com/api/v5/market/candles?instId={symbol}&bar=15m&limit=100"
    res = requests.get(url).json()
    if res.get("code") != "0":
        return [], []
    closes = [float(c[4]) for c in reversed(res["data"])]
    volumes = [float(c[5]) for c in reversed(res["data"])]
    return closes, volumes

def fetch_symbols():
    url = "https://www.okx.com/api/v5/market/tickers?instType=SWAP"
    res = requests.get(url).json()
    if res.get("code") != "0":
        return []
    return [item["instId"] for item in res["data"] if item["instId"].endswith("-USDT-SWAP")]

def generate_chart(prices, symbol):
    plt.figure(figsize=(6, 3))
    plt.plot(prices, color='red')
    plt.title(f"{symbol} 15m Chart")
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

def send_telegram(photo, caption):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    files = {"photo": ("chart.png", photo)}
    data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
    requests.post(url, files=files, data=data)

def analyze_with_groq(symbol, rsi, macd, gap, volume_spike):
    prompt = f"""
あなたは熟練の仮想通貨トレーダーAIです。
以下のテクニカル情報を元に、この銘柄をショートすべきか判断してください。

銘柄: {symbol}
・RSI: {rsi}
・MACDクロス: {"デッドクロス" if macd == "dead" else "なし"}
・移動平均乖離率: {gap}%
・出来高急増: {"あり" if volume_spike else "なし"}

フォーマット：
・ショートすべきか（はい/いいえ）：
・理由：
・利確ライン（TP）：
・損切ライン（SL）：
・利益の出る確率：
"""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        return res.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"⚠️ Groqエラー: {e}"

def load_notified():
    today = datetime.utcnow().strftime("%Y-%m-%d")
    if os.path.exists(NOTIFIED_FILE):
        with open(NOTIFIED_FILE) as f:
            data = json.load(f)
        return set(data.get(today, []))
    return set()

def save_notified(pairs):
    today = datetime.utcnow().strftime("%Y-%m-%d")
    if os.path.exists(NOTIFIED_FILE):
        with open(NOTIFIED_FILE) as f:
            data = json.load(f)
    else:
        data = {}
    data[today] = list(pairs)
    with open(NOTIFIED_FILE, "w") as f:
        json.dump(data, f)

def run_analysis():
    log("[INFO] 処理開始")
    notified = load_notified()
    symbols = fetch_symbols()
    new_notify = set()

    for symbol in symbols:
        if symbol in notified:
            continue
        prices, volumes = fetch_ohlcv(symbol)
        if len(prices) < 30:
            continue
        rsi = calculate_rsi(prices)
        macd = calculate_macd(prices)
        gap = calculate_ma_gap(prices)
        volume_spike = is_volume_spike(volumes)

        if rsi < 70 or macd != "dead" or gap < 5 or not volume_spike:
            continue

        result = analyze_with_groq(symbol, rsi, macd, gap, volume_spike)
        if "利益の出る確率：" in result:
            chart = generate_chart(prices, symbol)
            send_telegram(chart, f"📉 {symbol} ショート分析

{result}")
            new_notify.add(symbol)

    save_notified(notified | new_notify)
    if new_notify:
        log(f"[INFO] 通知済み: {len(new_notify)}件")
        return "[INFO] 通知しました。"
    else:
        log("[INFO] 通知対象がなかったためTelegram通知なし")
        return "[INFO] 通知対象なし。"
