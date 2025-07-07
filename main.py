import os
import json
import requests
from datetime import datetime, timedelta, timezone
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- 環境変数読み込み ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# JSTタイムゾーン
JST = timezone(timedelta(hours=9))

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
    res = requests.get(url)
    if res.status_code != 200:
        return [], []
    data = res.json()
    if data.get("code") != "0":
        return [], []
    closes = [float(c[4]) for c in reversed(data["data"])]
    volumes = [float(c[5]) for c in reversed(data["data"])]
    return closes, volumes

def fetch_symbols():
    url = "https://www.okx.com/api/v5/market/tickers?instType=SWAP"
    res = requests.get(url)
    if res.status_code != 200:
        return []
    data = res.json()
    if data.get("code") != "0":
        return []
    return [item["instId"] for item in data["data"] if item["instId"].endswith("-USDT-SWAP")]

def generate_chart(prices, symbol):
    plt.figure(figsize=(6, 3))
    plt.plot(prices, color='red')
    plt.title(f"{symbol} 15m Chart")
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf

def send_telegram(photo, caption):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    files = {"photo": ("chart.png", photo)}
    data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
    try:
        res = requests.post(url, files=files, data=data)
        if res.status_code != 200:
            log(f"[Telegram送信エラー] {res.status_code}: {res.text}")
    except Exception as e:
        log(f"[Telegram例外] {e}")

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

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    json_data = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 512,
    }
    try:
        res = requests.post(url, headers=headers, json=json_data, timeout=20)
        if res.status_code == 200:
            data = res.json()
            return data["choices"][0]["message"]["content"].strip()
        else:
            return f"⚠️ Groq APIエラー {res.status_code}: {res.text}"
    except Exception as e:
        return f"⚠️ Groq例外発生: {e}"

def load_notified():
    today = datetime.now(JST).strftime("%Y-%m-%d")
    if os.path.exists("notified_pairs.json"):
        with open("notified_pairs.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(data.get(today, []))
    return set()

def save_notified(pairs):
    today = datetime.now(JST).strftime("%Y-%m-%d")
    if os.path.exists("notified_pairs.json"):
        with open("notified_pairs.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}
    data[today] = list(pairs)
    with open("notified_pairs.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    now = datetime.now(JST)
    if not (now.hour >= 20 or (now.hour == 0 and now.minute <= 30)):
        log("[INFO] 実行時間外のためスキップ")
        return

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

        log(f"[INFO] Groq分析中: {symbol} (RSI={rsi})")
        result = analyze_with_groq(symbol, rsi, macd, gap, volume_spike)
        log(f"[Groq分析結果] {symbol}\n{result}")

        if "利益の出る確率：" in result:
            try:
                prob_str = result.split("利益の出る確率：")[-1].split("\n")[0]
                prob = int(prob_str.replace("%", "").strip())
                if prob >= 80:
                    chart = generate_chart(prices, symbol)
                    send_telegram(chart, f"📉 {symbol} ショート分析\n\n{result}")
                    new_notify.add(symbol)
            except Exception as e:
                log(f"[確率解析エラー] {e}")
                continue

    save_notified(notified | new_notify)
    if new_notify:
        log(f"[INFO] 通知済み: {len(new_notify)}件")
    else:
        log("[INFO] 通知対象がなかったためTelegram通知なし")

if __name__ == "__main__":
    main()
