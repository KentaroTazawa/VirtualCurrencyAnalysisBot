import os
import json
import time
import requests
from datetime import datetime, timedelta
from flask import Flask
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

# 環境変数
OKX_BASE_URL = "https://www.okx.com"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

app = Flask(__name__)

def get_symbols():
    url = f"{OKX_BASE_URL}/api/v5/public/instruments?instType=SWAP"
    response = requests.get(url)
    return [item["instId"] for item in response.json()["data"] if item["instId"].endswith("USDT-SWAP")]

def get_candlesticks(symbol):
    url = f"{OKX_BASE_URL}/api/v5/market/candles"
    params = {"instId": symbol, "bar": "15m", "limit": "30"}
    response = requests.get(url, params=params)
    df = pd.DataFrame(response.json()["data"], columns=[
        "timestamp", "open", "high", "low", "close", "volume", "_", "_", "_", "_", "_"
    ])
    df["close"] = df["close"].astype(float)
    df["open"] = df["open"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)

def calculate_indicators(df):
    df["rsi"] = compute_rsi(df["close"])
    df["macd"], df["macd_signal"] = compute_macd(df["close"])
    df["disparity"] = (df["close"] - df["close"].rolling(25).mean()) / df["close"].rolling(25).mean() * 100
    df["vol_ma"] = df["volume"].rolling(5).mean()
    df["volume_spike"] = df["volume"] > 1.5 * df["vol_ma"]
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(window=period).mean()
    ma_down = down.rolling(window=period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def is_macd_dead_cross(macd, signal):
    return macd.iloc[-2] > signal.iloc[-2] and macd.iloc[-1] < signal.iloc[-1]

def prompt_groq(symbol, df):
    prompt = f"""銘柄: {symbol}
現在のRSI: {df['rsi'].iloc[-1]:.1f}
MACDがデッドクロスしているか: {"はい" if is_macd_dead_cross(df["macd"], df["macd_signal"]) else "いいえ"}
移動平均乖離率: {df['disparity'].iloc[-1]:.2f}%
出来高が急増しているか: {"はい" if df['volume_spike'].iloc[-1] else "いいえ"}

この情報をもとに、以下の形式で判断してください：

・ショートすべきか（はい/いいえ）：
・理由：
・利確ライン（TP）：
・損切ライン（SL）：
・利益の出る確率（%）：
"""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "あなたは熟練の仮想通貨トレーダーです。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }
    res = requests.post(GROQ_API_URL, headers=headers, json=data, timeout=60)
    return res.json()["choices"][0]["message"]["content"]

def generate_chart(df, symbol):
    plt.figure(figsize=(8, 4))
    plt.plot(df["timestamp"], df["close"], label="Close")
    plt.title(symbol)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.grid()
    plt.legend()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return buf

def send_telegram_message(text, image_buf=None):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto" if image_buf else f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "caption": text} if image_buf else {"chat_id": TELEGRAM_CHAT_ID, "text": text}
    files = {"photo": image_buf} if image_buf else None
    requests.post(url, data=data, files=files)

def load_notified_pairs():
    today = datetime.utcnow().strftime("%Y-%m-%d")
    if os.path.exists("notified_pairs.json"):
        with open("notified_pairs.json", "r") as f:
            data = json.load(f)
        return data.get(today, [])
    return []

def save_notified_pair(pair):
    today = datetime.utcnow().strftime("%Y-%m-%d")
    data = {}
    if os.path.exists("notified_pairs.json"):
        with open("notified_pairs.json", "r") as f:
            data = json.load(f)
    if today not in data:
        data[today] = []
    if pair not in data[today]:
        data[today].append(pair)
    with open("notified_pairs.json", "w") as f:
        json.dump(data, f)

@app.route("/")
def index():
    return "OK"

@app.route("/run_analysis")
def run_analysis():
    print("[INFO] 処理開始")
    notified = load_notified_pairs()
    symbols = get_symbols()
    for symbol in symbols:
        try:
            df = get_candlesticks(symbol)
            df = calculate_indicators(df)

            rsi_ok = df["rsi"].iloc[-1] > 70
            macd_cross = is_macd_dead_cross(df["macd"], df["macd_signal"])
            disparity_ok = df["disparity"].iloc[-1] > 5
            volume_ok = df["volume_spike"].iloc[-1]

            if all([rsi_ok, macd_cross, disparity_ok, volume_ok]):
                ai_result = prompt_groq(symbol, df)
                print(f"[INFO] AI解析結果: {ai_result}")

                if "はい" in ai_result and "利益の出る確率" in ai_result:
                    try:
                        probability = int("".join(filter(str.isdigit, ai_result.split("利益の出る確率")[1][:3])))
                        if probability >= 80 and symbol not in notified:
                            chart = generate_chart(df, symbol)
                            send_telegram_message(f"{symbol}\n{ai_result}", chart)
                            save_notified_pair(symbol)
                    except:
                        print("[WARN] 確率の抽出に失敗")
        except Exception as e:
            print(f"[ERROR] {symbol}: {e}")
    return "Done"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
