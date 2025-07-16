import os
import json
import requests
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
from flask import Flask, send_file
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

def fetch_ohlcv(symbol: str):
    url = f"https://api.mexc.com/api/v3/klines?symbol={symbol}&interval=1m&limit=100"
    res = requests.get(url).json()
    prices = [float(candle[4]) for candle in res]  # Close price
    volumes = [float(candle[5]) for candle in res]  # Volume
    return prices, volumes

def calculate_indicators(prices, volumes):
    df = pd.DataFrame({"close": prices, "volume": volumes})
    df["rsi"] = df["close"].pct_change().apply(lambda x: max(x, 0)).rolling(window=14).mean() / \
                df["close"].pct_change().abs().rolling(window=14).mean() * 100
    df["ma"] = df["close"].rolling(window=20).mean()
    df["macd"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["disparity"] = (df["close"] - df["ma"]) / df["ma"] * 100
    df["volume_change"] = df["volume"].pct_change()
    return df

def generate_plot_image(prices, symbol):
    plt.figure(figsize=(10, 4))
    plt.plot(prices, label=symbol)
    plt.title(symbol)
    plt.legend()
    plt.grid(True)
    buffer = BytesIO()
    plt.savefig(buffer, format="PNG")
    buffer.seek(0)
    return buffer

def analyze_with_groq(df, symbol):
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)

    latest = df.iloc[-1]
    prompt = f"""
以下は仮想通貨の1分足データから計算したテクニカル指標の最新値です。
- 銘柄: {symbol}
- RSI: {latest['rsi']:.2f}
- MACD: {latest['macd']:.6f}
- 20MAとの乖離率: {latest['disparity']:.2f}%
- 出来高変化率: {latest['volume_change']:.2f}

このデータに基づいて、今この銘柄をショートすべきかどうかを「はい」か「いいえ」で答えてください。その理由も簡潔に説明してください。
"""

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8192"
    )
    return chat_completion.choices[0].message.content.strip()

def send_telegram_message(message, image_buffer=None):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    requests.post(url, data=data)

    if image_buffer:
        files = {"photo": image_buffer}
        data = {"chat_id": TELEGRAM_CHAT_ID}
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        requests.post(url, data=data, files=files)

@app.route("/")
def health_check():
    return "OK"

@app.route("/run_analysis")
def run_analysis():
    print("[INFO] 処理開始")
    try:
        symbols = requests.get("https://api.mexc.com/api/v3/ticker/price").json()
        top_symbols = [s["symbol"] for s in symbols if s["symbol"].endswith("USDT") and not s["symbol"].endswith("3SUSDT")][:10]

        notified = False
        for symbol in top_symbols:
            prices, volumes = fetch_ohlcv(symbol)
            df = calculate_indicators(prices, volumes)
            analysis = analyze_with_groq(df, symbol)

            if analysis.startswith("はい"):
                print(f"[INFO] 通知対象: {symbol}")
                image_buffer = generate_plot_image(prices, symbol)
                send_telegram_message(f"{symbol} をショートすべき理由:\n{analysis}", image_buffer)
                notified = True

        if not notified:
            print("[INFO] 通知対象がなかったためTelegram通知なし")

    except Exception as e:
        print("[ERROR]", str(e))
        send_telegram_message(f"Bot実行中にエラーが発生しました: {e}")

    return "OK"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
