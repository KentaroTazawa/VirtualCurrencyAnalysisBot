import os
import json
import requests
from datetime import datetime
from flask import Flask, request
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from groq import Groq

app = Flask(__name__)

# Telegram通知関数（ログ付き）
def send_telegram_message(message):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print("[INFO] Telegram送信成功:", response.text)
    except Exception as e:
        print("[ERROR] Telegram送信失敗:", e)
        print("[DEBUG] URL:", url)
        print("[DEBUG] Payload:", payload)

# 仮想通貨データ取得（ダミー実装）
def fetch_crypto_data():
    # 実際はAPIから取得すべき
    return [
        {"symbol": "BNTUSDT", "score": 87},
        {"symbol": "BROCKUSDT", "score": 91},
        {"symbol": "BTCUSDT", "score": 42},
    ]

# Groqによるフィルタ（例：スコアが80以上なら通知）
def analyze_with_groq(data):
    results = []
    for item in data:
        if item["score"] >= 80:
            results.append(item["symbol"])
    return results

@app.route("/")
def index():
    return "OK"

@app.route("/run_analysis")
def run_analysis():
    print("[INFO] 処理開始")

    data = fetch_crypto_data()
    symbols_to_notify = analyze_with_groq(data)

    if symbols_to_notify:
        for symbol in symbols_to_notify:
            print("[INFO] 通知対象:", symbol)
            send_telegram_message(f"ショート候補: {symbol}")
    else:
        print("[INFO] 通知対象がなかったためTelegram通知なし")

    return "OK"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
