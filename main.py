import os
import json
import requests
import pandas as pd
from flask import Flask, jsonify
from datetime import datetime
from zoneinfo import ZoneInfo

app = Flask(__name__)
notified_symbols = set()  # メモリ上で通知済みの通貨ペアを管理

def fetch_data(symbol):
    try:
        url = f"https://api.example.com/market/{symbol}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[ERROR] {symbol} → {e}")
        return None

def analyze_market():
    symbols = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "ZRX-USDT-SWAP", "ZIL-USDT-SWAP"]
    new_alerts = []

    for symbol in symbols:
        if symbol in notified_symbols:
            continue

        data = fetch_data(symbol)
        if not data:
            continue

        if "volume" in data and data["volume"] > 1000000:  # 仮の条件
            print(f"[ALERT] {symbol} has high volume: {data['volume']}")
            notified_symbols.add(symbol)
            new_alerts.append(symbol)

    return new_alerts

@app.route("/run_analysis", methods=["GET", "HEAD"])
def run():
    print("[INFO] 処理開始")
    new_alerts = analyze_market()
    print("[INFO] 処理完了")
    return jsonify({"new_alerts": new_alerts})

@app.route("/", methods=["GET"])
def home():
    return "OK"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
