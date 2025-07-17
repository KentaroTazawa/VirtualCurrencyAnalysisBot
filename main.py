import os
import pandas as pd
import requests
from flask import Flask
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

NOTIFY_API_KEY = os.getenv("TELEGRAM_API_KEY")
NOTIFY_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def fetch_data(symbol):
    url = f"https://api.coinglass.com/api/futures/coin_futures_chart?symbol={symbol}&interval=5m"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()["data"]
        if not data:
            raise ValueError(f"{symbol} のOHLCVデータが不足または存在しません")
        df = pd.DataFrame(data)
        df.columns = ["timestamp", "open", "high", "low", "close", "volume", "oi", "buy_vol", "sell_vol"]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except Exception as e:
        print(f"[ERROR] {symbol} → {e}")
        return None

def notify_telegram(message):
    url = f"https://api.telegram.org/bot{NOTIFY_API_KEY}/sendMessage"
    data = {"chat_id": NOTIFY_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=data)
    except Exception as e:
        print(f"[ERROR] 通知送信失敗: {e}")

@app.route("/")
def index():
    return "OK"

@app.route("/run_analysis")
def run_analysis():
    print("[INFO] 処理開始")
    symbols = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "ZRX-USDT-SWAP", "ZIL-USDT-SWAP"]
    for symbol in symbols:
        df = fetch_data(symbol)
        if df is not None:
            msg = f"{symbol} データ取得成功。最新終値: {df.iloc[-1]['close']}"
            notify_telegram(msg)
    print("[INFO] 処理完了")
    return "分析完了"
