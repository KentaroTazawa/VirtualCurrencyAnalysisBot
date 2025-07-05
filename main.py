import requests
import os
import openai
import pandas as pd

openai.api_key = os.getenv("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def fetch_ohlcv(symbol="BTCUSDT", interval="15m", limit=50):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    res = requests.get(url, params=params).json()
    
    if not isinstance(res, list):
        raise ValueError(f"Binance APIエラー: {res}")
    
    closes = [float(c[4]) for c in res]
    return closes


def send_to_gpt(closes):
    text = ", ".join([f"{c:.2f}" for c in closes])
    prompt = f"""
以下は仮想通貨BTCUSDTの15分足終値データ（最新から過去へ50本）です：
{text}

このチャートを分析して、「今ショートを仕掛けるべきか？」を判断し、
利確ライン（TP）と損切ライン（SL）も数字で提案してください。

形式：
・ショートすべきか：はい / いいえ
・理由：
・利確目安（TP）：
・損切目安（SL）：
"""

    res = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "あなたは熟練のトレーダーAIです。"},
            {"role": "user", "content": prompt}
        ]
    )
    return res["choices"][0]["message"]["content"]

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg})

def main():
    closes = fetch_ohlcv()
    result = send_to_gpt(closes)
    send_telegram("📉 BTCUSDTショート分析結果\n\n" + result)

if __name__ == "__main__":
    main()
