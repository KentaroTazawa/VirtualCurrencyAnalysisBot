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

load_dotenv()

OKX_BASE_URL = "https://www.okx.com"
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)
app = Flask(__name__)
notified_in_memory = {}

def coingecko_headers():
    return {
        "X-Cg-Pro-Api-Key": COINGECKO_API_KEY
    } if COINGECKO_API_KEY else {}

def send_error_to_telegram(error_message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": f"\u26a0\ufe0f \u30a8\u30e9\u30fc\u767a\u751f:\n```\n{error_message}\n```",
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, data=data)
    except:
        pass

def get_top10_symbols_by_24h_change():
    try:
        url = f"{OKX_BASE_URL}/api/v5/market/tickers?instType=SWAP"
        res = requests.get(url)
        tickers = res.json().get("data", [])
        filtered = [t for t in tickers if t["instId"].endswith("-USDT-SWAP") and t.get("last") and t.get("open24h")]

        def chg(t):
            try:
                if "change24h" in t:
                    return float(t["change24h"])
                return (float(t["last"]) - float(t["open24h"])) / float(t["open24h"]) * 100
            except:
                return -9999

        sorted_tickers = sorted(filtered, key=chg, reverse=True)
        return [t["instId"] for t in sorted_tickers[:10]]
    except Exception as e:
        send_error_to_telegram(f"\u6025\u4e0a\u6607\u9298\u67c4\u53d6\u5f97\u30a8\u30e9\u30fc:\n{str(e)}")
        return []

def get_coingecko_coin_list():
    try:
        res = requests.get(f"{COINGECKO_BASE_URL}/coins/list", headers=coingecko_headers())
        return res.json()
    except Exception as e:
        send_error_to_telegram(f"CoinGecko\u9298\u67c4\u4e00\u89a7\u53d6\u5f97\u30a8\u30e9\u30fc:\n{str(e)}")
        return []

def get_coin_id_from_symbol(symbol, coin_list):
    symbol_clean = symbol.replace("-USDT-SWAP", "").lower()
    if isinstance(coin_list, list):
        for coin in coin_list:
            if isinstance(coin, dict) and coin.get("symbol", "").lower() == symbol_clean:
                return coin.get("id")
    return None

def is_ath_today(coin_id):
    try:
        url = f"{COINGECKO_BASE_URL}/coins/{coin_id}/market_chart?vs_currency=usd&days=30"
        res = requests.get(url, headers=coingecko_headers())
        data = res.json()["prices"]
        prices = [price[1] for price in data]
        return prices[-1] == max(prices)
    except Exception as e:
        send_error_to_telegram(f"{coin_id} \u306eATH\u5224\u5b9a\u30a8\u30e9\u30fc:\n{str(e)}")
        return False

def fetch_ohlcv(symbol):
    url = f"{OKX_BASE_URL}/api/v5/market/candles?instId={symbol}&bar=15m&limit=100"
    try:
        res = requests.get(url)
        data = res.json()["data"]
        if not data:
            return None
        df = pd.DataFrame(data)
        df.columns = ["ts", "open", "high", "low", "close", "vol", "_1", "_2"]
        df = df[["ts", "open", "high", "low", "close", "vol"]]
        df = df.iloc[::-1].copy()
        df[["open", "high", "low", "close", "vol"]] = df[["open", "high", "low", "close", "vol"]].astype(float)
        return df
    except Exception as e:
        send_error_to_telegram(f"{symbol} \u306e\u30ed\u30fc\u30bd\u30af\u53d6\u5f97\u5931\u6557:\n{str(e)}")
        return None

def analyze_with_groq(df, symbol):
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    prompt = f"""
以下は {symbol} の15分足テクニカルデータです。価格が過去最高であることを踏まえ、今後短期的に下落する可能性を分析してください。

**構造化JSONでのみ返答してください（説明不要）**

{{
  "今後下落する可能性は高いか": "はい" または "いいえ",
  "理由": "～",
  "予測される下落幅": "-x.x%",
  "予測される下落タイミング": "例: 数時間以内、24時間以内など"
}}

参考データ:
- RSI近似: {latest['close'] / prev['close']:.4f}
- 直近価格: {latest['close']}
- 出来高: {latest['vol']}
"""

    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        content = response.choices[0].message.content
        json_match = re.search(r"\{[\s\S]*?\}", content)
        if json_match:
            return json.loads(json_match.group(0))
        else:
            return {
                "今後下落する可能性は高いか": "不明",
                "理由": "Groq出力が不完全",
                "予測される下落幅": "-?",
                "予測される下落タイミング": "不明"
            }
    except Exception as e:
        send_error_to_telegram(f"Groqエラー: {str(e)}")
        return {
            "今後下落する可能性は高いか": "不明",
            "理由": "Groq例外発生",
            "予測される下落幅": "-?",
            "予測される下落タイミング": "不明"
        }

def send_to_telegram(symbol, result):
    text = f"""\ud83d\udcc9 ATH銘柄警告: {symbol.replace("-USDT-SWAP", "")}

- 今後下落する可能性: {result.get('今後下落する可能性は高いか', '?')}
- 理由: {result.get('理由', '?')}
- 下落幅予測: {result.get('予測される下落幅', '?')}
- 下落タイミング: {result.get('予測される下落タイミング', '?')}
"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text
    }
    try:
        requests.post(url, data=data)
    except Exception as e:
        send_error_to_telegram(f"Telegram送信エラー:\n{str(e)}")

def run_analysis():
    symbols = get_top10_symbols_by_24h_change()
    coin_list = get_coingecko_coin_list()

    for symbol in symbols:
        try:
            coin_id = get_coin_id_from_symbol(symbol, coin_list)
            if not coin_id:
                continue
            if not is_ath_today(coin_id):
                continue

            df = fetch_ohlcv(symbol)
            if df is None:
                continue

            result = analyze_with_groq(df, symbol)
            send_to_telegram(symbol, result)
        except Exception as e:
            send_error_to_telegram(f"{symbol} 分析中にエラー:\n{traceback.format_exc()}")

@app.route("/")
def index():
    return "OK"

@app.route("/run_analysis")
def run_analysis_route():
    run_analysis()
    return "分析完了", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
