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
CC_BASE_URL = "https://min-api.cryptocompare.com/data"

CC_API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)
app = Flask(__name__)
notified_in_memory = {}

TOP_SYMBOLS_LIMIT = 10  # 24h変化率トップ10対象

def send_error_to_telegram(error_message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": f"⚠️ エラー発生:\n\n{error_message}"})
    except:
        pass

def get_top_symbols_by_24h_change(limit=TOP_SYMBOLS_LIMIT):
    try:
        url = f"{OKX_BASE_URL}/api/v5/market/tickers?instType=SWAP"
        res = requests.get(url)
        tickers = res.json().get("data", [])
        filtered = [t for t in tickers if t["instId"].endswith("-USDT-SWAP") and t.get("last") and t.get("open24h")]
        def chg(t):
            try:
                return (float(t["last"]) - float(t["open24h"])) / float(t["open24h"]) * 100
            except:
                return -9999
        sorted_tickers = sorted(filtered, key=chg, reverse=True)
        return [t["instId"] for t in sorted_tickers[:limit]]
    except Exception as e:
        send_error_to_telegram(f"急上昇銘柄取得エラー:\n{str(e)}")
        return []

def get_all_time_high(symbol_clean):
    try:
        url = f"{CC_BASE_URL}/v2/histohour?fsym={symbol_clean}&tsym=USD&limit=2000&api_key={CC_API_KEY}"
        res = requests.get(url)
        data = res.json()
        prices = [candle["high"] for candle in data.get("Data", {}).get("Data", []) if candle.get("high")]
        if not prices:
            return None
        return max(prices)
    except Exception as e:
        send_error_to_telegram(f"{symbol_clean} ATH計算失敗: {str(e)}")
        return None

def get_current_price(symbol_clean):
    try:
        url = f"{CC_BASE_URL}/pricemultifull?fsyms={symbol_clean}&tsyms=USD&api_key={CC_API_KEY}"
        res = requests.get(url)
        data = res.json()
        price = data.get("RAW", {}).get(symbol_clean, {}).get("USD", {}).get("PRICE")
        return price
    except Exception as e:
        send_error_to_telegram(f"{symbol_clean} 現在価格取得失敗: {str(e)}")
        return None

def fetch_ohlcv(symbol):
    try:
        url = f"{OKX_BASE_URL}/api/v5/market/candles?instId={symbol}&bar=15m&limit=100"
        res = requests.get(url)
        time.sleep(0.8)
        data = res.json().get("data", [])
        if not data:
            return None
        df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "vol", "_1", "_2"])
        df = df[["ts", "open", "high", "low", "close", "vol"]].iloc[::-1].copy()
        df[["open", "high", "low", "close", "vol"]] = df[["open", "high", "low", "close", "vol"]].astype(float)
        return df
    except Exception as e:
        send_error_to_telegram(f"{symbol} のローソク取得失敗:\n{str(e)}")
        return None

def analyze_with_groq(df, symbol):
    latest, prev = df.iloc[-1], df.iloc[-2]
    prompt = f"""
以下は {symbol} の15分足テクニカルデータです。価格が過去最高であることを踏まえ、今後短期的に下落する可能性を分析してください。

**構造化JSONでのみ返答してください**

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
        res = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        content = res.choices[0].message.content
        match = re.search(r"\{[\s\S]*?\}", content)
        return json.loads(match.group(0)) if match else {"今後下落する可能性は高いか": "不明"}
    except Exception as e:
        send_error_to_telegram(f"Groqエラー: {str(e)}")
        return {"今後下落する可能性は高いか": "不明"}

def send_to_telegram(symbol, result):
    text = f"""📉 ATH銘柄警告: {symbol.replace("-USDT-SWAP", "")}

- 今後下落する可能性: {result.get('今後下落する可能性は高いか', '?')}
- 理由: {result.get('理由', '?')}
- 下落幅予測: {result.get('予測される下落幅', '?')}
- 下落タイミング: {result.get('予測される下落タイミング', '?')}
"""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text})
    except Exception as e:
        send_error_to_telegram(f"Telegram送信エラー:\n{str(e)}")

def run_analysis():
    print("🚀 分析開始")
    symbols = get_top_symbols_by_24h_change()
    print(f"🔎 対象銘柄: {symbols}")
    for symbol in symbols:
        try:
            print(f"==============================")
            print(f"🔔 {symbol} の処理開始")
            symbol_clean = symbol.replace("-USDT-SWAP", "").upper()
            ath_price = get_all_time_high(symbol_clean)
            current_price = get_current_price(symbol_clean)
            print(f"💹 {symbol} 現在価格: {current_price} / ATH価格: {ath_price}")
            if current_price is None or ath_price is None or current_price < ath_price:
                print(f"ℹ️ {symbol} はATH未満またはデータ不足のためスキップ")
                continue
            df = fetch_ohlcv(symbol)
            if df is None:
                print(f"⚠️ {symbol} のローソク足データ取得失敗。スキップ")
                continue
            result = analyze_with_groq(df, symbol)
            send_to_telegram(symbol, result)
            print(f"✅ {symbol} の分析完了・通知送信済み")
            time.sleep(10)  # API制限回避
        except Exception as e:
            send_error_to_telegram(f"{symbol} 分析中にエラー:\n{traceback.format_exc()}")
    print("✅ 分析終了")

@app.route("/")
def index():
    return "OK"

@app.route("/run_analysis")
def run_analysis_route():
    run_analysis()
    return "分析完了", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
