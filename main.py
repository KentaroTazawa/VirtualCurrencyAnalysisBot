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
    return {"X-Cg-Pro-Api-Key": COINGECKO_API_KEY} if COINGECKO_API_KEY else {}

def send_error_to_telegram(error_message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": f"""\u26a0\ufe0f エラー発生:

{error_message}

"parse_mode": "Markdown"
    }
    try:
        requests.post(url, data=data)
    except:
        pass

def get_top10_symbols_by_24h_change():
    try:
        print("🔍 OKXからSWAP銘柄を取得中...")
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
        top_symbols = [t["instId"] for t in sorted_tickers[:10]]
        print(f"✅ 急上昇TOP10: {top_symbols}")
        return top_symbols, filtered
    except Exception as e:
        send_error_to_telegram(f"急上昇銘柄取得エラー:\n{str(e)}")
        return [], []

def get_coingecko_markets():
    try:
        print("🌐 CoinGecko markets取得中...")
        url = f"{COINGECKO_BASE_URL}/coins/markets"
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 250,
            "page": 1,
            "sparkline": False
        }
        res = requests.get(url, params=params, headers=coingecko_headers())
        return res.json()
    except Exception as e:
        send_error_to_telegram(f"CoinGeckoマーケット取得エラー:\n{str(e)}")
        return []

def find_coin_id(symbol, markets):
    symbol_clean = symbol.replace("-USDT-SWAP", "").lower()
    for coin in markets:
        if coin.get("symbol", "").lower() == symbol_clean:
            return coin.get("id"), coin.get("ath"), coin.get("current_price")
    return None, None, None

def is_ath_today(current_price, ath_price):
    try:
        if not current_price or not ath_price:
            return False
        return current_price >= ath_price
    except:
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
        send_error_to_telegram(f"{symbol} のローソク取得失敗:\n{str(e)}")
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
    text = f"""📉 ATH銘柄警告: {symbol.replace("-USDT-SWAP", "")}

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
    print("🚀 分析開始")
    symbols, okx_data = get_top10_symbols_by_24h_change()
    markets = get_coingecko_markets()

    for symbol in symbols:
        try:
            print(f"\n🔎 処理中: {symbol}")
            coin_id, ath_price, current_price = find_coin_id(symbol, markets)
            if not coin_id:
                print(f"❌ CoinGecko ID 未取得: {symbol}")
                continue

            print(f"🕒 ATH確認中: {coin_id}")
            if not is_ath_today(current_price, ath_price):
                print(f"📉 ATH未達: {symbol}")
                continue

            df = fetch_ohlcv(symbol)
            if df is None:
                print(f"📉 ローソク足取得失敗: {symbol}")
                continue

            result = analyze_with_groq(df, symbol)
            print(f"📬 通知送信中: {symbol}")
            send_to_telegram(symbol, result)

            time.sleep(10)

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
