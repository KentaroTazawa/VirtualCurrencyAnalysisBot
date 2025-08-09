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

# === API設定 ===
OKX_BASE_URL = "https://www.okx.com"
CMC_BASE_URL = "https://pro-api.coinmarketcap.com/v1"
CC_BASE_URL = "https://min-api.cryptocompare.com/data"

CMC_API_KEY = os.getenv("COINMARKETCAP_API_KEY")
CC_API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)
app = Flask(__name__)
notified_in_memory = {}

# キャッシュ設定
TOP_SYMBOLS_LIMIT = 5
CMC_COIN_LIST_CACHE = []
CMC_COIN_LIST_LAST_FETCH = None
CMC_COIN_LIST_TTL = timedelta(minutes=60)

def cmc_headers():
    return {"X-CMC_PRO_API_KEY": CMC_API_KEY}

def send_error_to_telegram(error_message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": f"⚠️ エラー発生:\n\n{error_message}"})
    except:
        pass

# === CoinMarketCapコインリスト取得 ===
def get_cmc_coin_list():
    global CMC_COIN_LIST_CACHE, CMC_COIN_LIST_LAST_FETCH
    if CMC_COIN_LIST_CACHE and CMC_COIN_LIST_LAST_FETCH and datetime.now() - CMC_COIN_LIST_LAST_FETCH < CMC_COIN_LIST_TTL:
        return CMC_COIN_LIST_CACHE
    try:
        url = f"{CMC_BASE_URL}/cryptocurrency/map"
        res = requests.get(url, headers=cmc_headers())
        if res.status_code != 200:
            send_error_to_telegram(f"CMCコインリスト取得失敗: HTTP {res.status_code}")
            return []
        data = res.json().get("data", [])
        CMC_COIN_LIST_CACHE = data
        CMC_COIN_LIST_LAST_FETCH = datetime.now()
        print(f"🌐 CMC 全コインリスト取得済み: {len(data)}件")
        return data
    except Exception as e:
        send_error_to_telegram(f"CMCコインリスト取得エラー:\n{str(e)}")
        return []

# === 銘柄シンボルからCoinMarketCapのID取得 ===
def find_coin_id(symbol):
    symbol_clean = symbol.replace("-USDT-SWAP", "").upper()
    coins = get_cmc_coin_list()
    for coin in coins:
        if coin.get("symbol") == symbol_clean:
            return coin.get("id")
    for coin in coins:
        if symbol_clean in coin.get("name", "").upper():
            return coin.get("id")
    return None

# === ATHと現在価格取得（CoinMarketCap→CryptoCompareフォールバック） ===
def get_market_data(coin_id, symbol):
    try:
        url = f"{CMC_BASE_URL}/cryptocurrency/quotes/latest?id={coin_id}"
        res = requests.get(url, headers=cmc_headers())
        if res.status_code == 200:
            data = res.json().get("data", {}).get(str(coin_id), {})
            price = data.get("quote", {}).get("USD", {}).get("price")
            ath_price = data.get("ath", {}).get("price", None)  # CMCはath直接ない場合あり
            return ath_price, price
        else:
            raise Exception(f"CMC失敗: {res.status_code}")
    except:
        try:
            # CryptoCompareフォールバック
            symbol_clean = symbol.replace("-USDT-SWAP", "").upper()
            url = f"{CC_BASE_URL}/pricemultifull?fsyms={symbol_clean}&tsyms=USD&api_key={CC_API_KEY}"
            res = requests.get(url)
            data = res.json()
            price = data.get("RAW", {}).get(symbol_clean, {}).get("USD", {}).get("PRICE")
            ath_price = data.get("RAW", {}).get(symbol_clean, {}).get("USD", {}).get("HIGH24HOUR")
            return ath_price, price
        except Exception as e:
            send_error_to_telegram(f"マーケットデータ取得失敗 ({symbol}): {str(e)}")
            return None, None

# === OKXのトップ銘柄取得 ===
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
        return [t["instId"] for t in sorted_tickers[:limit]], filtered
    except Exception as e:
        send_error_to_telegram(f"急上昇銘柄取得エラー:\n{str(e)}")
        return [], []

def is_ath_today(current_price, ath_price):
    try:
        return current_price and ath_price and current_price >= ath_price
    except:
        return False

def fetch_ohlcv(symbol):
    try:
        print(f"🕒 {symbol} のローソク足データ取得中...")
        url = f"{OKX_BASE_URL}/api/v5/market/candles?instId={symbol}&bar=15m&limit=100"
        res = requests.get(url)
        time.sleep(0.8)
        data = res.json().get("data", [])
        if not data:
            print(f"⚠️ {symbol} のローソク足データがありません")
            return None
        df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "vol", "_1", "_2"])
        df = df[["ts", "open", "high", "low", "close", "vol"]].iloc[::-1].copy()
        df[["open", "high", "low", "close", "vol"]] = df[["open", "high", "low", "close", "vol"]].astype(float)
        print(f"✅ {symbol} のローソク足データ取得完了")
        return df
    except Exception as e:
        send_error_to_telegram(f"{symbol} のローソク取得失敗:\n{str(e)}")
        return None

def analyze_with_groq(df, symbol):
    print(f"🔍 {symbol} をGroqで分析中...")
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
        result = json.loads(match.group(0)) if match else {"今後下落する可能性は高いか": "不明"}
        print(f"✅ {symbol} のGroq分析結果: {result}")
        return result
    except Exception as e:
        send_error_to_telegram(f"Groqエラー: {str(e)}")
        print(f"⚠️ {symbol} のGroq分析に失敗")
        return {"今後下落する可能性は高いか": "不明"}

def send_to_telegram(symbol, result):
    text = f"""📉 ATH銘柄警告: {symbol.replace("-USDT-SWAP", "")}

- 今後下落する可能性: {result.get('今後下落する可能性は高いか', '?')}
- 理由: {result.get('理由', '?')}
- 下落幅予測: {result.get('予測される下落幅', '?')}
- 下落タイミング: {result.get('予測される下落タイミング', '?')}
"""
    print(f"✉️ {symbol} の分析結果をTelegramに送信中...")
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text})
        print(f"✅ {symbol} の結果をTelegramに送信完了")
    except Exception as e:
        send_error_to_telegram(f"Telegram送信エラー:\n{str(e)}")
        print(f"⚠️ {symbol} のTelegram送信に失敗")

def run_analysis():
    print("🚀 分析開始")
    symbols, _ = get_top_symbols_by_24h_change()
    print(f"🔎 対象銘柄: {symbols}")
    for symbol in symbols:
        print(f"==============================")
        print(f"🔔 {symbol} の処理開始")
        try:
            coin_id = find_coin_id(symbol)
            if not coin_id:
                print(f"⚠️ {symbol} のCoinMarketCap IDが見つかりません")
                continue
            print(f"🎯 {symbol} のCoinMarketCap ID: {coin_id}")

            ath_price, current_price = get_market_data(coin_id, symbol)
            print(f"💹 {symbol} 現在価格: {current_price} / ATH価格: {ath_price}")
            if not is_ath_today(current_price, ath_price):
                print(f"ℹ️ {symbol} はATHではありません。スキップ")
                continue

            df = fetch_ohlcv(symbol)
            if df is None:
                print(f"⚠️ {symbol} のローソク足データが取得できませんでした。スキップ")
                continue

            result = analyze_with_groq(df, symbol)
            send_to_telegram(symbol, result)
            time.sleep(10)  # API制限回避
        except Exception as e:
            send_error_to_telegram(f"{symbol} 分析中にエラー:\n{traceback.format_exc()}")
            print(f"⚠️ {symbol} の処理中に例外発生")
    print("✅ 分析終了")

@app.route("/")
def index():
    return "OK"

@app.route("/run_analysis")
def run_analysis_route():
    run_analysis()
    return "分析完了", 200

if __name__ == "__main__":
    app.run()
