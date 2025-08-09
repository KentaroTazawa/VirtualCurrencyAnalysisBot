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
TOP_SYMBOLS_LIMIT = 30
CMC_COIN_LIST_CACHE = []
CMC_COIN_LIST_LAST_FETCH = None
CMC_COIN_LIST_TTL = timedelta(minutes=5)

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

# === ATHと現在価格取得（CryptoCompareヒストリカルで計算） ===
def get_all_time_high(symbol_clean):
    """CryptoCompareから全期間ヒストリカルデータを取得し、ATHを計算"""
    try:
        # 最大2000時間分のヒストリカル時足データを取得（約83日分）
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

def get_market_data(coin_id, symbol):
    """現在価格と全期間ATHを取得"""
    try:
        symbol_clean = symbol.replace("-USDT-SWAP", "").upper()
        # 現在価格取得
        price_url = f"{CC_BASE_URL}/pricemultifull?fsyms={symbol_clean}&tsyms=USD&api_key={CC_API_KEY}"
        res = requests.get(price_url)
        data = res.json()
        price = data.get("RAW", {}).get(symbol_clean, {}).get("USD", {}).get("PRICE")

        # 全期間ATH計算
        ath_price = get_all_time_high(symbol_clean)

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
    symbols, _ = get_top_symbols_by_24h_change()
    print(f"🔎 対象銘柄: {symbols}")
    for symbol in symbols:
        try:
            print(f"==============================")
            print(f"🔔 {symbol} の処理開始")
            coin_id = find_coin_id(symbol)
            print(f"🎯 {symbol} のCoinMarketCap ID: {coin_id}")
            ath_price, current_price = get_market_data(coin_id, symbol)
            print(f"💹 {symbol} 現在価格: {current_price} / ATH価格: {ath_price}")
            if not is_ath_today(current_price, ath_price):
                print(f"ℹ️ {symbol} はATHではありません。スキップ")
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
    app.run()
