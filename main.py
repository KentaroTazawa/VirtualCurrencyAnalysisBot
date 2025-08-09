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

MEXC_BASE_URL = "https://contract.mexc.com"

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
        url = f"{MEXC_BASE_URL}/api/v1/contract/market/tickers"
        res = requests.get(url)
        res.raise_for_status()
        data = res.json()
        tickers = data.get("data", [])
        # 24h変化率計算： (lastPrice - openPrice) / openPrice * 100
        filtered = []
        for t in tickers:
            try:
                symbol = t.get("symbol", "")
                last_price = float(t.get("lastPrice", 0))
                open_price = float(t.get("openPrice", 0))
                if open_price == 0:
                    continue
                change_pct = (last_price - open_price) / open_price * 100
                filtered.append({"symbol": symbol, "last_price": last_price, "change_pct": change_pct})
            except:
                continue
        sorted_tickers = sorted(filtered, key=lambda x: x["change_pct"], reverse=True)
        return sorted_tickers[:limit]
    except Exception as e:
        send_error_to_telegram(f"MEXC 急上昇銘柄取得エラー:\n{str(e)}")
        return []

def fetch_ohlcv(symbol, limit=2000):
    try:
        # MEXC先物の15分足ローソク足取得（limitは最大2000）
        url = f"{MEXC_BASE_URL}/api/v1/contract/candles?symbol={symbol}&interval=15m&limit={limit}"
        res = requests.get(url)
        res.raise_for_status()
        data = res.json()
        candles = data.get("data", [])
        if not candles:
            return None
        # candlesは [timestamp, open, high, low, close, volume] のリストのリスト
        df = pd.DataFrame(candles, columns=["ts", "open", "high", "low", "close", "vol"])
        df[["open", "high", "low", "close", "vol"]] = df[["open", "high", "low", "close", "vol"]].astype(float)
        df = df.iloc[::-1].copy()  # 昇順に並び替え
        return df
    except Exception as e:
        send_error_to_telegram(f"{symbol} のローソク取得失敗:\n{str(e)}")
        return None

def is_ath_today(current_price, df):
    try:
        # 過去のローソク足の高値の最高値をATHとみなす
        ath_price = df["high"].max()
        return current_price >= ath_price, ath_price
    except Exception:
        return False, None

def analyze_with_groq(df, symbol):
    if len(df) < 2:
        return {"今後下落する可能性は高いか": "不明"}
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
    text = f"""📉 ATH銘柄警告: {symbol}

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
    top_tickers = get_top_symbols_by_24h_change()
    symbols = [t["symbol"] for t in top_tickers]
    print(f"🔎 対象銘柄: {symbols}")
    for ticker in top_tickers:
        symbol = ticker["symbol"]
        current_price = ticker["last_price"]
        try:
            print(f"==============================")
            print(f"🔔 {symbol} の処理開始")
            df = fetch_ohlcv(symbol)
            if df is None:
                print(f"⚠️ {symbol} のローソク足データ取得失敗。スキップ")
                continue
            ath_flag, ath_price = is_ath_today(current_price, df)
            print(f"💹 {symbol} 現在価格: {current_price} / ATH価格: {ath_price}")
            if not ath_flag:
                print(f"ℹ️ {symbol} はATHではありません。スキップ")
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
