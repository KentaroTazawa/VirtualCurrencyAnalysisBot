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
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)
app = Flask(__name__)
notified_in_memory = {}
symbol_to_id_cache = {}
symbol_to_id_cache_time = None

def send_error_to_telegram(error_message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": f"⚠️ エラー発生:\n```\n{error_message}\n```",
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, data=data)
    except:
        pass

def fetch_okx_tickers():
    url = f"{OKX_BASE_URL}/api/v5/market/tickers?instType=SWAP"
    res = requests.get(url, timeout=5)
    res.raise_for_status()
    return res.json().get("data", [])

def get_top10_rising_symbols():
    tickers = fetch_okx_tickers()
    sorted_tickers = sorted(
        [t for t in tickers if t["instId"].endswith("USDT-SWAP") and t.get("open24h") and t.get("last")],
        key=lambda x: ((float(x["last"]) - float(x["open24h"])) / float(x["open24h"])) * 100,
        reverse=True
    )
    return [t["instId"] for t in sorted_tickers[:10]]

def fetch_coingecko_symbol_map():
    global symbol_to_id_cache, symbol_to_id_cache_time
    now = datetime.utcnow()

    if symbol_to_id_cache and symbol_to_id_cache_time and (now - symbol_to_id_cache_time < timedelta(hours=1)):
        return symbol_to_id_cache

    url = f"{COINGECKO_BASE_URL}/coins/list"
    res = requests.get(url, timeout=5)
    res.raise_for_status()
    data = res.json()
    symbol_to_id_cache = {item["symbol"].lower(): item["id"] for item in data}
    symbol_to_id_cache_time = now
    return symbol_to_id_cache

def get_coingecko_id(symbol_base):
    symbol_map = fetch_coingecko_symbol_map()
    return symbol_map.get(symbol_base.lower())

def fetch_ohlcv_coingecko(coin_id, max_retries=5, backoff_factor=2):
    url = f"{COINGECKO_BASE_URL}/coins/{coin_id}/ohlc?vs_currency=usd&days=max"
    for attempt in range(max_retries):
        try:
            res = requests.get(url, timeout=10)
            if res.status_code in [401, 403, 404]:
                print(f"[SKIP] CoinGecko非対応 (code {res.status_code}): {coin_id}")
                return None
            if res.status_code == 429:
                wait = backoff_factor ** attempt
                print(f"[RETRY] 429 Too Many Requests: {coin_id}, {wait}s 待機中...")
                time.sleep(wait)
                continue
            res.raise_for_status()
            data = res.json()
            if not data:
                return None
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df
        except requests.exceptions.HTTPError as e:
            send_error_to_telegram(f"CoinGecko HTTPエラー（{coin_id}）:\n{str(e)}")
            return None
        except Exception as e:
            send_error_to_telegram(f"CoinGecko取得失敗（{coin_id}）:\n{str(e)}")
            return None
    send_error_to_telegram(f"CoinGecko 429連続失敗（{coin_id}）")
    return None

def analyze_with_groq(df, symbol_base):
    latest = df.iloc[-1]
    prompt = f"""
次の仮想通貨 {symbol_base} は、史上最高値を更新した直後に価格が下落し始めています。
以下の価格履歴に基づいて、このタイミングでショートエントリーするのが妥当かどうかを分析してください。

最新の価格情報：
- 日時: {latest['timestamp']}
- 高値: {latest['high']}, 安値: {latest['low']}
- 始値: {latest['open']}, 終値: {latest['close']}

以下の形式でJSONで答えてください：
{{
  "ショートすべきか": "はい" または "いいえ",
  "理由": "〜〜",
  "利確ライン（TP）": "+x.x%",
  "損切ライン（SL）": "-x.x%",
  "利益の出る確率": 0〜100の数値
}}
"""
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.choices[0].message.content
        json_match = re.search(r"\{.*?\}", content, re.DOTALL)
        if not json_match:
            raise ValueError("JSON形式の出力が見つかりません")
        return json.loads(json_match.group(0))
    except Exception as e:
        send_error_to_telegram(f"Groq API エラー:\n{str(e)}")
        return {
            "ショートすべきか": "はい",
            "理由": "Groq失敗",
            "利確ライン（TP）": "Groq失敗",
            "損切ライン（SL）": "Groq失敗",
            "利益の出る確率": 0
        }

def send_to_telegram(symbol, result):
    emoji = "📉"
    title = "ショート"
    symbol_base = symbol.replace("-USDT-SWAP", "")
    text = f"""{emoji} {title}シグナル検出: {symbol_base}
- 利益確率: {result.get('利益の出る確率', '?')}%
- 理由: {result.get('理由', '不明')}
- 損切: {result.get('損切ライン（SL）', '?')} / 利確: {result.get('利確ライン（TP）', '?')}
"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, data=data)
    except Exception as e:
        send_error_to_telegram(f"Telegram送信エラー:\n{str(e)}")

def run_analysis():
    now = datetime.utcnow()
    top_symbols = get_top10_rising_symbols()
    checked = 0

    for symbol in top_symbols:
        if checked >= 5:
            break

        try:
            last_notified = notified_in_memory.get(symbol)
            if last_notified and now - last_notified < timedelta(minutes=60):
                continue

            symbol_base = symbol.replace("-USDT-SWAP", "")
            coingecko_id = get_coingecko_id(symbol_base)
            if not coingecko_id:
                print(f"[SKIP] CoinGecko ID 不明: {symbol_base}")
                continue

            df = fetch_ohlcv_coingecko(coingecko_id)
            if df is None or len(df) < 10:
                continue

            ath = df["high"].max()
            latest = df.iloc[-1]
            if latest["high"] < ath * 0.995:
                print(f"[SKIP] ATH未更新: {symbol_base}, 現在: {latest['high']}, ATH: {ath}")
                continue

            print(f"[CHECK] {symbol_base}: ATH更新検出、高値={latest['high']}, ATH={ath}")
            result = analyze_with_groq(df, symbol_base)
            if result.get("ショートすべきか") == "はい":
                send_to_telegram(symbol, result)
                notified_in_memory[symbol] = now

        except Exception as e:
            error_detail = traceback.format_exc()
            send_error_to_telegram(f"{symbol} 処理中の例外:\n{error_detail}")

        finally:
            time.sleep(7.5)
            checked += 1

@app.route("/")
def index():
    return "OK"

@app.route("/run_analysis")
def run_analysis_route():
    run_analysis()
    return "Analysis completed", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
