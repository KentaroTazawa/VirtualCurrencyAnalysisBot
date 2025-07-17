import os
import json
import time
from datetime import datetime, timedelta
import requests
import pandas as pd
from flask import Flask
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

OKX_BASE_URL = "https://www.okx.com"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)
app = Flask(__name__)
notified_in_memory = {}  # メモリ内で通知済み銘柄を保持

def fetch_ohlcv(symbol):
    url = f"{OKX_BASE_URL}/api/v5/market/candles?instId={symbol}&bar=15m&limit=100"
    print(f"[FETCH] {symbol} - OHLCVデータ取得中...")
    try:
        res = requests.get(url)
        res.raise_for_status()
        data = res.json().get("data")
        if not data or len(data) < 30:
            print(f"[SKIP] {symbol} - OHLCVデータが不足（{len(data) if data else 0}件）")
            return None
    except Exception as e:
        print(f"[ERROR] {symbol} - OHLCV取得失敗: {e}")
        return None

    df = pd.DataFrame(data)
    df.columns = ["col_" + str(i) for i in range(len(df.columns))]
    df = df.rename(columns={
        "col_0": "timestamp",
        "col_1": "open",
        "col_2": "high",
        "col_3": "low",
        "col_4": "close",
        "col_5": "volume"
    })
    df = df.iloc[::-1].copy()
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df

def calculate_indicators(df):
    print("[CALC] テクニカル指標計算中...")
    df["ema12"] = df["close"].ewm(span=12).mean()
    df["ema26"] = df["close"].ewm(span=26).mean()
    df["macd"] = df["ema12"] - df["ema26"]
    df["signal"] = df["macd"].ewm(span=9).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    df["ma25"] = df["close"].rolling(window=25).mean()
    df["disparity"] = (df["close"] - df["ma25"]) / df["ma25"] * 100
    df["vol_avg5"] = df["volume"].rolling(window=5).mean()

    return df

def passes_filters(df):
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    rsi_cond = latest["rsi"] >= 65
    macd_cross = prev["macd"] > prev["signal"] and latest["macd"] < latest["signal"]
    disparity_cond = latest["disparity"] > 3
    volume_cond = latest["volume"] > latest["vol_avg5"] * 1.3

    print(f"[FILTER] RSI={latest['rsi']:.2f}, MACDクロス={macd_cross}, 乖離率={latest['disparity']:.2f}%, Volume急増={volume_cond}")
    return rsi_cond and macd_cross and disparity_cond and volume_cond

def analyze_with_groq(df):
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    prompt = f"""
以下はある仮想通貨ペアの直近15分足のテクニカル指標です。
この情報に基づいて、ショートエントリーすべきかを分析してください。

RSI: {latest['rsi']:.2f}
MACD: {latest['macd']:.6f}, Signal: {latest['signal']:.6f}
MACDクロス: {'デッドクロス' if prev['macd'] > prev['signal'] and latest['macd'] < latest['signal'] else 'なし'}
移動平均乖離率: {latest['disparity']:.2f}%
出来高急増: {'はい' if latest['volume'] > latest['vol_avg5'] * 1.3 else 'いいえ'}

以下の形式でJSONで回答してください：
{{
  "ショートすべきか": "はい" または "いいえ",
  "理由": "〜〜",
  "利確ライン（TP）": "-x.x%",
  "損切ライン（SL）": "+x.x%",
  "利益の出る確率": 数値（0〜100）
}}
"""
    print("[GROQ] 分析リクエスト送信中...")
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}]
        )
        result = json.loads(response.choices[0].message.content)
        print(f"[GROQ] 分析結果: {result}")
        return result
    except Exception as e:
        print(f"[ERROR] Groq解析失敗: {e}")
        return {}

def send_to_telegram(symbol, result):
    text = (
        f"\u2b06\ufe0f ショートシグナル検出: {symbol}\n"
        f"\n"
        f"- 利益確率: {result.get('利益の出る確率', '?')}%\n"
        f"- 理由: {result.get('理由', '不明')}\n"
        f"- TP: {result.get('利確ライン（TP）', '?')} / SL: {result.get('損切ライン（SL）', '?')}\n"
    )
    print(f"[TELEGRAM] 通知送信: {symbol}")
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, data=data)
    except Exception as e:
        print(f"[ERROR] Telegram送信失敗: {e}")

def run_analysis():
    print("[INFO] 処理開始")
    now = datetime.utcnow()

    try:
        url = f"{OKX_BASE_URL}/api/v5/public/instruments?instType=SWAP"
        print("[INFO] 対象銘柄一覧を取得中...")
        symbols = [item["instId"] for item in requests.get(url).json()["data"] if item["instId"].endswith("-USDT-SWAP")]
        print(f"[INFO] 銘柄数: {len(symbols)} 件")
    except Exception as e:
        print(f"[ERROR] 銘柄取得失敗: {e}")
        return

    for symbol in symbols:
        try:
            last_notified = notified_in_memory.get(symbol)
            if last_notified and now - last_notified < timedelta(hours=1):
                print(f"[SKIP] {symbol} - 直近1時間以内に通知済み")
                continue

            df = fetch_ohlcv(symbol)
            if df is None:
                continue

            df = calculate_indicators(df)

            if not passes_filters(df):
                print(f"[SKIP] {symbol} - フィルター条件不一致")
                continue

            result = analyze_with_groq(df)

            if result.get("ショートすべきか") == "はい" and result.get("利益の出る確率", 0) >= 70:
                send_to_telegram(symbol, result)
                notified_in_memory[symbol] = now
                print(f"[NOTIFY] {symbol} - 通知完了")
            else:
                print(f"[SKIP] {symbol} - Groq判定: 通知対象外")

        except Exception as e:
            print(f"[ERROR] {symbol} - 処理中に例外: {e}")

    print("[INFO] 処理完了")

@app.route("/")
def index():
    return "OK"

@app.route("/run_analysis")
def run_analysis_route():
    run_analysis()
    return "Analysis completed"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
