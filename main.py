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

# 通知済み記録をメモリ上で保持（再起動でリセットされる）
notified_cache = {}

def fetch_ohlcv(symbol):
    url = f"{OKX_BASE_URL}/api/v5/market/candles?instId={symbol}&bar=15m&limit=100"
    res = requests.get(url).json()
    data = res.get("data")

    if not data or len(data) < 30:
        raise ValueError(f"{symbol} のOHLCVデータが不足または存在しません")

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
    df["disparity"] = ((df["close"] - df["ma25"]) / df["ma25"]) * 100

    df["volume_ma5"] = df["volume"].rolling(window=5).mean()
    df["volume_surge"] = df["volume"] > (df["volume_ma5"] * 1.3)

    return df

def passes_filter(df):
    rsi = df["rsi"].iloc[-1]
    macd_now = df["macd"].iloc[-1]
    macd_prev = df["macd"].iloc[-2]
    signal_now = df["signal"].iloc[-1]
    signal_prev = df["signal"].iloc[-2]
    disparity = df["disparity"].iloc[-1]
    volume_surge = df["volume_surge"].iloc[-1]

    macd_cross = macd_prev > signal_prev and macd_now < signal_now

    return (
        rsi >= 65 and
        macd_cross and
        disparity >= 3.5 and
        volume_surge
    )

def analyze_with_groq(df):
    rsi = df["rsi"].iloc[-1]
    macd_now = df["macd"].iloc[-1]
    macd_prev = df["macd"].iloc[-2]
    signal_now = df["signal"].iloc[-1]
    signal_prev = df["signal"].iloc[-2]
    disparity = df["disparity"].iloc[-1]
    volume_now = df["volume"].iloc[-1]
    volume_avg = df["volume_ma5"].iloc[-1]

    prompt = f"""
以下は仮想通貨の15分足データから抽出したテクニカル指標の要約です。
これらの情報に基づき、ショートポジションを取るべきかどうかを判断してください。

- RSI: {rsi:.2f}
- MACD（前回 → 今回）: {macd_prev:.4f} → {macd_now:.4f}
- Signal（前回 → 今回）: {signal_prev:.4f} → {signal_now:.4f}
- 移動平均乖離率（Disparity）: {disparity:.2f}%
- 出来高: {volume_now:.2f}（直近5本平均: {volume_avg:.2f}）

出力は以下のJSON形式でお願いします：
{{
  "ショートすべきか": "はい" または "いいえ",
  "理由": "...",
  "TP": "-4.0%" のように記述,
  "SL": "+2.0%" のように記述,
  "利益の出る確率": 数値（0～100）
}}
"""
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-70b-8192"
    )

    try:
        result = json.loads(chat_completion.choices[0].message.content)
        return result
    except Exception:
        return {"ショートすべきか": "いいえ", "利益の出る確率": 0}

def send_to_telegram(symbol, result):
    text = (
        f"**{symbol}** にショートシグナル\n"
        f"理由: {result['理由']}\n"
        f"利確(TP): {result['TP']} / 損切(SL): {result['SL']}\n"
        f"勝率: {result['利益の出る確率']}%"
    )
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    requests.post(url, data=data)

def run_analysis():
    print("[INFO] 処理開始")
    url = f"{OKX_BASE_URL}/api/v5/public/instruments?instType=SWAP"
    symbols = [item["instId"] for item in requests.get(url).json()["data"] if item["instId"].endswith("-USDT-SWAP")]

    now = datetime.utcnow()

    for symbol in symbols:
        try:
            last_notified = notified_cache.get(symbol)
            if last_notified and now - last_notified < timedelta(hours=1):
                continue

            df = fetch_ohlcv(symbol)
            df = calculate_indicators(df)

            if not passes_filter(df):
                continue

            result = analyze_with_groq(df)

            if result["ショートすべきか"] == "はい" and result.get("利益の出る確率", 0) >= 70:
                send_to_telegram(symbol, result)
                notified_cache[symbol] = now
                print(f"[NOTIFY] {symbol} - {result}")

        except Exception as e:
            print(f"[ERROR] {symbol} → {e}")

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
