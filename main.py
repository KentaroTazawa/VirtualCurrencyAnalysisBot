import os
import json
import time
import requests
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, jsonify
from datetime import datetime
from io import BytesIO
from pytz import timezone

# 環境変数の読み込み
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

app = Flask(__name__)
OKX_BASE_URL = "https://www.okx.com"
JST = timezone("Asia/Tokyo")
NOTIFIED_FILE = "notified_pairs.json"
NOTIFY_INTERVAL_SEC = 3600  # 1時間

# ----------------------------------------
# データ取得・指標計算
# ----------------------------------------

def fetch_symbols():
    url = f"{OKX_BASE_URL}/api/v5/public/instruments?instType=SWAP"
    res = requests.get(url).json()
    return [x["instId"] for x in res["data"] if x["instId"].endswith("USDT-SWAP")]

def fetch_ohlcv(symbol):
    url = f"{OKX_BASE_URL}/api/v5/market/candles?instId={symbol}&bar=15m&limit=100"
    res = requests.get(url).json()
    df = pd.DataFrame(res["data"], columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.iloc[::-1].copy()
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def is_dead_cross(macd, signal):
    return macd.iloc[-2] > signal.iloc[-2] and macd.iloc[-1] < signal.iloc[-1]

def calculate_indicators(df):
    df["rsi"] = compute_rsi(df["close"])
    df["macd"], df["signal"] = compute_macd(df["close"])
    df["ma25"] = df["close"].rolling(window=25).mean()
    df["disparity"] = (df["close"] - df["ma25"]) / df["ma25"] * 100
    df["vol_ma"] = df["volume"].rolling(window=5).mean()
    df["vol_spike"] = df["volume"] > df["vol_ma"] * 1.5
    return df

# ----------------------------------------
# 通知履歴管理
# ----------------------------------------

def load_notified():
    if os.path.exists(NOTIFIED_FILE):
        with open(NOTIFIED_FILE, "r") as f:
            return json.load(f)
    return {}

def save_notified(data):
    with open(NOTIFIED_FILE, "w") as f:
        json.dump(data, f)

def was_notified_recently(symbol, notified_dict):
    now = time.time()
    last = notified_dict.get(symbol)
    return last and (now - last) < NOTIFY_INTERVAL_SEC

# ----------------------------------------
# 通知・Groq・描画
# ----------------------------------------

def ask_groq(symbol, df):
    prompt = f"""
以下は{symbol}の15分足チャート分析結果です。
- RSI: {df['rsi'].iloc[-1]:.2f}
- MACD: {df['macd'].iloc[-1]:.5f}
- シグナル: {df['signal'].iloc[-1]:.5f}
- 乖離率: {df['disparity'].iloc[-1]:.2f}%
- 出来高急増: {"あり" if df["vol_spike"].iloc[-1] else "なし"}

この情報を元に、次の形式でショートすべきか判断してください：

・ショートすべきか（はい/いいえ）：
・理由：
・利確ライン（TP）：
・損切ライン（SL）：
・利益の出る確率（%）：
"""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
    return res.json()["choices"][0]["message"]["content"]

def send_telegram(text, df=None, symbol=None):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    requests.post(url, json=payload)

    if df is not None and symbol:
        plt.figure(figsize=(10, 4))
        plt.plot(df["close"], label="Close")
        plt.plot(df["ma25"], label="MA25", linestyle="--")
        plt.title(f"{symbol} - 15min")
        plt.legend()
        plt.grid()
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        photo_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        requests.post(photo_url, data={"chat_id": TELEGRAM_CHAT_ID}, files={"photo": buf})

# ----------------------------------------
# Flaskルート
# ----------------------------------------

@app.route("/")
def health():
    return "OK", 200

@app.route("/run_analysis")
def run_analysis():
    print("[INFO] 処理開始")
    symbols = fetch_symbols()
    notified_dict = load_notified()
    now = time.time()

    for sym in symbols:
        try:
            if was_notified_recently(sym, notified_dict):
                continue
            df = fetch_ohlcv(sym)
            df = calculate_indicators(df)
            if (
                df["rsi"].iloc[-1] >= 65 and
                is_dead_cross(df["macd"], df["signal"]) and
                df["disparity"].iloc[-1] > 3 and
                df["vol_spike"].iloc[-1]
            ):
                result = ask_groq(sym, df)
                if "はい" in result and "利益の出る確率" in result:
                    prob = int(result.split("利益の出る確率")[1].split("%")[0].strip("：: "))
                    if prob >= 65:
                        send_telegram(f"【{sym} 分析結果】\n{result}", df, sym)
                        notified_dict[sym] = now
        except Exception as e:
            print(f"[ERROR] {sym} → {e}")

    save_notified(notified_dict)
    return jsonify({"status": "done"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
