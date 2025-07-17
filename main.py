import os
import json
import time
from datetime import datetime, timedelta
import requests
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask
from PIL import Image
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

OKX_BASE_URL = "https://www.okx.com"
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NOTIFIED_FILE = "notified_pairs.json"

client = Groq(api_key=GROQ_API_KEY)
app = Flask(__name__)

# --- JSONによる通知履歴の読み書き ---
def load_notified():
    if os.path.exists(NOTIFIED_FILE):
        with open(NOTIFIED_FILE, "r") as f:
            return json.load(f)
    return {}

def save_notified(data):
    with open(NOTIFIED_FILE, "w") as f:
        json.dump(data, f)

# --- データ取得 ---
def fetch_ohlcv(symbol):
    url = f"{OKX_BASE_URL}/api/v5/market/candles?instId={symbol}&bar=15m&limit=100"
    res = requests.get(url).json()

    if not res.get("data") or len(res["data"]) < 30:
        raise ValueError(f"{symbol} のOHLCVデータが不足または存在しません")

    df = pd.DataFrame(res["data"], columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.iloc[::-1].copy()
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df

# --- テクニカル指標計算 ---
def calculate_indicators(df):
    if df.empty or len(df) < 26:
        raise ValueError("十分なデータがありません")

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

    return df

# --- チャート画像作成 ---
def generate_chart(df, symbol):
    plt.figure(figsize=(10, 4))
    plt.plot(df["close"], label="Close Price", color="black")
    plt.title(f"{symbol} Close Price")
    plt.xlabel("Index")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    image_path = f"{symbol.replace('/', '_')}.png"
    plt.savefig(image_path)
    plt.close()
    return image_path

# --- Groq で分析 ---
def analyze_with_groq(df):
    df_str = df[["close", "macd", "signal", "rsi"]].tail(30).to_string(index=False)

    prompt = f"""
以下は仮想通貨の15分足データです。MACD・Signal・RSIの観点から、買いシグナルまたは売りシグナルが出ているかを判断してください。
出力は以下のJSON形式で返してください：
{{
  "signal": "buy" または "sell" または "neutral",
  "confidence": 数値（0～100）
}}

```
{df_str}
```
"""

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8192"
    )

    try:
        result = json.loads(chat_completion.choices[0].message.content)
        return result
    except Exception:
        return {"signal": "neutral", "confidence": 0}

# --- Discord通知 ---
def send_to_discord(symbol, signal, confidence, image_path):
    with open(image_path, "rb") as f:
        image_data = f.read()

    filename = os.path.basename(image_path)
    payload = {
        "content": f"**{symbol}** に {signal.upper()} シグナル\n信頼度: {confidence}%",
    }
    files = {"file": (filename, image_data)}

    requests.post(DISCORD_WEBHOOK_URL, data=payload, files=files)

# --- メイン処理 ---
def run_analysis():
    print("[INFO] 処理開始")
    notified = load_notified()
    updated_notified = {}

    url = f"{OKX_BASE_URL}/api/v5/public/instruments?instType=SWAP"
    symbols = [item["instId"] for item in requests.get(url).json()["data"] if item["instId"].endswith("-USDT-SWAP")]

    now = datetime.utcnow()

    for symbol in symbols:
        try:
            last_notified = notified.get(symbol)
            if last_notified:
                last_dt = datetime.strptime(last_notified, "%Y-%m-%d %H:%M:%S")
                if now - last_dt < timedelta(hours=1):
                    continue

            df = fetch_ohlcv(symbol)
            df = calculate_indicators(df)

            result = analyze_with_groq(df)

            if result["signal"] in ["buy", "sell"] and result["confidence"] >= 65:
                image_path = generate_chart(df, symbol)
                send_to_discord(symbol, result["signal"], result["confidence"], image_path)
                updated_notified[symbol] = now.strftime("%Y-%m-%d %H:%M:%S")
                print(f"[NOTIFY] {symbol} - {result}")

        except Exception as e:
            print(f"[ERROR] {symbol} → {e}")

    notified.update(updated_notified)
    save_notified(notified)
    print("[INFO] 処理完了")

# --- Flask ルート ---
@app.route("/")
def index():
    return "OK"

@app.route("/run_analysis")
def run_analysis_route():
    run_analysis()
    return "Analysis completed"

# --- ポート指定 ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
