import os
import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
import io
from datetime import datetime
from flask import Flask
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NOTIFIED_FILE = "notified_pairs.json"

def fetch_okx_symbols():
    url = "https://www.okx.com/api/v5/public/instruments?instType=SWAP"
    res = requests.get(url).json()
    return [i["instId"] for i in res["data"] if i["instId"].endswith("USDT-SWAP")]

def fetch_ohlcv(symbol):
    url = f"https://www.okx.com/api/v5/market/candles?instId={symbol}&bar=15m&limit=30"
    res = requests.get(url).json()
    if res["code"] != '0':
        return None
    df = pd.DataFrame(res["data"], columns=[
        "timestamp", "open", "high", "low", "close", "volume", *_])
    df = df.iloc[::-1]
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_macd(series):
    exp1 = series.ewm(span=12).mean()
    exp2 = series.ewm(span=26).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9).mean()
    return macd, signal

def calc_indicators(df):
    df["ema25"] = df["close"].ewm(span=25).mean()
    df["divergence"] = (df["close"] - df["ema25"]) / df["ema25"] * 100
    df["rsi"] = compute_rsi(df["close"])
    macd, signal = compute_macd(df["close"])
    df["macd"], df["macd_signal"] = macd, signal
    df["volume_avg"] = df["volume"].rolling(5).mean()
    return df

def filter_symbols(symbols):
    result = []
    for sym in symbols:
        df = fetch_ohlcv(sym)
        if df is None or len(df) < 26:
            continue
        df = calc_indicators(df)
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        if (
            latest["rsi"] >= 70 and
            prev["macd"] > prev["macd_signal"] and latest["macd"] < latest["macd_signal"] and
            latest["divergence"] > 5 and
            latest["volume"] > latest["volume_avg"]
        ):
            result.append((sym, df))
    return result

def analyze_with_groq(symbol, df):
    l = df.iloc[-1]
    prompt = f"""以下は仮想通貨のテクニカルデータです。この銘柄をショートするべきか分析してください。

銘柄: {symbol}
RSI: {l["rsi"]:.2f}
MACD: {l["macd"]:.6f}
MACDシグナル: {l["macd_signal"]:.6f}
移動平均乖離率: {l["divergence"]:.2f}%
出来高: {l["volume"]:.2f}
過去5本の平均出来高: {l["volume_avg"]:.2f}

以下の形式で出力してください：
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
    data = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data, timeout=30)
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[Groq ERROR] {e}")
        return None

def plot_chart(df, symbol):
    fig, ax = plt.subplots(figsize=(8, 4))
    df["close"].plot(ax=ax, label="Close", color="black")
    df["ema25"].plot(ax=ax, label="EMA25", linestyle="--", color="blue")
    ax.set_title(f"{symbol} - 15min")
    ax.legend()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return buf

def send_telegram_message(text):
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                      data={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=10)
    except Exception as e:
        print(f"[Telegram ERROR] message: {e}")

def send_telegram_image(image, caption):
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
                      files={"photo": image},
                      data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption}, timeout=10)
    except Exception as e:
        print(f"[Telegram ERROR] image: {e}")

def load_notified():
    today = datetime.utcnow().strftime("%Y-%m-%d")
    if os.path.exists(NOTIFIED_FILE):
        with open(NOTIFIED_FILE, "r") as f:
            all_data = json.load(f)
    else:
        all_data = {}
    return all_data.get(today, [])

def save_notified(notified_today):
    today = datetime.utcnow().strftime("%Y-%m-%d")
    if os.path.exists(NOTIFIED_FILE):
        with open(NOTIFIED_FILE, "r") as f:
            all_data = json.load(f)
    else:
        all_data = {}
    all_data[today] = notified_today
    with open(NOTIFIED_FILE, "w") as f:
        json.dump(all_data, f)

@app.route("/")
def index():
    return "OK", 200

@app.route("/run_analysis")
def run_analysis():
    print("[INFO] 処理開始")
    symbols = fetch_okx_symbols()
    filtered = filter_symbols(symbols)
    notified = load_notified()

    for sym, df in filtered:
        if sym in notified:
            continue
        result = analyze_with_groq(sym, df)
        if not result or "利益の出る確率" not in result:
            continue
        try:
            prob = int(result.split("利益の出る確率")[1].split("%")[0].strip("：: ").strip())
        except:
            prob = 0
        if prob >= 80:
            buf = plot_chart(df, sym)
            send_telegram_image(buf, f"{sym} 分析結果")
            send_telegram_message(f"{sym}\n{result}")
            notified.append(sym)
            print(f"[INFO] 通知: {sym}")
    save_notified(notified)
    return "OK", 200

if __name__ == "__main__":
    app.run()
