import os
import json
import time
import threading
import requests
from datetime import datetime, timedelta
from flask import Flask
import openai
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np

# --- 環境変数読み込み ---
openai.api_key = os.getenv("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
NOTIFIED_FILE = "notified_pairs.json"

def load_notified():
    today = (datetime.utcnow() + timedelta(hours=9)).strftime("%Y-%m-%d")
    if os.path.exists(NOTIFIED_FILE):
        with open(NOTIFIED_FILE, "r") as f:
            data = json.load(f)
        return set(data.get(today, []))
    return set()

def save_notified(pairs):
    today = (datetime.utcnow() + timedelta(hours=9)).strftime("%Y-%m-%d")
    if os.path.exists(NOTIFIED_FILE):
        with open(NOTIFIED_FILE, "r") as f:
            data = json.load(f)
    else:
        data = {}
    data[today] = list(pairs)
    with open(NOTIFIED_FILE, "w") as f:
        json.dump(data, f)

def calculate_rsi(prices, period=14):
    prices = np.array(prices)
    deltas = np.diff(prices)
    seed = deltas[:period]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi = 100 - 100 / (1 + rs)
    return round(rsi, 2)

def generate_chart(prices, symbol):
    plt.figure(figsize=(6, 3))
    plt.plot(prices, color='red')
    plt.title(f"{symbol} 15m Close Price")
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

def analyze_with_gpt(prices, symbol):
    prompt = f"""
以下は{symbol}の15分足の終値データです：
{', '.join(map(str, prices))}

このチャートを分析して、
・ショートすべきか（はい/いいえ）
・理由
・利確ライン（TP）
・損切ライン（SL）
・利益の出る確率（％）
を出力してください。
形式：
・ショートすべきか：
・理由：
・利確ライン（TP）：
・損切ライン（SL）：
・利益の出る確率：
"""
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "あなたは優秀なトレードアナリストAIです。"},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ GPTエラー: {e}", flush=True)
        return None

def send_telegram_image(image_buf, caption):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    files = {"photo": ("chart.png", image_buf)}
    data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
    response = requests.post(url, files=files, data=data)
    return response.json()

def fetch_okx_symbols():
    url = "https://www.okx.com/api/v5/market/tickers?instType=SWAP"
    res = requests.get(url).json()
    if res.get("code") != "0":
        print("[ERROR] OKXデータ取得失敗", res, flush=True)
        return []
    return [item for item in res["data"] if item["instId"].endswith("-USDT-SWAP")]

def fetch_ohlcv(symbol):
    url = f"https://www.okx.com/api/v5/market/candles?instId={symbol}&bar=15m&limit=100"
    res = requests.get(url).json()
    if res.get("code") != "0":
        print(f"[ERROR] OHLCV取得失敗 {symbol}", res, flush=True)
        return []
    closes = [float(c[4]) for c in reversed(res["data"])]
    return closes

def main():
    now = datetime.utcnow() + timedelta(hours=9)  # JST
    print(f"[INFO] 処理開始（現在: {now.strftime('%H:%M')} JST）", flush=True)

    if not (now.hour == 20 and now.minute >= 30) and not (21 <= now.hour <= 23 or (now.hour == 0 and now.minute <= 30)):
        print("[INFO] 実行時間外のためスキップ", flush=True)
        return

    notified_today = load_notified()
    symbols = fetch_okx_symbols()
    rsi_results = []
    for item in symbols:
        symbol = item["instId"]
        if symbol in notified_today:
            continue
        prices = fetch_ohlcv(symbol)
        if len(prices) < 20:
            continue
        rsi = calculate_rsi(prices)
        if rsi > 70:
            rsi_results.append((symbol, rsi, prices))

    rsi_results.sort(key=lambda x: x[1], reverse=True)
    top3 = rsi_results[:3]
    newly_notified = set()

    for symbol, rsi, prices in top3:
        print(f"[INFO] GPT分析中: {symbol} (RSI={rsi})", flush=True)
        result = analyze_with_gpt(prices, symbol)
        if not result:
            continue
        print(f"[GPT分析結果] {symbol}\n{result}\n", flush=True)
        if "利益の出る確率" in result:
            try:
                percent = int(result.split("利益の出る確率：")[-1].replace("%", "").strip())
                if percent >= 80:
                    caption = f"\ud83d\udcc9 {symbol} ショート分析結果（OKX 15分足）\n\n{result}"
                    chart = generate_chart(prices, symbol)
                    send_telegram_image(chart, caption)
                    newly_notified.add(symbol)
            except:
                continue

    if newly_notified:
        notified_today |= newly_notified
        save_notified(notified_today)
        chart = generate_chart([0], "確認")
        send_telegram_image(chart, f"\u2705 Bot処理完了：{len(newly_notified)}件通知")
    else:
        print("[INFO] 通知対象がなかったためTelegram通知なし", flush=True)

def schedule_loop():
    while True:
        print("[スケジューラー] main() を実行します", flush=True)
        try:
            main()
        except Exception as e:
            print(f"[ERROR] main処理中にエラー: {e}", flush=True)

        for i in range(5):
            print(f"[スケジューラー] 次回実行まであと {5 - i} 分", flush=True)
            time.sleep(60)

# --- Flaskサーバー起動 ---
app = Flask(__name__)

@app.route("/")
def index():
    return "OK"

if __name__ == "__main__":
    threading.Thread(target=schedule_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=10000)
