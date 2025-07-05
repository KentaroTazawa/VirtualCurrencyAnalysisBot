import os
import json
import time
import threading
import requests
from datetime import datetime, timezone, timedelta
from flask import Flask
import openai
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# --- フォント設定（日本語警告回避用） ---
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# --- 環境変数 ---
openai.api_key = os.getenv("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# --- 通知履歴ファイル ---
NOTIFIED_FILE = "notified_pairs.json"

def get_jst_now():
    return datetime.now(timezone.utc) + timedelta(hours=9)

def load_notified():
    today = get_jst_now().strftime("%Y-%m-%d")
    if os.path.exists(NOTIFIED_FILE):
        with open(NOTIFIED_FILE, "r") as f:
            data = json.load(f)
        return set(data.get(today, []))
    return set()

def save_notified(pairs):
    today = get_jst_now().strftime("%Y-%m-%d")
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
    plt.figure(figsize=(6,3))
    plt.plot(prices, color='red')
    plt.title(f"{symbol} 15m Close Price")
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
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
・損切りライン（SL）
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
            model="gpt-4",
            messages=[
                {"role": "system", "content": "あなたは優秀なトレードアナリストAIです。"},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ GPTエラー: {e}"

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
        print("[ERROR] OKXデータ取得失敗", res)
        return []
    return [item for item in res["data"] if item["instId"].endswith("-USDT-SWAP")]

def fetch_ohlcv(symbol):
    url = f"https://www.okx.com/api/v5/market/candles?instId={symbol}&bar=15m&limit=100"
    res = requests.get(url).json()
    if res.get("code") != "0":
        print(f"[ERROR] OHLCV取得失敗 {symbol}", res)
        return []
    closes = [float(c[4]) for c in reversed(res["data"])]
    return closes

def main():
    now = get_jst_now()
    if not (
        (now.hour == 20 and now.minute >= 30) or
        (21 <= now.hour <= 23) or
        (now.hour == 0 and now.minute <= 30)
    ):
        print(f"[INFO] 実行時間外のためスキップ（現在: {now.strftime('%H:%M')} JST）")
        return

    print(f"[INFO] 処理開始（現在: {now.strftime('%H:%M')} JST）")

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

    if not rsi_results:
        print("[INFO] RSIが70超の銘柄なし")
        return

    print("[INFO] RSI70超銘柄一覧：")
    for sym, rsi_val, _ in sorted(rsi_results, key=lambda x: x[1], reverse=True)[:10]:
        print(f" - {sym}: RSI={rsi_val}")

    rsi_results.sort(key=lambda x: x[1], reverse=True)
    top3 = rsi_results[:3]
    newly_notified = set()

    for symbol, rsi, prices in top3:
        result = analyze_with_gpt(prices, symbol)
        print(f"[GPT分析結果] {symbol}\n{result}\n")
        if "利益の出る確率" in result:
            try:
                percent = int(result.split("利益の出る確率：")[-1].replace("%", "").strip())
                if percent >= 80:
                    caption = f"📉 {symbol} ショート分析結果（OKX 15分足）\n\n{result}"
                    chart = generate_chart(prices, symbol)
                    send_telegram_image(chart, caption)
                    newly_notified.add(symbol)
                    print(f"[通知] {symbol} をTelegramに通知しました（利益確率 {percent}%）")
                else:
                    print(f"[スキップ] {symbol} 利益確率 {percent}% は80%未満")
            except Exception as e:
                print(f"[ERROR] GPT結果解析エラー: {e}")
                continue

    notified_today |= newly_notified
    save_notified(notified_today)

    if newly_notified:
        send_telegram_image(generate_chart([0], "CONFIRM"), f"✅ Bot処理完了：{len(newly_notified)}件通知")
        print(f"[INFO] 通知完了：{len(newly_notified)}件")
    else:
        print("[INFO] 通知対象なし：Telegram通知スキップ")

def schedule_loop():
    while True:
        try:
            print("[スケジューラー] main() を実行します")
            main()
        except Exception as e:
            print("[ERROR] main処理中にエラー:", e)

        for i in range(10, 0, -1):
            print(f"[スケジューラー] 次の実行まであと {i} 分...")
            time.sleep(60)

# --- Flask サーバー起動 ---
app = Flask(__name__)

@app.route("/")
def index():
    return "OK"

if __name__ == "__main__":
    threading.Thread(target=schedule_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=10000)
