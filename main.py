import os
import json
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

NOTIFIED_FILE = "notified_pairs.json"

def log(msg):
    print(msg, flush=True)

def calculate_rsi(prices, period=14):
    deltas = np.diff(prices)
    seed = deltas[:period]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi = 100 - 100 / (1 + rs)
    return round(rsi, 2)

def calculate_macd(prices):
    ema12 = pd.Series(prices).ewm(span=12).mean()
    ema26 = pd.Series(prices).ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    recent_cross = macd.iloc[-2] > signal.iloc[-2] and macd.iloc[-1] < signal.iloc[-1]
    return "dead" if recent_cross else "none"

def calculate_ma_gap(prices):
    ma = np.mean(prices[-25:])
    current = prices[-1]
    return round(((current - ma) / ma) * 100, 2)

def is_volume_spike(volumes):
    avg = np.mean(volumes[:-5])
    return volumes[-1] > avg * 1.5

def fetch_ohlcv(symbol):
    url = f"https://www.okx.com/api/v5/market/candles?instId={symbol}&bar=15m&limit=100"
    res = requests.get(url).json()
    if res.get("code") != "0":
        log(f"[WARN] OHLCV取得失敗: {symbol} - {res.get('msg')}")
        return [], []
    closes = [float(c[4]) for c in reversed(res["data"])]
    volumes = [float(c[5]) for c in reversed(res["data"])]
    return closes, volumes

def fetch_symbols():
    url = "https://www.okx.com/api/v5/market/tickers?instType=SWAP"
    res = requests.get(url).json()
    if res.get("code") != "0":
        log(f"[WARN] 銘柄リスト取得失敗 - {res.get('msg')}")
        return []
    symbols = [item["instId"] for item in res["data"] if item["instId"].endswith("-USDT-SWAP")]
    log(f"[INFO] 取得銘柄数: {len(symbols)}")
    return symbols

def generate_chart(prices, symbol):
    plt.figure(figsize=(6, 3))
    plt.plot(prices, color='red')
    plt.title(f"{symbol} 15m Chart")
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

def send_telegram(photo, caption):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    files = {"photo": ("chart.png", photo)}
    data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
    res = requests.post(url, files=files, data=data)
    if res.status_code != 200:
        log(f"[WARN] Telegram通知失敗: {res.text}")
    else:
        log("[INFO] Telegram通知成功")

def analyze_with_groq(symbol, rsi, macd, gap, volume_spike):
    prompt = f"""
あなたは熟練の仮想通貨トレーダーAIです。
以下のテクニカル情報を元に、この銘柄をショートすべきか判断してください。

銘柄: {symbol}
・RSI: {rsi}
・MACDクロス: {"デッドクロス" if macd == "dead" else "なし"}
・移動平均乖離率: {gap}%
・出来高急増: {"あり" if volume_spike else "なし"}

フォーマット：
・ショートすべきか（はい/いいえ）：
・理由：
・利確ライン（TP）：
・損切ライン（SL）：
・利益の出る確率：
"""
    # Groq APIのrequestsによる直接呼び出し例（ダミーURLとヘッダー例）
    url = "https://api.groq.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()
        return content
    except Exception as e:
        log(f"⚠️ Groqエラー: {e}")
        return None

def load_notified():
    today = datetime.utcnow().strftime("%Y-%m-%d")
    if os.path.exists(NOTIFIED_FILE):
        with open(NOTIFIED_FILE) as f:
            data = json.load(f)
        return set(data.get(today, []))
    return set()

def save_notified(pairs):
    today = datetime.utcnow().strftime("%Y-%m-%d")
    if os.path.exists(NOTIFIED_FILE):
        with open(NOTIFIED_FILE) as f:
            data = json.load(f)
    else:
        data = {}
    data[today] = list(pairs)
    with open(NOTIFIED_FILE, "w") as f:
        json.dump(data, f, indent=2)

def main():
    now = datetime.utcnow() + timedelta(hours=9)
    if not (now.hour >= 20 or (now.hour == 0 and now.minute <= 30)):
        log("[INFO] 実行時間外のためスキップ")
        return

    log("[INFO] 処理開始")
    notified = load_notified()
    symbols = fetch_symbols()
    new_notify = set()

    for symbol in symbols:
        if symbol in notified:
            log(f"[DEBUG] {symbol} は既に通知済みのためスキップ")
            continue
        prices, volumes = fetch_ohlcv(symbol)
        if len(prices) < 30:
            log(f"[DEBUG] {symbol} の価格データが不足({len(prices)})のためスキップ")
            continue
        rsi = calculate_rsi(prices)
        macd = calculate_macd(prices)
        gap = calculate_ma_gap(prices)
        volume_spike = is_volume_spike(volumes)

        log(f"[DEBUG] {symbol} RSI={rsi} MACD={macd} MA乖離率={gap}% 出来高急増={'あり' if volume_spike else 'なし'}")

        # 判定条件
        if rsi < 70:
            log(f"[DEBUG] {symbol} はRSI < 70のためスキップ")
            continue
        if macd != "dead":
            log(f"[DEBUG] {symbol} はデッドクロスなしのためスキップ")
            continue
        if gap < 5:
            log(f"[DEBUG] {symbol} はMA乖離率 < 5%のためスキップ")
            continue
        if not volume_spike:
            log(f"[DEBUG] {symbol} は出来高急増なしのためスキップ")
            continue

        log(f"[INFO] Groq分析中: {symbol} (RSI={rsi})")
        result = analyze_with_groq(symbol, rsi, macd, gap, volume_spike)
        if result is None:
            log(f"[WARN] {symbol} のGroq分析に失敗")
            continue
        log(f"[Groq分析結果] {symbol}\n{result}")

        # 利益の出る確率を抽出
        prob = 0
        if "利益の出る確率：" in result:
            try:
                prob_str = result.split("利益の出る確率：")[-1].split("\n")[0].replace("%", "").strip()
                prob = int(prob_str)
            except Exception as e:
                log(f"[WARN] {symbol} 利益確率解析失敗: {e}")
                continue

        if prob >= 80:
            chart = generate_chart(prices, symbol)
            send_telegram(chart, f"📉 {symbol} ショート分析\n\n{result}")
            new_notify.add(symbol)
        else:
            log(f"[DEBUG] {symbol} 利益確率 {prob}% は80%未満のため通知しない")

    save_notified(notified | new_notify)

    if new_notify:
        log(f"[INFO] 通知済み件数: {len(new_notify)}")
    else:
        log("[INFO] 通知対象がなかったためTelegram通知なし")

if __name__ == "__main__":
    main()
