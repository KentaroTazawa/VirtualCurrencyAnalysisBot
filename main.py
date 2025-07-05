import requests
import os
import json
import datetime
import openai
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from telegram import Bot

# 環境変数
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
openai.api_key = OPENAI_API_KEY

# 通知済み銘柄ファイル
NOTIFIED_FILE = "notified_today.json"

def load_notified_symbols():
    today = str(datetime.date.today())
    if os.path.exists(NOTIFIED_FILE):
        with open(NOTIFIED_FILE, 'r') as f:
            data = json.load(f)
        if data.get("date") == today:
            return set(data.get("symbols", []))
    return set()

def save_notified_symbols(symbols):
    today = str(datetime.date.today())
    with open(NOTIFIED_FILE, 'w') as f:
        json.dump({"date": today, "symbols": list(symbols)}, f)

def get_usdt_swap_symbols():
    url = "https://www.okx.com/api/v5/public/instruments?instType=SWAP"
    res = requests.get(url).json()
    symbols = [item["instId"] for item in res["data"] if item["instId"].endswith("-USDT-SWAP")]
    return symbols

def fetch_ohlcv(inst_id, bar="15m", limit=100):
    url = f"https://www.okx.com/api/v5/market/candles?instId={inst_id}&bar={bar}&limit={limit}"
    res = requests.get(url).json()
    if "data" not in res:
        return []
    return list(reversed(res["data"]))

def calculate_rsi(closes, period=14):
    delta = pd.Series(closes).diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = -delta.clip(upper=0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

def generate_chart(inst_id, closes):
    plt.figure(figsize=(8, 3))
    plt.plot(closes, label=inst_id)
    plt.title(f"{inst_id} 15m Close")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

def gpt_analysis(inst_id, closes):
    prompt = f"""
以下は仮想通貨{inst_id}の15分足終値データ（最新から過去へ100本）です：
{', '.join([str(c) for c in closes])}

このチャートを分析して、「今ショートを仕掛けるべきか？」を判断し、
利確ライン（TP）、損切ライン（SL）、利益が出る確率（%）を提案してください。

形式：
・ショートすべきか：はい / いいえ
・理由：
・利確目安（TP）：
・損切目安（SL）：
・利益が出る確率（%）：
"""
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "あなたは熟練のトレーダーAIです。"},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ GPTエラー: {e}"

def main():
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    notified_symbols = load_notified_symbols()
    symbols = get_usdt_swap_symbols()

    rsi_data = []
    for sym in symbols:
        if sym in notified_symbols:
            continue
        ohlcv = fetch_ohlcv(sym)
        if len(ohlcv) < 20:
            continue
        closes = [float(c[4]) for c in ohlcv]
        rsi = calculate_rsi(closes)
        if rsi > 70:
            rsi_data.append((sym, rsi, closes))

    top_rsi = sorted(rsi_data, key=lambda x: x[1], reverse=True)[:3]

    for sym, rsi_val, closes in top_rsi:
        result = gpt_analysis(sym, closes)
        if "利益が出る確率（%）：" in result:
            try:
                percent = int(result.split("利益が出る確率（%）：")[-1].strip().replace("%", ""))
                if percent >= 80:
                    img = generate_chart(sym, closes)
                    bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=img, caption=f"📉 {sym} ショート分析結果（OKX 15分足）\n\n{result}")
                    notified_symbols.add(sym)
            except:
                pass

    save_notified_symbols(notified_symbols)

if __name__ == "__main__":
    main()
