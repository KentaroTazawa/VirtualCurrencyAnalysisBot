import requests
import os
import openai
import time

openai.api_key = os.getenv("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def get_top_movers(limit=30):
    url = "https://api.bybit.com/v5/market/tickers"
    params = {"category": "linear"}
    res = requests.get(url, params=params)

    if res.status_code != 200:
        raise ValueError(f"Ticker APIエラー: {res.status_code} / {res.text}")

    data = res.json()
    if data.get("retCode") != 0:
        raise ValueError(f"Ticker APIレスポンス異常: {data}")

    tickers = data["result"]["list"]
    sorted_tickers = sorted(
        tickers,
        key=lambda x: abs(float(x["change24h"])),
        reverse=True
    )
    top_symbols = [t["symbol"] for t in sorted_tickers if t["symbol"].endswith("USDT")]
    return top_symbols[:limit]

def fetch_ohlcv_bybit(symbol="BTCUSDT", interval="15", limit=50):
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    res = requests.get(url, params=params)

    if res.status_code != 200:
        raise ValueError(f"Kline APIエラー: {res.status_code} / {res.text}")

    data = res.json()
    if data.get("retCode") != 0:
        raise ValueError(f"Kline APIレスポンス異常: {data}")

    candles = data["result"]["list"]
    closes = [float(c[4]) for c in candles]
    closes.reverse()
    return closes

def send_to_gpt(closes, symbol="BTCUSDT"):
    text = ", ".join([f"{c:.2f}" for c in closes])
    prompt = f"""
以下は仮想通貨 {symbol} の15分足終値データ（最新から過去へ50本）です：
{text}

このチャートを分析して、「今ショートを仕掛けるべきか？」を判断し、
利確ライン（TP）と損切ライン（SL）も数字で提案してください。

形式：
・ショートすべきか：はい / いいえ
・理由：
・利確目安（TP）：
・損切目安（SL）：
"""

    try:
        res = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "あなたは熟練のトレーダーAIです。"},
                {"role": "user", "content": prompt}
            ]
        )
        return res["choices"][0]["message"]["content"]
    except Exception as e:
        return f"GPTエラー: {e}"

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg})
    except Exception as e:
        print(f"Telegram送信エラー: {e}")

def main():
    try:
        top_symbols = get_top_movers(limit=30)
        for symbol in top_symbols:
            try:
                time.sleep(0.3)  # レート制限対策
                closes = fetch_ohlcv_bybit(symbol=symbol, interval="15", limit=50)
                result = send_to_gpt(closes, symbol=symbol)
                send_telegram(f"📉 {symbol} ショート分析結果（Bybit 15分足）\n\n{result}")
            except Exception as e:
                send_telegram(f"⚠️ {symbol} 分析エラー: {e}")
                continue
    except Exception as e:
        send_telegram(f"❗️全体エラー: {e}")
        print(f"全体エラー: {e}")

if __name__ == "__main__":
    main()
