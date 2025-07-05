import requests
import os
import openai
import time

# === 環境変数からAPIキーなどを読み込み ===
openai.api_key = os.getenv("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# === 共通のリクエストヘッダー（CloudFront対策）===
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; CryptoBot/1.0; +https://example.com/bot)"
}

# === 1. Bybitの急変動上位通貨を取得 ===
def get_top_movers(limit=30):
    url = "https://api.bybit.com/v5/market/tickers"
    params = {"category": "linear"}
    res = requests.get(url, params=params, headers=HEADERS)

    if res.status_code != 200:
        raise ValueError(f"Ticker APIエラー: {res.status_code} / {res.text}")

    data = res.json()
    if data.get("retCode") != 0:
        raise ValueError(f"Ticker APIレスポンス異常: {data}")

    tickers = data["result"]["list"]
    sorted_tickers = sorted(
        tickers,
        key=lambda x: abs(float(x["change24h"])),  # 上昇/下落の大きさ順
        reverse=True
    )
    top_symbols = [t["symbol"] for t in sorted_tickers if t["symbol"].endswith("USDT")]
    return top_symbols[:limit]

# === 2. 各通貨の終値データを取得 ===
def fetch_ohlcv_bybit(symbol="BTCUSDT", interval="15", limit=50):
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    res = requests.get(url, params=params, headers=HEADERS)

    if res.status_code != 200:
        raise ValueError(f"Kline APIエラー: {res.status_code} / {res.text}")

    data = res.json()
    if data.get("retCode") != 0:
        raise ValueError(f"Kline APIレスポンス異常: {data}")

    candles = data["result"]["list"]
    closes = [float(c[4]) for c in candles]
    closes.reverse()  # 古い順に並べ替え
    return closes

# === 3. GPT-4に送ってショート判断をさせる ===
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
        return f"⚠️ GPTエラー: {e}"

# === 4. Telegram通知 ===
def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg})
    except Exception as e:
        print(f"Telegram送信エラー: {e}")

# === 5. メイン処理 ===
def main():
    try:
        top_symbols = get_top_movers(limit=30)
        for symbol in top_symbols:
            try:
                time.sleep(0.4)  # Bybitレート制限対策（1秒に2回以下）
                closes = fetch_ohlcv_bybit(symbol=symbol, interval="15", limit=50)
                result = send_to_gpt(closes, symbol=symbol)
                send_telegram(f"📉 {symbol} ショート分析結果（Bybit 15分足）\n\n{result}")
            except Exception as e:
                send_telegram(f"⚠️ {symbol} 分析エラー: {e}")
                continue
    except Exception as e:
        send_telegram(f"❗️Bot全体でエラーが発生しました: {e}")
        print(f"全体エラー: {e}")

if __name__ == "__main__":
    main()
