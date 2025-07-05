import requests
import os
import openai

# === 環境変数の読み込み ===
openai.api_key = os.getenv("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# === 1. Bybitから価格変動が大きい上位30通貨を取得 ===
def get_top_movers(limit=30):
    url = "https://api.bybit.com/v5/market/tickers"
    params = {"category": "linear"}
    res = requests.get(url, params=params).json()

    if res.get("retCode") != 0:
        raise ValueError(f"Bybit ticker APIエラー: {res}")

    tickers = res["result"]["list"]
    sorted_tickers = sorted(
        tickers,
        key=lambda x: abs(float(x["change24h"])),  # 上昇・下落の絶対値でソート
        reverse=True
    )
    top_symbols = [t["symbol"] for t in sorted_tickers if t["symbol"].endswith("USDT")]
    return top_symbols[:limit]

# === 2. 各通貨のOHLCV（終値）を取得 ===
def fetch_ohlcv_bybit(symbol="BTCUSDT", interval="15", limit=50):
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,  # "1", "3", "5", "15", etc.
        "limit": limit
    }
    res = requests.get(url, params=params).json()

    if res.get("retCode") != 0:
        raise ValueError(f"Bybit kline APIエラー: {res}")

    candles = res["result"]["list"]
    closes = [float(c[4]) for c in candles]
    closes.reverse()
    return closes

# === 3. GPT-4に送って分析させる ===
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
    res = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "あなたは熟練のトレーダーAIです。"},
            {"role": "user", "content": prompt}
        ]
    )
    return res["choices"][0]["message"]["content"]

# === 4. Telegram通知 ===
def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    res = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg})
    return res.json()

# === 5. メイン処理 ===
def main():
    try:
        top_symbols = get_top_movers(limit=30)
        for symbol in top_symbols:
            try:
                closes = fetch_ohlcv_bybit(symbol=symbol, interval="15", limit=50)
                result = send_to_gpt(closes, symbol=symbol)
                send_telegram(f"📉 {symbol} ショート分析結果（Bybit 15分足）\n\n{result}")
            except Exception as e:
                print(f"⚠️ {symbol} 分析エラー: {e}")
                continue
    except Exception as e:
        send_telegram(f"❗️Bot全体でエラーが発生しました: {e}")
        print(f"全体エラー: {e}")

if __name__ == "__main__":
    main()
