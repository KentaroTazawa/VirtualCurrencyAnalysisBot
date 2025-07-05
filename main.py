import requests
import os
import openai
import time

# === 環境変数 ===
openai.api_key = os.getenv("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# === 急変動上位30通貨を取得（change24hは手計算） ===
def get_top_movers_okx(limit=30):
    url = "https://www.okx.com/api/v5/market/tickers"
    params = {"instType": "SWAP"}  # 無期限先物
    res = requests.get(url, params=params)

    if res.status_code != 200:
        raise ValueError(f"OKX ticker取得失敗: {res.status_code} / {res.text}")

    data = res.json().get("data", [])
    tickers_with_change = []

    for t in data:
        try:
            if not t["instId"].endswith("-USDT"):
                continue
            last = float(t["last"])
            open_ = float(t["open24h"])
            if open_ == 0:
                continue
            change_pct = (last - open_) / open_ * 100
            tickers_with_change.append((t["instId"], abs(change_pct)))
        except Exception:
            continue

    sorted_tickers = sorted(tickers_with_change, key=lambda x: x[1], reverse=True)
    return [t[0] for t in sorted_tickers[:limit]]

# === 15分足の終値データ取得 ===
def fetch_okx_closes(symbol="BTC-USDT", interval="15m", limit=50):
    url = "https://www.okx.com/api/v5/market/candles"
    params = {"instId": symbol, "bar": interval, "limit": limit}
    res = requests.get(url, params=params)

    if res.status_code != 200:
        raise ValueError(f"OKXローソク足取得失敗: {res.status_code} / {res.text}")

    candles = res.json().get("data", [])
    closes = [float(c[4]) for c in candles]
    closes.reverse()
    return closes

# === GPTに送信して分析 ===
def send_to_gpt(closes, symbol="BTC-USDT"):
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

# === Telegram通知 ===
def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg})
    except Exception as e:
        print(f"Telegram送信エラー: {e}")

# === メイン処理 ===
def main():
    try:
        top_symbols = get_top_movers_okx(limit=30)
        for symbol in top_symbols:
            try:
                time.sleep(0.4)  # レート制限対策
                closes = fetch_okx_closes(symbol=symbol, interval="15m", limit=50)
                result = send_to_gpt(closes, symbol=symbol)
                send_telegram(f"📉 {symbol} ショート分析結果（OKX 15分足）\n\n{result}")
            except Exception as e:
                send_telegram(f"⚠️ {symbol} 分析エラー: {e}")
    except Exception as e:
        send_telegram(f"❗️Bot全体エラー: {e}")
        print(f"全体エラー: {e}")

if __name__ == "__main__":
    main()
