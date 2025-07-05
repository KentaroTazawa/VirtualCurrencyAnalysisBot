import requests
import os
import openai
import time

# === 環境変数読み込み ===
openai.api_key = os.getenv("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# === 急変動上位30通貨を取得（USDT無期限先物）===
def get_top_movers_okx(limit=30):
    url = "https://www.okx.com/api/v5/market/tickers"
    params = {"instType": "SWAP"}  # 無期限先物（日本でも安定）
    res = requests.get(url, params=params)

    if res.status_code != 200:
        raise ValueError(f"OKX ticker取得失敗: {res.status_code} / {res.text}")
    
    data = res.json()
    tickers = data.get("data", [])
    
    # 絶対変動率でソート
    sorted_tickers = sorted(
        tickers,
        key=lambda x: abs(float(x["change24h"])),
        reverse=True
    )

    # USDT建ての通貨のみ抽出
    top_symbols = [t["instId"] for t in sorted_tickers if t["instId"].endswith("-USDT")]
    return top_symbols[:limit]

# === ローソク足（終値）取得 ===
def fetch_okx_closes(symbol="BTC-USDT", interval="15m", limit=50):
    url = "https://www.okx.com/api/v5/market/candles"
    params = {
        "instId": symbol,
        "bar": interval,
        "limit": limit
    }
    res = requests.get(url, params=params)
    if res.status_code != 200:
        raise ValueError(f"OKXローソク足取得失敗: {res.status_code} / {res.text}")
    
    data = res.json().get("data", [])
    closes = [float(c[4]) for c in data]
    closes.reverse()
    return closes

# === GPTに送ってショート判断 ===
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
