import requests
import os
import openai
import time

# 環境変数から読み込み
openai.api_key = os.getenv("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg})
    except Exception as e:
        print(f"Telegram送信エラー: {e}")

def get_top_movers_okx(limit=10):
    url = "https://www.okx.com/api/v5/market/tickers"
    params = {"instType": "SWAP"}
    res = requests.get(url, params=params)
    if res.status_code != 200:
        raise ValueError(f"OKX ticker取得失敗: {res.status_code} / {res.text}")
    data = res.json().get("data", [])
    tickers_with_change = []
    for t in data:
        try:
            instId = t.get("instId", "")
            if "-USDT" not in instId:
                continue
            last = float(t["last"])
            open_ = float(t["open24h"])
            if open_ == 0:
                continue
            change_pct = (last - open_) / open_ * 100
            tickers_with_change.append((instId, abs(change_pct)))
        except Exception as e:
            print(f"ティッカー処理エラー: {e}")
            continue
    sorted_tickers = sorted(tickers_with_change, key=lambda x: x[1], reverse=True)
    return [t[0] for t in sorted_tickers[:limit]]

def fetch_okx_closes(symbol="BTC-USDT", interval="15m", limit=50):
    url = "https://www.okx.com/api/v5/market/candles"
    params = {"instId": symbol, "bar": interval, "limit": limit}
    res = requests.get(url, params=params)
    if res.status_code != 200:
        raise ValueError(f"OKXローソク足取得失敗: {res.status_code} / {res.text}")
    candles = res.json().get("data", [])
    closes = [float(c[4]) for c in candles]
    closes.reverse()  # 古い順に並び替え
    return closes

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
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # gpt-4より安価で無料枠向き
            messages=[
                {"role": "system", "content": "あなたは熟練のトレーダーAIです。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ GPTエラー: {e}"

def main():
    send_telegram("🚀 Bot起動確認：main.py 実行スタート ✅")
    try:
        top_symbols = get_top_movers_okx(limit=10)
        send_telegram(f"📊 対象銘柄数: {len(top_symbols)}")
        for symbol in top_symbols:
            try:
                time.sleep(3)  # GPT呼び出しごとに3秒待機
                closes = fetch_okx_closes(symbol=symbol, interval="15m", limit=50)
                result = send_to_gpt(closes, symbol=symbol)
                send_telegram(f"📉 {symbol} ショート分析結果（OKX 15分足）\n\n{result}")
            except Exception as e:
                send_telegram(f"⚠️ {symbol} 分析エラー: {e}")
    except Exception as e:
        send_telegram(f"❗️Bot全体エラー: {e}")
    finally:
        send_telegram("✅ Bot処理完了しました（main.py 終了）")

if __name__ == "__main__":
    main()
