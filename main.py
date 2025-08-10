import os
import json
import time
import traceback
from datetime import datetime
import requests
import pandas as pd
from flask import Flask
from groq import Groq
from dotenv import load_dotenv
import re

load_dotenv()

MEXC_BASE_URL = "https://contract.mexc.com"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)
app = Flask(__name__)

TOP_SYMBOLS_LIMIT = 30  # 24h変化率トップxx対象

def send_error_to_telegram(error_message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(
            url,
            data={"chat_id": TELEGRAM_CHAT_ID, "text": f"⚠️ エラー発生:\n\n{error_message}"},
            timeout=10
        )
    except:
        pass

def get_top_symbols_by_24h_change(limit=TOP_SYMBOLS_LIMIT):
    """MEXC の先物ティッカー（24h変化）を取得"""
    try:
        url = f"{MEXC_BASE_URL}/api/v1/contract/ticker"
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        data = res.json()

        tickers = data.get("data", [])
        filtered = []
        for t in tickers:
            try:
                symbol = t.get("symbol", "")
                last_price = float(t.get("lastPrice", 0))
                rise_fall_rate = float(t.get("riseFallRate", 0)) * 100  # 0.0139 -> 1.39%
                filtered.append({"symbol": symbol, "last_price": last_price, "change_pct": rise_fall_rate})
            except:
                continue
        sorted_tickers = sorted(filtered, key=lambda x: x["change_pct"], reverse=True)
        return sorted_tickers[:limit]
    except requests.exceptions.Timeout:
        send_error_to_telegram("MEXC 急上昇銘柄取得エラー: タイムアウト発生")
        return []
    except Exception as e:
        send_error_to_telegram(f"MEXC 急上昇銘柄取得エラー:\n{str(e)}")
        return []

def get_available_contract_symbols():
    """contract/detail から先物の正式 symbol 一覧を取得（フィルター用）"""
    try:
        url = f"{MEXC_BASE_URL}/api/v1/contract/detail"
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        data = res.json()
        arr = data.get("data", []) or []
        return [it.get("symbol") for it in arr if it.get("symbol")]
    except Exception as e:
        send_error_to_telegram(f"先物銘柄一覧取得失敗:\n{str(e)}")
        return []

def fetch_ohlcv(symbol, interval='15m', max_retries=3, timeout_sec=15):
    """
    MEXC の contract K-line を取得する（retry有り）
    使うエンドポイント: /api/v1/contract/kline/{symbol}?interval=Min15
    （interval は MEXC 形式にマッピング）
    """
    # interval mapping
    imap = {
        '1m': 'Min1', '5m': 'Min5', '15m': 'Min15', '30m': 'Min30',
        '60m': 'Min60', '4h': 'Hour4', '8h': 'Hour8', '1d': 'Day1',
        '1w': 'Week1', '1M': 'Month1'
    }
    interval_param = imap.get(interval, 'Min15')
    url = f"{MEXC_BASE_URL}/api/v1/contract/kline/{symbol}?interval={interval_param}"

    for attempt in range(1, max_retries + 1):
        try:
            res = requests.get(url, timeout=timeout_sec)
            res.raise_for_status()
            data = res.json()

            # APIが success=false を返す場合のチェック
            if not data.get("success", False):
                # エラーメッセージを取り、リトライ/終了を判断
                err_msg = data.get("message") or data.get("code") or "Unknown"
                raise ValueError(f"API returned success=false: {err_msg}")

            k = data.get("data", {}) or {}
            times = k.get("time") or []
            if not times:
                raise ValueError("kline data empty")

            # debug: 最初のローソクをログ
            first_sample = {
                "time": times[0],
                "open": (k.get("open")[0] if k.get("open") else None),
                "high": (k.get("high")[0] if k.get("high") else None),
                "low": (k.get("low")[0] if k.get("low") else None),
                "close": (k.get("close")[0] if k.get("close") else None),
                "vol": (k.get("vol")[0] if k.get("vol") else None),
            }
            print(f"📝 {symbol} kline sample (first): {first_sample}")

            # build rows from arrays (length may vary, so guard indexes)
            open_arr = k.get("open", [])
            high_arr = k.get("high", [])
            low_arr = k.get("low", [])
            close_arr = k.get("close", [])
            vol_arr = k.get("vol", [])

            rows = []
            n = len(times)
            for i in range(n):
                row = {
                    "ts": int(times[i]),
                    "open": float(open_arr[i]) if i < len(open_arr) and open_arr[i] is not None else None,
                    "high": float(high_arr[i]) if i < len(high_arr) and high_arr[i] is not None else None,
                    "low": float(low_arr[i]) if i < len(low_arr) and low_arr[i] is not None else None,
                    "close": float(close_arr[i]) if i < len(close_arr) and close_arr[i] is not None else None,
                    "vol": float(vol_arr[i]) if i < len(vol_arr) and vol_arr[i] is not None else None,
                }
                rows.append(row)

            df = pd.DataFrame(rows)
            # 時刻は秒単位の可能性が高いので、必要なら ms に変換するなどの処理はここで行う
            df = df.sort_values("ts").reset_index(drop=True)
            return df

        except requests.exceptions.Timeout:
            print(f"⚠️ {symbol} のローソク取得タイムアウト（試行 {attempt}/{max_retries}）")
            if attempt == max_retries:
                send_error_to_telegram(f"{symbol} のローソク取得失敗: タイムアウト発生")
        except Exception as e:
            print(f"⚠️ {symbol} のローソク取得エラー: {e}（試行 {attempt}/{max_retries}）")
            # 最終試行なら通知
            if attempt == max_retries:
                send_error_to_telegram(f"{symbol} のローソク取得失敗:\n{str(e)}")
        time.sleep(1)  # リトライ間隔

    return None

def fetch_daily_ohlcv_max(symbol):
    """日足の最大件数まで取得"""
    return fetch_ohlcv(symbol, interval='1d')

def is_ath_today(current_price, df_15m, df_daily):
    try:
        # 15分足と日足の両方から最高値を抽出
        ath_price = max(df_15m["high"].max(), df_daily["high"].max())
        # ATHの90%以上の場合 True とする
        return current_price >= ath_price * 0.9, ath_price
    except Exception:
        return False, None

def analyze_with_groq(df, symbol):
    if len(df) < 2:
        return {"今後下落する可能性は高いか": "不明"}

    latest, prev = df.iloc[-1], df.iloc[-2]

    # 最新100本のみ、必要カラムだけ、小数点4桁
    df_reduced = df.tail(100)[["ts", "open", "high", "low", "close"]].round(4)
    # tsは秒なので、そのままでもいいが可読性のためにdatetime化する場合は以下
    df_reduced["time"] = pd.to_datetime(df_reduced["ts"], unit='s').astype(str)
    df_reduced = df_reduced.drop(columns=["ts"])

    prompt = f"""
以下は {symbol} の15分足データ（最新100本）です。
価格が過去最高であることを踏まえ、今後短期的に下落する可能性を分析してください。

**構造化JSONでのみ返答してください**

{{
  "今後下落する可能性は高いか": "はい" または "いいえ",
  "理由": "～",
  "予測される下落幅": "-x.x%",
  "予測される下落タイミング": "例: 数時間以内、24時間以内など"
}}

参考データ:
- 前回比: {latest['close'] / prev['close']:.4f}
- 直近価格: {latest['close']}

15分足データ（time, open, high, low, close）:
{df_reduced.to_dict(orient='records')}
"""

    print(f"📝 Groqに送信するプロンプト（{symbol}）:\n{prompt}")

    try:
        res = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        content = res.choices[0].message.content
        match = re.search(r"\{[\s\S]*?\}", content)
        return json.loads(match.group(0)) if match else {"今後下落する可能性は高いか": "不明"}
    except Exception as e:
        send_error_to_telegram(f"Groqエラー: {str(e)}")
        return {"今後下落する可能性は高いか": "不明"}
        
def send_to_telegram(symbol, result):
    text = f"""📉 ATH銘柄警告: {symbol}

- 今後下落する可能性: {result.get('今後下落する可能性は高いか', '?')}
- 理由: {result.get('理由', '?')}
- 下落幅予測: {result.get('予測される下落幅', '?')}
- 下落タイミング: {result.get('予測される下落タイミング', '?')}
"""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=10)
    except requests.exceptions.Timeout:
        send_error_to_telegram("Telegram送信エラー: タイムアウト発生")
    except Exception as e:
        send_error_to_telegram(f"Telegram送信エラー:\n{str(e)}")

def run_analysis():
    print("🚀 分析開始")
    top_tickers = get_top_symbols_by_24h_change()
    available = get_available_contract_symbols()
    # 取得可能な symbol のみ残す（念のため）
    top_tickers = [t for t in top_tickers if t["symbol"] in available]
    symbols = [t["symbol"] for t in top_tickers]
    print(f"🔎 対象銘柄: {symbols}")
    for ticker in top_tickers:
        symbol = ticker["symbol"]
        current_price = ticker["last_price"]
        try:
            print("==============================")
            print(f"🔔 {symbol} の処理開始")
            df_15m = fetch_ohlcv(symbol, interval='15m')
            if df_15m is None:
                print(f"⚠️ {symbol} の15分足データ取得失敗。スキップ")
                continue
            df_daily = fetch_daily_ohlcv_max(symbol)
            if df_daily is None:
                print(f"⚠️ {symbol} の日足取得失敗。スキップ")
                continue

            ath_flag, ath_price = is_ath_today(current_price, df_15m, df_daily)
            print(f"💹 {symbol} 現在価格: {current_price} / ATH価格: {ath_price}")
            if not ath_flag:
                print(f"ℹ️ {symbol} はATHではありません。スキップ")
                continue

            result = analyze_with_groq(df_15m, symbol)
            send_to_telegram(symbol, result)
            print(f"✅ {symbol} の分析完了・通知送信済み")
            time.sleep(1)
        except Exception:
            send_error_to_telegram(f"{symbol} 分析中にエラー:\n{traceback.format_exc()}")
    print("✅ 分析終了")

@app.route("/")
def index():
    return "OK"

@app.route("/run_analysis", methods=["GET", "HEAD"])
def run_analysis_route():
    run_analysis()
    return "分析完了", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
