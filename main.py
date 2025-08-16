import os
import json
import time
import traceback
from datetime import datetime, timedelta
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

TOP_SYMBOLS_LIMIT = 20  # 24h変化率トップxx対象
NOTIFICATION_CACHE = {}  # {symbol: last_notified_timestamp}


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
                rise_fall_rate = float(t.get("riseFallRate", 0)) * 100
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

            if not data.get("success", False):
                err_msg = data.get("message") or data.get("code") or "Unknown"
                raise ValueError(f"API returned success=false: {err_msg}")

            k = data.get("data", {}) or {}
            times = k.get("time") or []
            if not times:
                raise ValueError("kline data empty")

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
            df = df.sort_values("ts").reset_index(drop=True)
            return df

        except requests.exceptions.Timeout:
            print(f"⚠️ {symbol} のローソク取得タイムアウト（試行 {attempt}/{max_retries}）")
            if attempt == max_retries:
                send_error_to_telegram(f"{symbol} のローソク取得失敗: タイムアウト発生")
        except Exception as e:
            print(f"⚠️ {symbol} のローソク取得エラー: {e}（試行 {attempt}/{max_retries}）")
            if attempt == max_retries:
                send_error_to_telegram(f"{symbol} のローソク取得失敗:\n{str(e)}")
        time.sleep(1)

    return None


def fetch_daily_ohlcv_max(symbol):
    return fetch_ohlcv(symbol, interval='1d')


def is_ath_today(current_price, df_15m, df_daily):
    try:
        ath_price = max(df_15m["high"].max(), df_daily["high"].max())
        return current_price >= ath_price * 0.9, ath_price
    except Exception:
        return False, None


def calculate_indicators(df):
    """RSI, ボラティリティ, 出来高変化率, 移動平均乖離率を計算"""
    result = {}
    if len(df) < 2:
        return result

    close = df['close']
    vol = df['vol']

    # RSI (14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14, min_periods=1).mean()
    avg_loss = loss.rolling(14, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    result['RSI'] = round(rsi.iloc[-1], 2)

    # ボラティリティ（標準偏差）
    result['Volatility'] = round(close.pct_change().rolling(14, min_periods=1).std().iloc[-1] * 100, 2)

    # 出来高変化率
    result['VolChange'] = round(vol.pct_change().rolling(14, min_periods=1).mean().iloc[-1] * 100, 2)

    # 移動平均乖離率（15本）
    ma = close.rolling(15, min_periods=1).mean()
    result['MA_Deviation'] = round((close.iloc[-1] / ma.iloc[-1] - 1) * 100, 2)

    return result


def analyze_with_groq(df, symbol):
    if len(df) < 2:
        return {"今後下落する可能性は高いか": "不明"}

    df_reduced = df.iloc[::-1].iloc[::4].head(100).iloc[::-1]
    records = df_reduced[['ts', 'close', 'vol']].to_dict(orient='records')
    indicators = calculate_indicators(df_reduced)

    # NaNやinfを避けるため安全に文字列化
    safe_indicators = ", ".join([f"{k}: {v}" for k, v in indicators.items()])

    now_plus_9h = datetime.utcnow() + timedelta(hours=9)
    now_str = now_plus_9h.strftime("%Y年%m月%d日 %H:%M")

    prompt = f"""
以下は {symbol} の1時間足相当データ（15分足を4本に1本間引き、最新100本まで）です。
価格が過去最高であることを踏まえ、今後短期的に下落する可能性を分析してください。
各種テクニカル指標も参考にしてください: {safe_indicators}

**必ず以下の条件を守って「厳密なJSON形式」で返答してください**：
- JSONのキー・値はすべてダブルクォーテーションで囲む
- JSON以外の文字は出力しない
- 項目は以下の通り（必ず含める）:
- 「理由」は必ず60文字以内の自然な日本語で書くこと（最後は絵文字で終わること）
- 「下落可能性」は必ず小数第2位までの%で返す（毎回同じような値にならないようにきちんと分析に基づいて示すこと）
- 「下落幅」も必ず小数第2位までの%で返す
- 「下落時期」はJSTで「YYYY年MM月DD日 HH:MM」の形式で返し、きちんと分析に基づいて分刻みで示すこと（現在日時は{now_str}です）
- 「推奨損切り水準」と「推奨利確水準」も必ず小数第1位までの%で返す

この全データ(JSON配列形式)も必ず全て活かして分析してください:
{records}
"""
    print(f"📝 Groqに送信するプロンプト（{symbol}）:\n{prompt}")

    try:
        res = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.25
        )
        content = res.choices[0].message.content

        # JSONを正規化
        match = re.search(r"\{[\s\S]*\}", content)
        if not match:
            raise ValueError("Groq出力にJSONが含まれていません")

        json_text = match.group(0)

        # JSONのキーが日本語の場合でもダブルクォートで囲まれているかチェック
        # （Groqが守らなかった場合のフォールバック）
        fixed_json = re.sub(r'([{\s,])([^\s":]+?):', r'\1"\2":', json_text)

        result = json.loads(fixed_json)
        result['Indicators'] = indicators  # Telegram通知にも追加
        return result
    
    except Exception as e:
        send_error_to_telegram(f"Groqエラー: {str(e)}")
        return {"今後下落する可能性は高いか": "不明"}


def send_to_telegram(symbol, result):
    display_symbol = symbol.replace("_USDT", "")
    indicators = result.get('Indicators', {})
    indicator_text = "\n".join([f"{k}: {v}" for k, v in indicators.items()]) if indicators else ""
    text = f"""📉 ATH下落予測:　{display_symbol}

　予測時刻:　{result.get('下落時期', '?')}
　下落確率:　{result.get('下落可能性', '?')}
下落幅予測:　{result.get('下落幅', '?')}
　利確水準:　{result.get('推奨利確水準', '?')}
　損切水準:　{result.get('推奨損切り水準', '?')}

--- 解説 ---
{result.get('理由', '?')}

--- 指標 ---
{indicator_text}
"""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=10)
    except requests.exceptions.Timeout:
        send_error_to_telegram("Telegram送信エラー: タイムアウト発生")
    except Exception as e:
        send_error_to_telegram(f"Telegram送信エラー:\n{str(e)}")


def run_analysis():
    top_tickers = get_top_symbols_by_24h_change()
    available = get_available_contract_symbols()
    top_tickers = [t for t in top_tickers if t["symbol"] in available]
    symbols = [t["symbol"] for t in top_tickers]

    now = datetime.utcnow()
    for ticker in top_tickers:
        symbol = ticker["symbol"]
        current_price = ticker["last_price"]

        last_time = NOTIFICATION_CACHE.get(symbol)
        if last_time and (now - last_time) < timedelta(hours=1):
            continue

        try:
            df_15m = fetch_ohlcv(symbol, interval='15m')
            if df_15m is None:
                continue
            df_daily = fetch_daily_ohlcv_max(symbol)
            if df_daily is None:
                continue

            ath_flag, ath_price = is_ath_today(current_price, df_15m, df_daily)
            if not ath_flag:
                continue

            result = analyze_with_groq(df_15m, symbol)
            send_to_telegram(symbol, result)
            NOTIFICATION_CACHE[symbol] = now
            time.sleep(1)
        except Exception:
            send_error_to_telegram(f"{symbol} 分析中にエラー:\n{traceback.format_exc()}")


@app.route("/")
def index():
    return "OK"


@app.route("/run_analysis", methods=["GET", "HEAD"])
def run_analysis_route():
    run_analysis()
    return "分析完了", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
