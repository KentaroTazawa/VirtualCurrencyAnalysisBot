import os
import time
import traceback
from datetime import datetime, timedelta
import requests
import pandas as pd
from flask import Flask
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

MEXC_BASE_URL = "https://contract.mexc.com"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
app = Flask(__name__)

# ====== 運用パラメータ（緩めにして機会を増やす） ======
TOP_SYMBOLS_LIMIT = 40            # 候補の母集団（24h上昇上位）
MAX_ALERTS_PER_RUN = 5            # 1回の実行で通知する最大件数（増やす）
COOLDOWN_HOURS = 1.0              # 同一銘柄のクールダウン（短縮）
USE_GROQ_COMMENTARY = False       # TrueでGroq簡易解説を付与
GROQ_MODEL = "llama3-70b-8192"

# ====== シグナル・しきい値（過熱検出へ全面移行） ======
MIN_24H_CHANGE_PCT = 10.0         # 候補最低24h変化率（やや緩め）
RSI_OB_5M = 72.0                  # 5分RSIがこの値超えで過熱
RSI_OB_15M = 70.0                 # 15分RSI過熱
BB_PERIOD = 20
BB_K = 2.0
BB_UPPER_BREAK_PCT = 0.002        # 上限バンド超えの許容超過率(0.2%)
EMA_DEV_PERIOD = 50               # 50EMAからの乖離
EMA_DEV_MIN_PCT = 7.5             # 乖離が+7.5%以上
VOL_SPIKE_LOOKBACK = 20
VOL_SPIKE_MULT = 2.5               # 出来高が過去20本平均の2.5倍
IMPULSE_PCT_5M = 0.04             # 直近急騰の最低合計上昇率(4%)
CONSEC_GREEN_1H = 3               # 1h連続陽線本数

# スコア
SCORE_THRESHOLD = 4               # 通知に必要な合計スコア（緩め）

# 利確・損切り（固定R管理）
ATR_PERIOD = 14
SL_ATR_MULT = 0.5                 # スイング高値 + 0.5*ATR
TP1_R = 1.0
TP2_R = 2.0

NOTIFICATION_CACHE = {}  # {symbol: last_notified_timestamp}

# ========= ユーティリティ =========

def mexc_get(path: str, timeout=10):
    url = f"{MEXC_BASE_URL}{path}"
    res = requests.get(url, timeout=timeout)
    res.raise_for_status()
    return res.json()


def send_error_to_telegram(error_message: str):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(
            url,
            data={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": f"⚠️ エラー発生:\n\n{error_message[:3800]}",
            },
            timeout=10,
        )
    except:
        pass


def tg_send_md(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text[:4096],
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }
    try:
        requests.post(url, data=payload, timeout=10)
    except requests.exceptions.Timeout:
        send_error_to_telegram("Telegram送信エラー: タイムアウト発生")
    except Exception as e:
        send_error_to_telegram(f"Telegram送信エラー:\n{str(e)}")


# ========= データ取得 =========

def get_top_symbols_by_24h_change(limit=TOP_SYMBOLS_LIMIT):
    try:
        data = mexc_get("/api/v1/contract/ticker")
        tickers = data.get("data", [])
        filtered = []
        for t in tickers:
            try:
                symbol = t.get("symbol", "")
                last_price = float(t.get("lastPrice", 0))
                change_pct = float(t.get("riseFallRate", 0)) * 100
                if change_pct >= MIN_24H_CHANGE_PCT and symbol.endswith("_USDT"):
                    filtered.append({"symbol": symbol, "last_price": last_price, "change_pct": change_pct})
            except:
                continue
        filtered.sort(key=lambda x: x["change_pct"], reverse=True)
        return filtered[:limit]
    except Exception as e:
        send_error_to_telegram(f"MEXC 急上昇銘柄取得エラー:\n{str(e)}")
        return []


def get_available_contract_symbols():
    try:
        data = mexc_get("/api/v1/contract/detail")
        arr = data.get("data", []) or []
        return {it.get("symbol") for it in arr if it.get("symbol")}
    except Exception as e:
        send_error_to_telegram(f"先物銘柄一覧取得失敗:\n{str(e)}")
        return set()


def fetch_ohlcv(symbol, interval='15m', max_retries=3, timeout_sec=15):
    imap = {
        '1m': 'Min1', '5m': 'Min5', '15m': 'Min15', '30m': 'Min30',
        '60m': 'Min60', '4h': 'Hour4', '8h': 'Hour8', '1d': 'Day1',
        '1w': 'Week1', '1M': 'Month1'
    }
    interval_param = imap.get(interval, 'Min15')
    url = f"/api/v1/contract/kline/{symbol}?interval={interval_param}"

    for attempt in range(1, max_retries + 1):
        try:
            data = mexc_get(url, timeout=timeout_sec)
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
                rows.append({
                    "ts": int(times[i]),
                    "open": float(open_arr[i]) if i < len(open_arr) and open_arr[i] is not None else None,
                    "high": float(high_arr[i]) if i < len(high_arr) and high_arr[i] is not None else None,
                    "low": float(low_arr[i]) if i < len(low_arr) and low_arr[i] is not None else None,
                    "close": float(close_arr[i]) if i < len(close_arr) and close_arr[i] is not None else None,
                    "vol": float(vol_arr[i]) if i < len(vol_arr) and vol_arr[i] is not None else None,
                })
            df = pd.DataFrame(rows).dropna()
            df = df.sort_values("ts").reset_index(drop=True)
            return df
        except Exception as e:
            if attempt == max_retries:
                send_error_to_telegram(f"{symbol} の{interval}ローソク取得失敗:\n{str(e)}")
            time.sleep(1)
    return None


# ========= 指標 =========

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(period, min_periods=1).mean()
    avg_loss = loss.rolling(period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]; low = df["low"]; close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def bollinger_bands(series: pd.Series, period: int = 20, k: float = 2.0):
    ma = series.rolling(period, min_periods=1).mean()
    std = series.rolling(period, min_periods=1).std(ddof=0)
    upper = ma + k * std
    lower = ma - k * std
    return ma, upper, lower


def upper_wick_ratio(row) -> float:
    rng = row["high"] - row["low"]
    if rng <= 0:
        return 0.0
    return (row["high"] - max(row["open"], row["close"])) / rng


def volume_spike(vol_series: pd.Series, lookback: int, mult: float) -> bool:
    if len(vol_series) < lookback + 1:
        return False
    ma = vol_series.rolling(lookback, min_periods=1).mean()
    return vol_series.iloc[-1] >= ma.iloc[-1] * mult


def recent_impulse(df: pd.DataFrame, bars=6, pct=0.05) -> bool:
    if len(df) < bars + 1:
        return False
    c0 = df["close"].iloc[-bars-1]
    c1 = df["close"].iloc[-1]
    return (c1 / c0 - 1.0) >= pct


def break_of_structure_short(df_5m: pd.DataFrame) -> bool:
    if len(df_5m) < 10:
        return False
    lows = df_5m["low"]; closes = df_5m["close"]
    recent_low = lows.iloc[-4:-1].min()
    return closes.iloc[-1] < recent_low


def count_consecutive_green(df: pd.DataFrame) -> int:
    # 陽線を1、陰線を0
    body = (df["close"] - df["open"]) > 0
    cnt = 0
    for val in body.iloc[::-1]:
        if val:
            cnt += 1
        else:
            break
    return cnt


# ========= スコアリング（過熱ショート特化） =========

def score_short_setup(symbol: str, df_5m: pd.DataFrame, df_15m: pd.DataFrame, df_60m: pd.DataFrame):
    score = 0
    notes = []

    # 直近急騰（5m）
    if recent_impulse(df_5m, bars=6, pct=IMPULSE_PCT_5M):
        score += 1; notes.append("5m直近急騰")

    # RSI過熱
    rsi5 = rsi(df_5m["close"], 14).iloc[-1]
    rsi15 = rsi(df_15m["close"], 14).iloc[-1]
    if rsi5 >= RSI_OB_5M:
        score += 2; notes.append(f"RSI5m過熱({rsi5:.1f})")
    if rsi15 >= RSI_OB_15M:
        score += 2; notes.append(f"RSI15m過熱({rsi15:.1f})")

    # ボリンジャー上限ブレイク
    _, upper5, _ = bollinger_bands(df_5m["close"], BB_PERIOD, BB_K)
    if df_5m["close"].iloc[-1] > upper5.iloc[-1] * (1.0 + BB_UPPER_BREAK_PCT):
        score += 2; notes.append("BB上限オーバー")

    # EMA50からの正乖離
    ema50_5 = ema(df_5m["close"], EMA_DEV_PERIOD)
    dev_pct = (df_5m["close"].iloc[-1] / ema50_5.iloc[-1] - 1.0) * 100.0
    if dev_pct >= EMA_DEV_MIN_PCT:
        score += 2; notes.append(f"+{dev_pct:.1f}% 50EMA乖離")

    # 出来高スパイク
    if volume_spike(df_5m["vol"], VOL_SPIKE_LOOKBACK, VOL_SPIKE_MULT):
        score += 2; notes.append("出来高スパイク")

    # 1h連続陽線
    if count_consecutive_green(df_60m) >= CONSEC_GREEN_1H:
        score += 1; notes.append(f"1h連続陽線≥{CONSEC_GREEN_1H}")

    # 反転の芽（BOS）
    if break_of_structure_short(df_5m):
        score += 2; notes.append("5m BOS下抜け")

    return score, notes


# ========= 取引計画 =========

def plan_short_trade(df_5m: pd.DataFrame):
    close = df_5m["close"]
    high = df_5m["high"]

    swing_high = high.iloc[-5:-1].max()
    entry = close.iloc[-1]  # 現値成行
    atr_val = atr(df_5m, ATR_PERIOD).iloc[-1]
    sl = swing_high + SL_ATR_MULT * atr_val
    risk = abs(sl - entry)
    if risk <= 0:
        sl = swing_high + 1.0 * atr_val
        risk = abs(sl - entry)
    tp1 = entry - TP1_R * risk
    tp2 = entry - TP2_R * risk
    r_multiple = (entry - tp2) / risk if risk > 0 else 0
    return {
        "entry": round(entry, 6),
        "sl": round(sl, 6),
        "tp1": round(tp1, 6),
        "tp2": round(tp2, 6),
        "atr": round(atr_val, 6),
        "risk_per_unit": round(risk, 6),
        "r_multiple_to_tp2": round(r_multiple, 2),
    }


# ========= Groq（任意の短文解説） =========

def groq_commentary(symbol: str, notes: list, plan: dict) -> str:
    if not (USE_GROQ_COMMENTARY and client):
        return ""
    try:
        now_jst = (datetime.utcnow() + timedelta(hours=9)).strftime("%Y-%m-%d %H:%M")
        brief = (
            f"{symbol} のショート候補。根拠: {', '.join(notes)}。\n"
            f"想定: エントリ {plan['entry']}, SL {plan['sl']}, TP1 {plan['tp1']}, TP2 {plan['tp2']}。\n"
            f"{now_jst} JST"
        )
        res = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": f"日本語で、以下を80〜140文字で簡潔に要約して: {brief}"}],
            temperature=0.2,
            max_tokens=140,
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        send_error_to_telegram(f"Groqエラー: {str(e)}")
        return ""


# ========= 通知 =========

def send_short_signal(symbol: str, current_price: float, score: int, notes: list, plan: dict, change_pct: float, indicators: dict, comment: str):
    display_symbol = symbol.replace("_USDT", "")
    ind_text = "\n".join([f"- {k}: {v}" for k, v in indicators.items()]) if indicators else ""
    notes_text = ", ".join(notes)

    text = f"""*📉 ショート候補: {display_symbol}*
24h変化率: {change_pct:.2f}%  / 現値: {current_price}

*スコア:* {score} / 必要 {SCORE_THRESHOLD}
*根拠:* {notes_text}

*計画*
- Entry: `{plan['entry']}`
- SL: `{plan['sl']}`  (risk/qty: `{plan['risk_per_unit']}`)
- TP1: `{plan['tp1']}` ({TP1_R}R)
- TP2: `{plan['tp2']}` ({TP2_R}R, 到達R: {plan['r_multiple_to_tp2']})

*参考指標*
{ind_text}

{comment}
"""
    tg_send_md(text)


# ========= メイン =========

def run_analysis():
    top_tickers = get_top_symbols_by_24h_change()
    available = get_available_contract_symbols()
    top_tickers = [t for t in top_tickers if t["symbol"] in available]

    # クールダウン
    now = datetime.utcnow()
    cooled = []
    for t in top_tickers:
        last_time = NOTIFICATION_CACHE.get(t["symbol"])
        if last_time and (now - last_time) < timedelta(hours=COOLDOWN_HOURS):
            continue
        cooled.append(t)

    # スコアリング
    scored = []
    for t in cooled:
        symbol = t["symbol"]
        current_price = t["last_price"]

        try:
            df_5m = fetch_ohlcv(symbol, interval='5m')
            df_15m = fetch_ohlcv(symbol, interval='15m')
            df_60m = fetch_ohlcv(symbol, interval='60m')
            if any(x is None or x.empty for x in [df_5m, df_15m, df_60m]):
                continue

            # 前提：直近の衝動が弱くても拾えるよう、条件を緩めに
            # （過熱系シグナルの合意で拾う）
            score, notes = score_short_setup(symbol, df_5m, df_15m, df_60m)

            if score >= SCORE_THRESHOLD and break_of_structure_short(df_5m):
                plan = plan_short_trade(df_5m)

                indicators = {
                    "RSI(5m)": round(rsi(df_5m["close"], 14).iloc[-1], 2),
                    "RSI(15m)": round(rsi(df_15m["close"], 14).iloc[-1], 2),
                    "+乖離(5m,EMA50)": round((df_5m["close"].iloc[-1] / ema(df_5m["close"], EMA_DEV_PERIOD).iloc[-1] - 1) * 100, 2),
                    "ATR(5m)": round(atr(df_5m, ATR_PERIOD).iloc[-1], 6),
                    "出来高(5m)最新/平均": round(df_5m["vol"].iloc[-1] / max(1e-9, df_5m["vol"].rolling(VOL_SPIKE_LOOKBACK, min_periods=1).mean().iloc[-1]), 2),
                }

                comment = groq_commentary(symbol, notes, plan) if USE_GROQ_COMMENTARY else ""

                scored.append({
                    "symbol": symbol,
                    "score": score,
                    "notes": notes,
                    "plan": plan,
                    "current_price": current_price,
                    "change_pct": t["change_pct"],
                    "indicators": indicators,
                })
        except Exception:
            send_error_to_telegram(f"{symbol} 分析中にエラー:\n{traceback.format_exc()}")

    # スコア順に上位のみ通知
    scored.sort(key=lambda x: (x["score"], x["change_pct"]), reverse=True)
    alerts_sent = 0
    for s in scored[:MAX_ALERTS_PER_RUN]:
        send_short_signal(
            s["symbol"], s["current_price"], s["score"], s["notes"], s["plan"], s["change_pct"], s["indicators"],
            comment=groq_commentary(s["symbol"], s["notes"], s["plan"]) if USE_GROQ_COMMENTARY else "",
        )
        NOTIFICATION_CACHE[s["symbol"]] = now
        alerts_sent += 1
        time.sleep(1)

    #if alerts_sent == 0:
        #tg_send_md("_今回は条件を満たすショート候補はありませんでした。_")


@app.route("/")
def index():
    return "OK"


@app.route("/run_analysis", methods=["GET", "HEAD"])
def run_analysis_route():
    run_analysis()
    return "分析完了", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
