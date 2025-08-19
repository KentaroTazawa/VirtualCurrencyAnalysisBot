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
import math

load_dotenv()

MEXC_BASE_URL = "https://contract.mexc.com"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
app = Flask(__name__)

# ====== 運用パラメータ ======
TOP_SYMBOLS_LIMIT = 30           # 候補の母集団（24h上昇上位）
MAX_ALERTS_PER_RUN = 3           # 1回の実行で通知する最大件数
COOLDOWN_HOURS = 2               # 同一銘柄のクールダウン
USE_GROQ_COMMENTARY = False      # TrueでGroq簡易解説を付与
GROQ_MODEL = "llama3-70b-8192"

# シグナル・しきい値
MIN_24H_CHANGE_PCT = 8.0         # 候補最低24h変化率
WICK_RATIO_MIN = 0.35            # 上ヒゲ比率(upper_wick / total_range)の最低値
VOLUME_CLIMAX_MULT = 2.0         # 出来高クライマックス(過去20本平均の何倍)
RSI_PERIOD = 14
ATR_PERIOD = 14
SCORE_THRESHOLD = 6              # 通知に必要な合計スコア
ATH_SWIPE_TOL = 0.997            # ATH*0.997以上で「ほぼ到達スイープ」扱い
LOOKBACK_DAYS_FOR_SWEEP = 7      # 直近n日高値スイープも対象

NOTIFICATION_CACHE = {}  # {symbol: last_notified_timestamp}


# ========= ユーティリティ =========
def send_error_to_telegram(error_message: str):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(
            url,
            data={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": f"⚠️ エラー発生:\n\n{error_message[:3800]}",
            },
            timeout=10
        )
    except:
        pass


def tg_send_md(text: str):
    """Telegram Markdown: 4096文字ケア"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text[:4096],
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, data=payload, timeout=10)
    except requests.exceptions.Timeout:
        send_error_to_telegram("Telegram送信エラー: タイムアウト発生")
    except Exception as e:
        send_error_to_telegram(f"Telegram送信エラー:\n{str(e)}")


def mexc_get(path: str, timeout=10):
    url = f"{MEXC_BASE_URL}{path}"
    res = requests.get(url, timeout=timeout)
    res.raise_for_status()
    return res.json()


def get_top_symbols_by_24h_change(limit=TOP_SYMBOLS_LIMIT):
    # 24h上昇上位を抽出
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

def upper_wick_ratio(row) -> float:
    rng = row["high"] - row["low"]
    if rng <= 0: return 0.0
    return (row["high"] - max(row["open"], row["close"])) / rng

def volume_climax(vol_series: pd.Series, lookback: int = 20, mult: float = 2.0) -> bool:
    if len(vol_series) < lookback + 1: return False
    ma = vol_series.rolling(lookback, min_periods=1).mean()
    return vol_series.iloc[-1] >= ma.iloc[-1] * mult

def bearish_divergence(close: pd.Series, rsi_series: pd.Series, lookback=30) -> bool:
    if len(close) < lookback + 5: return False
    c = close.iloc[-lookback:]
    r = rsi_series.iloc[-lookback:]
    # 直近2つのスイング高値／RSI高値を雑に検出
    h1_idx = c.idxmax()
    c2 = c[c.index < h1_idx]
    if c2.empty: return False
    h2_idx = c2.idxmax()
    # 価格は高値更新、RSIは更新できず
    return (close.loc[h1_idx] > close.loc[h2_idx]) and (r.loc[h1_idx] <= r.loc[h2_idx])

def recent_impulse(df: pd.DataFrame, bars=6, pct=0.06) -> bool:
    """直近barsで終値が合計+6%以上など"""
    if len(df) < bars + 1: return False
    c0 = df["close"].iloc[-bars-1]
    c1 = df["close"].iloc[-1]
    return (c1 / c0 - 1.0) >= pct

def day_high_within(df_day: pd.DataFrame, days: int) -> float:
    if df_day is None or df_day.empty: return None
    return df_day["high"].tail(days).max()

def is_ath_or_recent_sweep(current_price: float, df_15m: pd.DataFrame, df_daily: pd.DataFrame):
    """ATHか直近n日高値を“ほぼ”上抜いた（スイープ）か"""
    try:
        ath = max(df_15m["high"].max(), df_daily["high"].max())
        recent_high = day_high_within(df_daily, LOOKBACK_DAYS_FOR_SWEEP)
        threshold = ath * ATH_SWIPE_TOL
        cond_ath = current_price >= threshold
        cond_recent = (recent_high is not None) and (current_price >= recent_high * ATH_SWIPE_TOL)
        return (cond_ath or cond_recent), ath, recent_high
    except Exception:
        return False, None, None

def break_of_structure_short(df_5m: pd.DataFrame) -> bool:
    """直近の押し安値割れ(BOS)を簡易判定."""
    if len(df_5m) < 10: return False
    highs = df_5m["high"]; lows = df_5m["low"]; closes = df_5m["close"]
    # 直近のスイング: 直前まで高値更新が続いた後、安値を下抜け
    recent_low = lows.iloc[-4:-1].min()
    return closes.iloc[-1] < recent_low

def score_short_setup(df_5m: pd.DataFrame, df_15m: pd.DataFrame, df_day: pd.DataFrame):
    """ショート向きセットアップをスコア化"""
    score = 0
    notes = []

    # ボラ拡大の衝動
    if recent_impulse(df_5m, bars=6, pct=0.05):
        score += 1; notes.append("直近急騰")

    # 上ヒゲ判定（直近足）
    last = df_5m.iloc[-1]
    uw = upper_wick_ratio(last)
    if uw >= WICK_RATIO_MIN and last["close"] < last["open"]:
        score += 2; notes.append("上ヒゲ陰線")

    # 出来高クライマックス
    if volume_climax(df_5m["vol"], lookback=20, mult=VOLUME_CLIMAX_MULT):
        score += 2; notes.append("出来高クライマックス")

    # RSIダイバージェンス
    rsi5 = rsi(df_5m["close"], RSI_PERIOD)
    if bearish_divergence(df_5m["close"], rsi5, lookback=30):
        score += 2; notes.append("RSIベアダイバージェンス")

    # BOS
    if break_of_structure_short(df_5m):
        score += 3; notes.append("BOS下抜け")

    # 15分で乖離縮小（過熱後の減速感）
    ema_fast = ema(df_15m["close"], 8)
    ema_slow = ema(df_15m["close"], 21)
    if ema_fast.iloc[-1] - ema_slow.iloc[-1] < (ema_fast.iloc[-5] - ema_slow.iloc[-5]):
        score += 1; notes.append("15m乖離縮小")

    # ATH/直近高値スイープ気味か
    current_price = df_5m["close"].iloc[-1]
    swept, ath, recent_high = is_ath_or_recent_sweep(current_price, df_15m, df_day)
    if swept:
        score += 1; notes.append("ATH/直近高値スイープ")

    return score, notes

def plan_short_trade(df_5m: pd.DataFrame):
    """エントリー/SL/TP計算（BOS後リターンムーブ想定）"""
    close = df_5m["close"]
    high = df_5m["high"]
    low = df_5m["low"]

    swing_high = high.iloc[-5:-1].max()
    entry = close.iloc[-1]  # 簡易に現在値成行
    atr_val = atr(df_5m, ATR_PERIOD).iloc[-1]
    sl = swing_high + 0.5 * atr_val
    risk = abs(sl - entry)
    if risk <= 0:
        sl = swing_high + atr_val
        risk = abs(sl - entry)
    tp1 = entry - 1.0 * risk
    tp2 = entry - 2.0 * risk
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
            messages=[{"role": "user", "content": f"日本語で、以下を60〜120文字で簡潔に要約して: {brief}"}],
            temperature=0.2,
            max_tokens=120,
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
- TP1: `{plan['tp1']}` (1R)
- TP2: `{plan['tp2']}` (2R, 到達R: {plan['r_multiple_to_tp2']})

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
            df_day = fetch_ohlcv(symbol, interval='1d')
            if any(x is None or x.empty for x in [df_5m, df_15m, df_day]):
                continue

            # 前提：直近の衝動＆スイープ気味
            swept, _, _ = is_ath_or_recent_sweep(current_price, df_15m, df_day)
            if not (recent_impulse(df_5m, bars=6, pct=0.05) and swept):
                continue

            score, notes = score_short_setup(df_5m, df_15m, df_day)
            if score >= SCORE_THRESHOLD and break_of_structure_short(df_5m):
                # 計画計算
                plan = plan_short_trade(df_5m)

                # 指標の一部も表示（モメンタム/ボラの把握用）
                indicators = {
                    "RSI(5m)": round(rsi(df_5m["close"], RSI_PERIOD).iloc[-1], 2),
                    "ATR(5m)": round(atr(df_5m, ATR_PERIOD).iloc[-1], 6),
                    "上ヒゲ比率(直近)": round(upper_wick_ratio(df_5m.iloc[-1]), 2),
                }

                # 追加の簡易コメント（任意）
                comment = groq_commentary(symbol, notes, plan) if USE_GROQ_COMMENTARY else ""

                scored.append({
                    "symbol": symbol,
                    "score": score,
                    "notes": notes,
                    "plan": plan,
                    "current_price": current_price,
                    "change_pct": t["change_pct"],
                    "indicators": indicators
                })
        except Exception:
            send_error_to_telegram(f"{symbol} 分析中にエラー:\n{traceback.format_exc()}")

    # スコア順に上位のみ通知
    scored.sort(key=lambda x: (x["score"], x["change_pct"]), reverse=True)
    alerts_sent = 0
    for s in scored[:MAX_ALERTS_PER_RUN]:
        send_short_signal(
            s["symbol"], s["current_price"], s["score"], s["notes"], s["plan"], s["change_pct"], s["indicators"],
            comment=groq_commentary(s["symbol"], s["notes"], s["plan"]) if USE_GROQ_COMMENTARY else ""
        )
        NOTIFICATION_CACHE[s["symbol"]] = now
        alerts_sent += 1
        time.sleep(1)

    if alerts_sent == 0:
        tg_send_md("_今回は条件を満たすショート候補はありませんでした。_")


@app.route("/")
def index():
    return "OK"


@app.route("/run_analysis", methods=["GET", "HEAD"])
def run_analysis_route():
    run_analysis()
    return "分析完了", 200


if __name__ == "__main__":
    # Render等でPORTが与えられる前提
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
