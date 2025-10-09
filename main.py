# main.py (修正版: Groq JSON応答 + confidence利用 + 最小変更で緩和/デバッグ切替を追加)
import os
import time
import traceback
import logging
import sys
import json
import re
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
TOP_SYMBOLS_LIMIT = 20  # 候補の母集団（24h上昇上位）
MAX_ALERTS_PER_RUN = 5  # 1回の実行で通知する最大件数（増やす）
COOLDOWN_HOURS = 1.0  # 同一銘柄のクールダウン（短縮）
USE_GROQ_COMMENTARY = False  # TrueでGroq簡易解説を付与
GROQ_MODEL = "llama-3.1-8b-instant"

# ====== シグナル・しきい値（過熱検出へ全面移行） ======
MIN_24H_CHANGE_PCT = 10.0  # 候補最低24h変化率（やや緩め）
RSI_OB_5M = 72.0
RSI_OB_15M = 70.0
BB_PERIOD = 20
BB_K = 2.0
BB_UPPER_BREAK_PCT = 0.002
EMA_DEV_PERIOD = 50
EMA_DEV_MIN_PCT = 7.5
VOL_SPIKE_LOOKBACK = 20
VOL_SPIKE_MULT = 2.5
IMPULSE_PCT_5M = 0.04
CONSEC_GREEN_1H = 3

# ====== 通知条件用パラメータ ======
SCORE_THRESHOLD = 5
TP1_THRESHOLD = -5

# quick debug / operational switches via ENV
RELAX_NOTIFICATION_RULES = os.getenv("RELAX_NOTIFICATION_RULES", "0") == "1"
# If set, do not require BOS to send alerts (useful for debugging / 段階的導入)
DISABLE_TP1_CHECK = os.getenv("DISABLE_TP1_CHECK", "0") == "1"
OVERRIDE_SCORE_THRESHOLD = int(os.getenv("OVERRIDE_SCORE_THRESHOLD", "0") or 0)

if OVERRIDE_SCORE_THRESHOLD > 0:
    SCORE_THRESHOLD = OVERRIDE_SCORE_THRESHOLD

ATR_PERIOD = 14
SL_ATR_MULT = 0.5
TP1_R = 1.0
TP2_R = 2.0

NOTIFICATION_CACHE = {}  # {symbol: last_notified_timestamp}

# ========= BOS 関連パラメタ（新規、上書きは環境で後で行ってください） ========
BOS_RECENT_GAIN_THRESHOLD = 0.02  # 2% に緩める（元は0.03）
BOS_RSI_MAX = 65
BOS_REQUIRE_VOLUME = True
BOS_VOL_MULT = 1.8

# ========= AI confidence 統合ルール用閾値 ========
CONF_FOR_ANY_SCORE = 0.75
CONF_FOR_LOW_SCORE = 0.90

# ========= ロガー =========
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(stream=sys.stdout, level=getattr(logging, LOG_LEVEL, logging.INFO),
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("VirtualCurrencyAnalysisBot")

# Check Telegram envs early
TELEGRAM_ENABLED = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)
if not TELEGRAM_ENABLED:
    logger.warning("Telegram is not fully configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID). Notifications will not be sent until these are set.")

# ========= ユーティリティ =========
def mexc_get(path: str, timeout=10):
    url = f"{MEXC_BASE_URL}{path}"
    try:
        logger.debug(f"HTTP GET: {url}")
        res = requests.get(url, timeout=timeout)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        logger.error(f"mexc_get error for {url}: {e}")
        raise

def send_error_to_telegram(error_message: str):
    logger.error(error_message)
    if not TELEGRAM_ENABLED:
        logger.warning("send_error_to_telegram: TELEGRAM not configured, skipping.")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        res = requests.post(
            url,
            data={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": f"⚠️ エラー発生:\n\n{error_message[:3800]}",
            },
            timeout=10,
        )
        if res.status_code != 200:
            logger.error(f"send_error_to_telegram: Telegram API returned {res.status_code}: {res.text}")
    except Exception as e:
        logger.error(f"Failed to send error to Telegram: {e}")

def tg_send_md(text: str):
    if not TELEGRAM_ENABLED:
        logger.warning("tg_send_md: TELEGRAM not configured, skipping message.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text[:4096],
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }
    try:
        res = requests.post(url, data=payload, timeout=10)
        if res.status_code != 200:
            send_error_to_telegram(f"Telegram送信失敗: status={res.status_code} body={res.text}")
    except requests.exceptions.Timeout:
        send_error_to_telegram("Telegram送信エラー: タイムアウト発生")
    except Exception as e:
        send_error_to_telegram(f"Telegram送信エラー:\n{str(e)}")

# ========= データ取得 =========
def get_top_symbols_by_24h_change(limit=TOP_SYMBOLS_LIMIT):
    try:
        data = mexc_get("/api/v1/contract/ticker")
        tickers = data.get("data", [])
        logger.info(f"Fetched {len(tickers)} tickers from /ticker")
        filtered = []
        for t in tickers:
            try:
                symbol = t.get("symbol", "")
                last_price = float(t.get("lastPrice", 0))
                change_pct = float(t.get("riseFallRate", 0)) * 100
                if change_pct >= MIN_24H_CHANGE_PCT and symbol.endswith("_USDT"):
                    filtered.append({"symbol": symbol, "last_price": last_price, "change_pct": change_pct})
            except Exception:
                continue
        filtered.sort(key=lambda x: x["change_pct"], reverse=True)
        logger.info(f"{len(filtered)} symbols passed 24h change filter (>{MIN_24H_CHANGE_PCT}%)")
        return filtered[:limit]
    except Exception as e:
        send_error_to_telegram(f"MEXC 急上昇銘柄取得エラー:\n{str(e)}")
        return []

def get_available_contract_symbols():
    try:
        data = mexc_get("/api/v1/contract/detail")
        arr = data.get("data", []) or []
        symbols = {it.get("symbol") for it in arr if it.get("symbol")}
        logger.info(f"Fetched {len(symbols)} available contract symbols")
        return symbols
    except Exception as e:
        send_error_to_telegram(f"先物銘柄一覧取得失敗:\n{str(e)}")
        return set()

def fetch_ohlcv(symbol, interval='15m', max_retries=3, timeout_sec=15):
    imap = {
        '1m': 'Min1', '5m': 'Min5', '15m': 'Min15', '30m': 'Min30',
        '60m': 'Min60', '4h': 'Hour4', '8h': 'Hour8', '1d': 'Day1', '1w': 'Week1', '1M': 'Month1'
    }
    interval_param = imap.get(interval, 'Min15')
    url = f"/api/v1/contract/kline/{symbol}?interval={interval_param}"
    for attempt in range(1, max_retries + 1):
        try:
            logger.debug(f"Fetching kline for {symbol} interval {interval} (attempt {attempt})")
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
            logger.debug(f"Fetched {len(df)} rows for {symbol} {interval}")
            return df
        except Exception as e:
            logger.warning(f"[{symbol}] {interval} fetch attempt {attempt} failed: {e}")
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
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
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
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()

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

# ========= BOS 判定（非AI） =========
def break_of_structure_short(df_5m: pd.DataFrame) -> bool:
    # 少し緩めの閾値（BOS_RECENT_GAIN_THRESHOLD）を使うように変更（最小限の修正）
    recent_n = 3
    prev_n = 6
    min_bars = recent_n + prev_n + 3
    if len(df_5m) < min_bars:
        return False
    c0 = df_5m["close"].iloc[-(recent_n + prev_n + 1)]
    c1 = df_5m["close"].iloc[-(recent_n + 1)]
    recent_gain = (c1 / c0 - 1.0)
    if recent_gain < BOS_RECENT_GAIN_THRESHOLD:  # 0.02 に緩和
        return False
    lows = df_5m["low"]; closes = df_5m["close"]
    recent_low = lows.iloc[-(recent_n + 1):-1].min()
    prev_low = lows.iloc[-(recent_n + prev_n + 1):-(recent_n + 1)].min()
    bos_triggered = (recent_low < prev_low) and (closes.iloc[-1] < recent_low)
    if not bos_triggered:
        return False
    rsi_series = rsi(df_5m["close"], 14)
    if len(rsi_series) < 1 or rsi_series.iloc[-1] >= BOS_RSI_MAX:
        return False
    # 出来高確認（任意）
    if BOS_REQUIRE_VOLUME:
        vol_ratio = df_5m["vol"].iloc[-1] / max(1e-9, df_5m["vol"].rolling(20).mean().iloc[-1])
        if vol_ratio < BOS_VOL_MULT:
            return False
    return True

# ========= Groq 応答の JSON パーサ（新規） =========
def parse_groq_json_response(raw_text: str):
    """
    raw_text から最初の JSON オブジェクトを抽出して parse -> dict を返す。
    期待するキー: decision (YES/NO), confidence (0-1), reason (str)
    戻り値: (decision_bool, confidence_float, reason_str)
    """
    try:
        m = re.search(r'\{.*\}', raw_text, re.S)
        if not m:
            # 直接 YES/NO 単語で返している場合のフォールバック
            txt = raw_text.strip().upper()
            if "YES" in txt and "NO" not in txt:
                return True, 0.5, "raw_yes_fallback"
            if "NO" in txt and "YES" not in txt:
                return False, 0.5, "raw_no_fallback"
            return False, 0.0, "ambiguous_no_json"
        obj = json.loads(m.group(0))
        decision = obj.get("decision", "")
        conf = float(obj.get("confidence", 0.0))
        reason = str(obj.get("reason", "") or "")[:200]
        decision_bool = str(decision).strip().upper() == "YES"
        conf = max(0.0, min(1.0, conf))
        return decision_bool, conf, reason
    except Exception as e:
        logger.warning(f"parse_groq_json_response failed: {e} -- raw:{raw_text[:200]}")
        return False, 0.0, "parse_error"

# ========= BOS 判定（AI） - 改良版: JSON 応答 + confidence を返す（変更） =========
def break_of_structure_short_ai(symbol: str, df_5m: pd.DataFrame):
    """
    戻り値: (decision_bool, confidence_float, reason_str)
    - decision_bool: Groq が YES と判断したか
    - confidence_float: 0.0-1.0 の推定確信度（モデルに依存、プロンプト上で数値を出すよう要請）
    - reason_str: 短文理由またはフォールバック文字列
    """
    # まずは非AI判定が True ならそのまま True, high confidence で返す
    if break_of_structure_short(df_5m):
        return True, 0.99, "rule_based_bos"
    if not client:
        return False, 0.0, "groq_not_configured"
    try:
        # 特徴量を短くまとめる（過度に長文にしない）
        rsi_series = rsi(df_5m["close"], 14)
        rsi_val = float(rsi_series.iloc[-1])
        highs, lows, closes = df_5m["high"], df_5m["low"], df_5m["close"]
        # recent_gain を直近8本 / 直近20本などで柔軟に
        if len(closes) >= 20:
            recent_gain = (closes.iloc[-4] / closes.iloc[-10] - 1.0) * 100
        else:
            recent_gain = (closes.iloc[-4] / closes.iloc[0] - 1.0) * 100
        dev_pct = (closes.iloc[-1] / ema(df_5m["close"], EMA_DEV_PERIOD).iloc[-1] - 1.0) * 100
        vol_mean = df_5m["vol"].rolling(20, min_periods=1).mean().iloc[-1]
        vol_ratio = (df_5m["vol"].iloc[-1] / max(1e-9, vol_mean)) if vol_mean > 0 else 0.0
        recent_closes = df_5m["close"].iloc[-8:].tolist() if len(df_5m) >= 8 else df_5m["close"].tolist()

        payload = {
            "symbol": symbol,
            "rsi14": round(rsi_val, 2),
            "ema50_dev_pct": round(dev_pct, 2),
            "vol_ratio": round(vol_ratio, 2),
            "last_close": round(float(closes.iloc[-1]), 8),
            "recent_closes": [round(float(x), 8) for x in recent_closes],
        }

        # プロンプトで JSON だけを返すよう強く指示（最小限の説明を外さない）
        prompt = (
            "You are a concise quantitative trading assistant.\n"
            "Input (JSON): " + json.dumps(payload) + ".\n"
            "Answer ONLY with a JSON object containing keys:\n"
            '  - "decision": "YES" or "NO"\n'
            '  - "confidence": a number between 0.0 and 1.0\n'
            '  - "reason": short (<=60 chars) explanation\n'
            "Do NOT include any other text outside the JSON."
        )

        res = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=120,
        )
        raw = res.choices[0].message.content
        logger.info(f"[{symbol}] Groq raw: {raw[:300]}")
        # ログファイルにも保存（検証用）
        try:
            with open("ai_responses.log", "a", encoding="utf-8") as f:
                f.write(f"{datetime.utcnow().isoformat()} | {symbol} | {raw.replace(chr(10),' ')}\n")
        except Exception:
            logger.debug("Failed to append to ai_responses.log (non-fatal)")

        decision_bool, conf, reason = parse_groq_json_response(raw)
        return decision_bool, conf, reason
    except Exception as e:
        logger.warning(f"[{symbol}] BOS AI判定失敗: {e}")
        return False, 0.0, "exception"

def count_consecutive_green(df: pd.DataFrame) -> int:
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
    bos_decision = False
    ai_conf = 0.0
    ai_reason = ""
    if recent_impulse(df_5m, bars=6, pct=IMPULSE_PCT_5M):
        score += 1; notes.append("5m直近急騰")
    rsi5 = rsi(df_5m["close"], 14).iloc[-1]
    rsi15 = rsi(df_15m["close"], 14).iloc[-1]
    if rsi5 >= RSI_OB_5M:
        score += 2; notes.append(f"RSI5m過熱({rsi5:.1f})")
    if rsi15 >= RSI_OB_15M:
        score += 2; notes.append(f"RSI15m過熱({rsi15:.1f})")
    _, upper5, _ = bollinger_bands(df_5m["close"], BB_PERIOD, BB_K)
    if df_5m["close"].iloc[-1] > upper5.iloc[-1] * (1.0 + BB_UPPER_BREAK_PCT):
        score += 2; notes.append("BB上限オーバー")
    ema50_5 = ema(df_5m["close"], EMA_DEV_PERIOD)
    dev_pct = (df_5m["close"].iloc[-1] / ema50_5.iloc[-1] - 1.0) * 100.0
    if dev_pct >= EMA_DEV_MIN_PCT:
        score += 2; notes.append(f"+{dev_pct:.1f}% 50EMA乖離")
    if volume_spike(df_5m["vol"], VOL_SPIKE_LOOKBACK, VOL_SPIKE_MULT):
        score += 2; notes.append("出来高スパイク")
    if count_consecutive_green(df_60m) >= CONSEC_GREEN_1H:
        score += 1; notes.append(f"1h連続陽線≥{CONSEC_GREEN_1H}")

    # AI 判定をここでスコアに加える（挙動を保ちつつ confidence を利用）
    try:
        bos_decision, ai_conf, ai_reason = break_of_structure_short_ai(symbol, df_5m)
        # ログを残す
        logger.debug(f"{symbol} AI判定 -> decision={bos_decision}, conf={ai_conf:.2f}, reason={ai_reason}")
        # スコアへの貢献は、AIがYESかつ confidence がある程度高い場合に加点
        if bos_decision:
            # 高信頼なら +2、やや低めなら +1（安全側）
            if ai_conf >= CONF_FOR_ANY_SCORE:
                score += 2
                notes.append(f"5m BOS下抜け(Groq判定 conf={ai_conf:.2f})")
            elif ai_conf >= 0.5:
                score += 1
                notes.append(f"5m BOS示唆(Groq弱 conf={ai_conf:.2f})")
            else:
                notes.append(f"Groq低信頼({ai_conf:.2f}) 無視")
    except Exception as e:
        logger.warning(f"{symbol} AI判定で例外: {e}")

    logger.debug(f"{symbol} scoring -> score={score}, notes={notes}")
    return score, notes, bos_decision, ai_conf, ai_reason

# ========= 取引計画 =========
def plan_short_trade(df_5m: pd.DataFrame):
    close = df_5m["close"]
    high = df_5m["high"]
    swing_high = high.iloc[-5:-1].max()
    entry = close.iloc[-1]
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
    entry = plan['entry']
    sl = plan['sl']
    tp1 = plan['tp1']
    tp2 = plan['tp2']
    sl_pct = (sl - entry) / entry * 100
    tp1_pct = (tp1 - entry) / entry * 100
    tp2_pct = (tp2 - entry) / entry * 100
    web_link = f"https://www.mexc.com/futures/{symbol}"
    open_link_text = f"[Webで開く]({web_link})"
    text = f"""*▶️ トレード画面:* {open_link_text}
* ショート候補: {display_symbol}* 24h変化率: {change_pct:.2f}% / 現値: {current_price}
*スコア:* {score} / 必要 {SCORE_THRESHOLD}
*根拠:* {notes_text}
*計画 (%表記)*
- Entry: `{entry}`
- SL: `{sl_pct:+.2f}%` (risk/qty: `{plan['risk_per_unit']}`)
- TP1: `{tp1_pct:+.2f}%` ({TP1_R}R)
- TP2: `{tp2_pct:+.2f}%` ({TP2_R}R, 到達R: {plan['r_multiple_to_tp2']})
*参考指標*
{ind_text}
{comment}
"""
    tg_send_md(text)

# ========= メイン =========
def run_analysis():
    logger.info("=== run_analysis started ===")
    top_tickers = get_top_symbols_by_24h_change()
    available = get_available_contract_symbols()
    before_filter_count = len(top_tickers)
    top_tickers = [t for t in top_tickers if t["symbol"] in available]
    logger.info(f"Top tickers: {before_filter_count} -> {len(top_tickers)} after availability filter")

    now = datetime.utcnow()
    cooled = []
    for t in top_tickers:
        last_time = NOTIFICATION_CACHE.get(t["symbol"])
        if last_time and (now - last_time) < timedelta(hours=COOLDOWN_HOURS):
            logger.info(f"Skipping {t['symbol']} due to cooldown. last_notified={last_time}")
            continue
        cooled.append(t)
    logger.info(f"{len(cooled)} symbols remain after cooldown")

    scored = []
    for t in cooled:
        symbol = t["symbol"]
        current_price = t["last_price"]
        #logger.info(f"Processing {symbol}: price={current_price}, 24h_change={t['change_pct']:.2f}%")
        try:
            df_5m = fetch_ohlcv(symbol, interval='5m')
            df_15m = fetch_ohlcv(symbol, interval='15m')
            df_60m = fetch_ohlcv(symbol, interval='60m')
            if any(x is None or x.empty for x in [df_5m, df_15m, df_60m]):
                #logger.warning(f"{symbol} skipped: missing OHLCV data -> 5m:{None if df_5m is None else len(df_5m)}, 15m:{None if df_15m is None else len(df_15m)}, 60m:{None if df_60m is None else len(df_60m)}")
                continue

            # 非AI BOS と AI BOS の統合判定（AI が有効なら補正）
            score, notes, bos_decision, ai_conf, ai_reason = score_short_setup(symbol, df_5m, df_15m, df_60m)
            #logger.info(f"{symbol} scored: score={score}, bos_decision={bos_decision}, ai_conf={ai_conf:.2f}, notes={notes}")

            # 通知条件: (1) スコア閾値以上 AND ((BOSがある) OR (緩和モードON))
            if score >= SCORE_THRESHOLD and (bos_decision or RELAX_NOTIFICATION_RULES):
                plan = plan_short_trade(df_5m)
                entry = plan['entry']
                tp1 = plan['tp1']
                tp1_pct = (tp1 - entry) / entry * 100

                # TP1 チェックはENVで無効化可能（デバッグ用）
                if not DISABLE_TP1_CHECK and tp1_pct > TP1_THRESHOLD:
                    logger.info(f"{symbol} skipped: TP1 threshold not met (tp1_pct={tp1_pct:.2f}% > {TP1_THRESHOLD}%)")
                else:
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
                        "comment": comment,
                    })
                    logger.info(f"{symbol} added to scored list (tp1_pct={tp1_pct:.2f}%)")
            #else:
            #    logger.info(f"{symbol} skipped: conditions not met (score {score} / needed {SCORE_THRESHOLD}, bos_decision {bos_decision}, RELAX={RELAX_NOTIFICATION_RULES})")
        except Exception:
            logger.error(f"{symbol} 分析中にエラー:\n{traceback.format_exc()}")

    scored.sort(key=lambda x: (x["score"], x["change_pct"]), reverse=True)
    logger.info(f"{len(scored)} total candidates after scoring; preparing to send up to {MAX_ALERTS_PER_RUN} alerts")
    alerts_sent = 0
    for s in scored[:MAX_ALERTS_PER_RUN]:
        try:
            logger.info(f"Sending alert for {s['symbol']} (score={s['score']}, change={s['change_pct']:.2f}%)")
            send_short_signal(
                s["symbol"], s["current_price"], s["score"], s["notes"], s["plan"], s["change_pct"], s["indicators"],
                comment=s.get("comment", "")
            )
            NOTIFICATION_CACHE[s["symbol"]] = now
            logger.info(f"Notification recorded for {s['symbol']} at {now}")
            alerts_sent += 1
            time.sleep(1)
        except Exception as e:
            logger.error(f"Failed to send alert for {s['symbol']}: {e}")

    if alerts_sent == 0:
        logger.info("No alerts sent in this run.")

@app.route("/")
def index():
    return "OK"

@app.route("/run_analysis", methods=["GET", "HEAD"])
def run_analysis_route():
    run_analysis()
    return "分析完了", 200

if __name__ == "__main__":
    # optional quick test: if env says so, attempt to send a test message (helps debug render env)
    if os.getenv("SEND_TELEGRAM_TEST_ON_STARTUP", "0") == "1":
        try:
            tg_send_md("VirtualCurrencyAnalysisBot 起動テストメッセージ")
        except Exception as e:
            logger.error(f"Startup test message failed: {e}")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
