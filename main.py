import os
import time
import traceback
import json
import hmac
import hashlib
import math
from datetime import datetime, timedelta
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
from flask import Flask
from dotenv import load_dotenv

load_dotenv()

MEXC_BASE_URL = os.getenv("MEXC_BASE_URL", "https://contract.mexc.com")

# Required env (for trading)
MEXC_API_KEY = os.getenv("MEXC_API_KEY")
MEXC_API_SECRET = os.getenv("MEXC_API_SECRET")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Groq client left as optional (original code used Groq)
try:
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
except Exception:
    client = None

app = Flask(__name__)

# --- HTTP session with retries and keep-alive ---
session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=frozenset(["GET", "POST", "PUT", "DELETE", "HEAD"])
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)
session.headers.update({"User-Agent": "mexc-bot/1.0"})


# ====== 運用パラメータ（緩めにして機会を増やす） ======
TOP_SYMBOLS_LIMIT = 40
MAX_ALERTS_PER_RUN = 5
COOLDOWN_HOURS = 1.0
USE_GROQ_COMMENTARY = False
GROQ_MODEL = "llama3-70b-8192"

# ====== シグナル・しきい値（過熱検出へ全面移行） ======
MIN_24H_CHANGE_PCT = 10.0
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

# スコア
SCORE_THRESHOLD = 5

# 利確・損切り（固定R管理）
ATR_PERIOD = 14
SL_ATR_MULT = 0.5
TP1_R = 1.0
TP2_R = 2.0

# 自動注文用条件（ユーザー指定）
AUTO_ORDER_MIN_SCORE = 6
# 判定基準: ここでは plan['tp2'] の想定利幅（絶対値）が >= AUTO_ORDER_MIN_TP_PCT と解釈
AUTO_ORDER_MIN_TP_PCT = 8.0

# 実際の注文パラメータ（自動注文時に使う）
AUTO_ORDER_LEVERAGE = 1
AUTO_ORDER_ASSET_PCT = 0.30  # 資産の30%分を注文（notional）
AUTO_ORDER_TAKE_PROFIT_PCT = 3.0
AUTO_ORDER_STOP_LOSS_PCT = 15.0

NOTIFICATION_CACHE = {}  # {symbol: last_notified_timestamp}

# ========= ユーティリティ =========
def mexc_get(path: str, timeout=10, params=None):
    url = f"{MEXC_BASE_URL}{path}"
    # use session with retries; timeout=(connect, read)
    res = session.get(url, timeout=(5, max(30, timeout)), params=params)
    res.raise_for_status()
    return res.json()

def send_error_to_telegram(error_message: str):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        session.post(
            url,
            data={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": f"⚠️ エラー発生:\n\n{error_message[:3800]}",
            },
            timeout=(5, 10),
        )
    except Exception:
        pass

def tg_send_md(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text[:4096],
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }
    print(f"⚠️ log0001")
    try:
        session.post(url, data=payload, timeout=(5, 10))
        print(f"⚠️ log0002")
    except requests.exceptions.Timeout:
        send_error_to_telegram("Telegram送信エラー: タイムアウト発生")
    except Exception as e:
        send_error_to_telegram(f"Telegram送信エラー:\n{str(e)}")

# ======== MEXC Private API helpers (署名付き) ===========
def _build_query_string(params: dict) -> str:
    if not params:
        return ""
    # GET用: 辞書順 & で結合
    pairs = []
    for k in sorted(params.keys()):
        v = params[k]
        pairs.append(f"{k}={v}")
    return "&".join(pairs)

def _sign_message(access_key: str, secret_key: str, req_time: str, param_string: str) -> str:
    target = f"{access_key}{req_time}{param_string}"
    sig = hmac.new(secret_key.encode(), target.encode(), hashlib.sha256).hexdigest()
    return sig

def mexc_private_get(path: str, params: dict = None, timeout=15):
    if not MEXC_API_KEY or not MEXC_API_SECRET:
        raise RuntimeError("MEXC API keys not configured (MEXC_API_KEY/MEXC_API_SECRET).")
    url = f"{MEXC_BASE_URL}{path}"
    req_time = str(int(time.time() * 1000))
    param_string = _build_query_string(params) if params else ""
    signature = _sign_message(MEXC_API_KEY, MEXC_API_SECRET, req_time, param_string)
    headers = {
        "ApiKey": MEXC_API_KEY,
        "Request-Time": req_time,
        "Signature": signature,
        "Content-Type": "application/json",
    }
    r = session.get(url, headers=headers, params=params, timeout=(5, max(30, timeout)))
    r.raise_for_status()
    return r.json()

def mexc_private_post(path: str, body: dict = None, timeout=15):
    if not MEXC_API_KEY or not MEXC_API_SECRET:
        raise RuntimeError("MEXC API keys not configured (MEXC_API_KEY/MEXC_API_SECRET).")
    url = f"{MEXC_BASE_URL}{path}"
    req_time = str(int(time.time() * 1000))
    # POST の場合は JSON 文字列をそのまま署名に使う（docs に合わせ、余計な空白は抑える）
    param_string = json.dumps(body, separators=(",", ":"), ensure_ascii=False) if body else ""
    signature = _sign_message(MEXC_API_KEY, MEXC_API_SECRET, req_time, param_string)
    headers = {
        "ApiKey": MEXC_API_KEY,
        "Request-Time": req_time,
        "Signature": signature,
        "Content-Type": "application/json",
    }
    r = session.post(url, headers=headers, data=param_string.encode("utf-8"), timeout=(5, max(30, timeout)))
    r.raise_for_status()
    return r.json()

# ========= データ取得 =========
def get_top_symbols_by_24h_change(limit=TOP_SYMBOLS_LIMIT):
    try:
        data = mexc_get("/api/v1/contract/ticker")
        tickers = data.get("data", []) or []
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
        return filtered[:limit]
    except Exception as e:
        send_error_to_telegram(f"MEXC 急上昇銘柄取得エラー:\n{str(e)}")
        return []

def get_available_contract_symbols():
    try:
        data = mexc_get("/api/v1/contract/detail")
        arr = data.get("data", []) or []
        available = set()
        for it in arr:
            symbol = it.get("symbol")
            state = it.get("state", 0)  # 0 = 正常稼働
            if symbol and state == 0:
                available.add(symbol)
        return available
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
    if break_of_structure_short(df_5m):
        score += 2; notes.append("5m BOS下抜け")
    return score, notes

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

# ========= MEXC: 注文に必要な補助 =========
def get_contract_detail(symbol: str):
    try:
        # public endpoint supports symbol query
        resp = mexc_get(f"/api/v1/contract/detail?symbol={symbol}")
        data = resp.get("data") or []
        if isinstance(data, list):
            return data[0] if data else None
        return data
    except Exception as e:
        send_error_to_telegram(f"契約情報取得失敗 {symbol}:\n{str(e)}")
        return None

def get_usdt_asset():
    try:
        resp = mexc_private_get("/api/v1/private/account/assets", {"currency": "USDT"})
        if isinstance(resp, list) and len(resp) > 0:
            return float(resp[0].get("availableBalance", 0.0))
        return 0.0
    except Exception as e:
        send_error_to_telegram(f"アセット取得失敗:\n{str(e)}")
        return 0.0

def calculate_volume_for_notional(symbol: str, price: float, notional_usdt: float):
    """
    単純化: vol = notional / (price * contractSize)
    丸めは volUnit に合わせて下方向に切り捨て。最小 vol は minVol
    """
    detail = get_contract_detail(symbol)
    if not detail:
        return None, "contract detail not found"
    contract_size = float(detail.get("contractSize", 1.0))
    vol_unit = int(detail.get("volUnit", 1) or 1)
    min_vol = float(detail.get("minVol", 1.0) or 1.0)
    # notional をコントラクト数に変換
    # 1 contract notional = price * contract_size
    one_contract_notional = max(1e-9, price * contract_size)
    raw_vol = notional_usdt / one_contract_notional
    # floor to vol_unit
    try:
        vol = math.floor(raw_vol / vol_unit) * vol_unit
    except Exception:
        vol = int(max(1, math.floor(raw_vol)))
    if vol < min_vol:
        vol = int(min_vol)
    if vol <= 0:
        vol = int(max(1, math.floor(raw_vol)))
    return int(vol), None

# ====== 修正: 通知系関数の統一 ======
def send_message_to_telegram(text: str, markdown: bool = True):
    """通常メッセージ通知"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram設定未了:", text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
    }
    if markdown:
        payload["parse_mode"] = "Markdown"
    try:
        session.post(url, json=payload, timeout=(5, 10))
    except Exception as e:
        print(f"Telegram送信失敗: {e}")


# ====== 修正: test_order_placeable をバリデーション関数にする ======
def validate_order(symbol: str, entry_price: float, vol: int, leverage: int = 1) -> (bool, str):
    """
    注文可能かをテストする（実際の test endpoint を使わず、契約条件・残高などでチェック）
    戻り値: (可能か, エラー理由 or 空文字)
    """
    detail = get_contract_detail(symbol)
    if not detail:
        return False, "契約情報取得できず"
    # price scale チェック
    price_scale = int(detail.get("priceScale", 6) or 6)
    # 最小価格単位のチェック
    rounded = round(entry_price, price_scale)
    if abs(rounded - entry_price) > (10 ** (-price_scale)):
        return False, f"価格刻みが priceScale({price_scale}) に合わない: {entry_price} -> {rounded}"
    # vol 単位チェック
    min_vol = float(detail.get("minVol", 1.0) or 1.0)
    vol_unit = int(detail.get("volUnit", 1) or 1)
    if vol < min_vol:
        return False, f"vol({vol})が最小vol({min_vol})未満"
    if vol % vol_unit != 0:
        return False, f"vol単位({vol_unit})に合わない: vol={vol}"
    # 残高チェック
    balance = get_usdt_asset()
    if balance is None:
        return False, "残高取得失敗"
    notional = entry_price * vol * float(detail.get("contractSize",1.0))
    if notional > balance:
        return False, f"notional({notional:.4f})が獲得可能残高({balance:.4f})を超過"
    return True, ""

# ====== 修正: place_market_short_order 内で validate_order を使う ======
def place_market_short_order(symbol: str, entry_price: float, vol: int, leverage: int = 1,
                             tp_pct: float = AUTO_ORDER_TAKE_PROFIT_PCT, sl_pct: float = AUTO_ORDER_STOP_LOSS_PCT):
    if not MEXC_API_KEY or not MEXC_API_SECRET:
        return False, {"error": "MEXC API key/secret not set in env."}

    # Validate before real order
    ok, reason = validate_order(symbol, entry_price, vol, leverage)
    if not ok:
        msg = f"❌ 注文前バリデーション失敗\n銘柄: {symbol}\n理由: {reason}"
        send_message_to_telegram(msg)
        return False, {"error": f"validation failed: {reason}"}

    detail = get_contract_detail(symbol) or {}
    price_scale = int(detail.get("priceScale", 6) or 6)
    tp_price = round(entry_price * (1.0 - tp_pct / 100.0), price_scale)
    sl_price = round(entry_price * (1.0 + sl_pct / 100.0), price_scale)

    body = {
        "symbol": symbol,
        "vol": vol,
        "leverage": int(leverage),
        "side": 3,  # ショート
        "type": 5,  # 成行? 注文タイプ確認が必要
        "openType": 1,
        "stopLossPrice": sl_price,
        "takeProfitPrice": tp_price,
    }
    try:
        resp = mexc_private_post("/api/v1/private/order/submit", body=body)
        success = bool(resp.get("success") is True or resp.get("code") == 0)

        if success:
            order_data = resp.get("data", {})
            order_id = order_data.get("orderId", order_data.get("order_id", "N/A"))
            msg = (f"✅ 自動ショート注文 成功\n"
                   f"銘柄: {symbol}\n"
                   f"vol: {vol}\n"
                   f"leverage: {leverage}\n"
                   f"注文ID: {order_id}")
            send_message_to_telegram(msg)
        else:
            # エラー理由をメッセージに含める
            msg = (f"❌ 自動ショート注文 失敗\n"
                   f"銘柄: {symbol}\n"
                   f"レスポンス: {resp}")
            send_message_to_telegram(msg)

        return success, resp
    except Exception as e:
        msg = f"❌ 自動ショート注文 例外発生\n銘柄: {symbol}\nエラー: {str(e)}"
        send_message_to_telegram(msg)
        return False, {"error": str(e)}
        
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
    text = f"""*📉 ショート候補: {display_symbol}*

24h変化率: {change_pct:.2f}%  / 現値: {current_price}

スコア: {score} / 必要 {SCORE_THRESHOLD}
根拠: {notes_text}

計画 (%表記)
Entry: {entry}
SL: {sl_pct:+.2f}%  (risk/qty: {plan['risk_per_unit']})
TP1: {tp1_pct:+.2f}% ({TP1_R}R)
TP2: {tp2_pct:+.2f}% ({TP2_R}R, 到達R: {plan['r_multiple_to_tp2']})

参考指標
{ind_text}

{comment}
"""
    tg_send_md(text)

# ========= メイン =========
def run_analysis():
    top_tickers = get_top_symbols_by_24h_change()
    available = get_available_contract_symbols()
    top_tickers = [t for t in top_tickers if t["symbol"] in available]
    now = datetime.utcnow()
    cooled = []
    for t in top_tickers:
        last_time = NOTIFICATION_CACHE.get(t["symbol"])
        if last_time and (now - last_time) < timedelta(hours=COOLDOWN_HOURS):
            continue
        cooled.append(t)
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
    scored.sort(key=lambda x: (x["score"], x["change_pct"]), reverse=True)
    alerts_sent = 0
    for s in scored[:MAX_ALERTS_PER_RUN]:
        comment = groq_commentary(s["symbol"], s["notes"], s["plan"]) if USE_GROQ_COMMENTARY else ""
        send_short_signal(
            s["symbol"], s["current_price"], s["score"], s["notes"], s["plan"], s["change_pct"], s["indicators"],
            comment=comment,
        )
        NOTIFICATION_CACHE[s["symbol"]] = now
        alerts_sent += 1
        # 自動注文の判定
        try:
            entry = s["plan"]["entry"]
            tp2_pct = (s["plan"]["tp2"] - entry) / entry * 100.0
            if s["score"] >= AUTO_ORDER_MIN_SCORE and abs(tp2_pct) >= AUTO_ORDER_MIN_TP_PCT:
                # 自動注文を試みる
                # 1) 残高取得 -> notional
                balance = get_usdt_asset()
                if balance is None:
                    print(f"自動注文失敗: 残高取得できませんでした: {s['symbol']}")
                else:
                    notional = balance * AUTO_ORDER_ASSET_PCT
                    vol, err = calculate_volume_for_notional(s["symbol"], entry, notional)
                    if err:
                        print(f"自動注文失敗: ボリューム計算失敗 {s['symbol']}: {err}")
                    else:
                        success, resp = place_market_short_order(s["symbol"], entry, vol, leverage=AUTO_ORDER_LEVERAGE,
                                                                 tp_pct=AUTO_ORDER_TAKE_PROFIT_PCT, sl_pct=AUTO_ORDER_STOP_LOSS_PCT)
                        if success:
                            order_id = resp.get("data") or resp.get("orderId") or resp
                            print(f"✅ 自動ショート注文 成功\n銘柄: {s['symbol']}\nvol: {vol}\nleverage: {AUTO_ORDER_LEVERAGE}\n注文ID: {order_id}")
                        else:
                            # 注文失敗: 理由通知
                            err_msg = resp
                            print(f"❌ 自動ショート注文 失敗\n銘柄: {s['symbol']}\nreason: {json.dumps(err_msg, ensure_ascii=False)[:1500]}")
        except Exception:
            send_error_to_telegram(f"自動注文処理で例外:\n{traceback.format_exc()}")
        time.sleep(1)

@app.route("/")
def index():
    return "OK"

@app.route("/run_analysis", methods=["GET", "HEAD"])
def run_analysis_route():
    run_analysis()
    return "分析完了", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
