import os
import time
import traceback
import json
import hmac
import hashlib
import math
import logging
import threading
from datetime import datetime, timedelta
import requests
import pandas as pd
from flask import Flask, request
from dotenv import load_dotenv

# =====================
# 初期設定
# =====================
load_dotenv()

MEXC_BASE_URL = os.getenv("MEXC_BASE_URL", "https://contract.mexc.com")
MEXC_API_KEY = os.getenv("MEXC_API_KEY")
MEXC_SECRET_KEY = os.getenv("MEXC_SECRET_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
RUN_SECRET = os.getenv("RUN_SECRET")

AUTO_ORDER_ENABLE = os.getenv("AUTO_ORDER_ENABLE", "false").lower() == "true"
AUTO_ORDER_NOTIONAL = float(os.getenv("AUTO_ORDER_NOTIONAL", "10"))
AUTO_ORDER_LEVERAGE = int(os.getenv("AUTO_ORDER_LEVERAGE", "1"))

# Flask
app = Flask(__name__)
RUN_LOCK = threading.Lock()

# Telegram通知
def send_message_to_telegram(text: str, markdown=False):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "Markdown" if markdown else None,
        }
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        logging.error(f"Telegram送信失敗: {e}")

def send_error_to_telegram(text: str):
    send_message_to_telegram(f"❌ エラー:\n{text}")

def tg_send_md(text: str):
    # テスト用print削除し、通知統一
    send_message_to_telegram(text, markdown=True)

# =====================
# HTTP セッション設定
# =====================
from requests.adapters import HTTPAdapter, Retry
session = requests.Session()
retries = Retry(total=3, backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retries)
session.mount("https://", adapter)
session.mount("http://", adapter)

def _sign_message(query: str, timestamp: str) -> str:
    payload = f"{MEXC_API_KEY}{timestamp}{query}"
    return hmac.new(
        MEXC_SECRET_KEY.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()

def mexc_get(path: str, params=None):
    try:
        url = MEXC_BASE_URL + path
        resp = session.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        send_error_to_telegram(f"mexc_get失敗: {path} {e}")
        return None

def mexc_private_get(path: str, params=None):
    try:
        ts = str(int(time.time() * 1000))
        query = "&".join([f"{k}={v}" for k, v in (params or {}).items()])
        sig = _sign_message(query, ts)
        headers = {
            "ApiKey": MEXC_API_KEY,
            "Request-Time": ts,
            "Signature": sig,
        }
        url = MEXC_BASE_URL + path
        resp = session.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        send_error_to_telegram(f"mexc_private_get失敗: {path} {e}")
        return None

def mexc_private_post(path: str, data=None):
    try:
        ts = str(int(time.time() * 1000))
        body = json.dumps(data or {})
        sig = _sign_message(body, ts)
        headers = {
            "ApiKey": MEXC_API_KEY,
            "Request-Time": ts,
            "Signature": sig,
            "Content-Type": "application/json",
        }
        url = MEXC_BASE_URL + path
        resp = session.post(url, headers=headers, data=body, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        send_error_to_telegram(f"mexc_private_post失敗: {path} {e}")
        return None

# =====================
# テクニカル指標
# =====================
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=period-1, adjust=False).mean()
    ma_down = down.ewm(com=period-1, adjust=False).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window=period).mean()

def bollinger_bands(series: pd.Series, period: int = 20, num_std=2):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper, lower

# =====================
# データ取得
# =====================
def fetch_ohlcv(symbol: str, interval="1m", limit=200):
    data = mexc_get("/api/v1/contract/kline/" + symbol,
                    {"interval": interval, "limit": limit})
    if not data:
        return None
    try:
        df = pd.DataFrame(data, columns=[
            "time", "open", "high", "low", "close", "volume"
        ])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
        return df
    except Exception as e:
        send_error_to_telegram(f"fetch_ohlcv変換失敗 {symbol}: {e}")
        return None

def get_contract_detail(symbol: str):
    data = mexc_get("/api/v1/contract/detail")
    if not data:
        return None
    for d in data.get("data", []):
        if d.get("symbol") == symbol:
            return d
    return None

# =====================
# ボリューム計算
# =====================
def calculate_volume_for_notional(symbol: str, price: float, notional_usdt: float):
    detail = get_contract_detail(symbol)
    if not detail:
        return None, "contract detail not found"
    contract_size = float(detail.get("contractSize", 1.0))
    vol_step = float(detail.get("volUnit", 1.0))  # 修正: float扱い
    min_vol = float(detail.get("minVol", vol_step))
    one_contract_notional = max(1e-12, price * contract_size)
    raw_vol = notional_usdt / one_contract_notional
    steps = math.floor(raw_vol / vol_step)
    vol = steps * vol_step
    if vol < min_vol:
        vol = min_vol
    if vol <= 0:
        return None, "calculated volume <= 0"
    # 整数単位なら int に変換
    if abs(vol - round(vol)) < 1e-9 and vol_step >= 1:
        vol = int(round(vol))
    return vol, None

# =====================
# スコアリング
# =====================
def score_short_setup(df: pd.DataFrame):
    if df is None or len(df) < 50:
        return 0, []
    signals = []
    score = 0
    close = df["close"]

    # RSI
    rsi_val = rsi(close, 14).iloc[-1]
    if rsi_val > 70:
        score += 2
        signals.append(f"RSI過熱 {rsi_val:.1f}")

    # EMA 乖離
    ema50 = ema(close, 50).iloc[-1]
    if close.iloc[-1] > ema50 * 1.05:
        score += 2
        signals.append(f"50EMA乖離 +{(close.iloc[-1]/ema50-1)*100:.1f}%")

    # ボリンジャーバンド
    upper, _ = bollinger_bands(close)
    if close.iloc[-1] > upper.iloc[-1]:
        score += 1
        signals.append("BB上抜け")

    return score, signals

# =====================
# トレードプラン
# =====================
def plan_short_trade(df: pd.DataFrame):
    entry = df["close"].iloc[-1]
    atr_val = atr(df).iloc[-1]
    sl = entry + 2 * atr_val
    tp = entry - 2 * atr_val
    return {"entry": entry, "sl": sl, "tp": tp}

# =====================
# 注文処理
# =====================
def validate_order(symbol: str, vol, entry_price: float):
    detail = get_contract_detail(symbol)
    if not detail:
        return False, "detailなし"
    min_vol = float(detail.get("minVol", 1))
    vol_step = float(detail.get("volUnit", 1))
    price_scale = int(detail.get("priceScale", 2))
    tick = 10 ** -price_scale

    if vol < min_vol:
        return False, f"vol {vol} < minVol {min_vol}"
    if abs(vol / vol_step - round(vol / vol_step)) > 1e-9:
        return False, f"vol {vol} not multiple of step {vol_step}"
    if abs(entry_price / tick - round(entry_price / tick)) > 1e-9:
        return False, f"price {entry_price} not aligned with tick {tick}"
    return True, None

def place_market_short_order(symbol: str, vol, leverage: int):
    try:
        # レバレッジ設定
        mexc_private_post("/api/v1/private/position/leverage",
                          {"symbol": symbol, "positionMode": "Hedge", "leverage": leverage})

        # 成行ショート注文
        data = {
            "symbol": symbol,
            "vol": vol,
            "side": 2,  # 2=ショート
            "type": 1,  # 1=成行
            "openType": "ISOLATED",
            "positionId": 0,
        }
        resp = mexc_private_post("/api/v1/private/order/submit", data)
        if not resp or resp.get("success") != True:
            return None, f"注文失敗: {resp}"
        return resp.get("data", {}).get("orderId"), None
    except Exception as e:
        return None, str(e)

# =====================
# シグナル通知
# =====================
NOTIFICATION_CACHE = {}

def send_short_signal(symbol: str, score, signals, plan, vol=None, order_id=None):
    msg = f"📉 ショート候補: {symbol}\n"
    msg += f"スコア: {score}\n"
    msg += "根拠: " + ", ".join(signals) + "\n"
    msg += f"計画: Entry {plan['entry']:.4f}, SL {plan['sl']:.4f}, TP {plan['tp']:.4f}\n"
    if vol:
        msg += f"数量: {vol}\n"
    if order_id:
        msg += f"✅ 注文成功 ID: {order_id}\n"
    tg_send_md(msg)

# =====================
# 分析本体
# =====================
def run_analysis():
    try:
        # 急騰銘柄を取得
        rising = mexc_get("/api/v1/contract/ticker")
        if not rising:
            return
        sorted_symbols = sorted(rising.get("data", []),
                                key=lambda x: float(x.get("riseFallRate", 0)),
                                reverse=True)[:10]

        for item in sorted_symbols:
            symbol = item.get("symbol")
            df = fetch_ohlcv(symbol, "1m", 200)
            if df is None or len(df) < 50:
                continue

            score, signals = score_short_setup(df)
            if score < 5:
                continue

            plan = plan_short_trade(df)
            cache_key = f"{symbol}-{int(df['time'].iloc[-1].timestamp())}"
            if cache_key in NOTIFICATION_CACHE:
                continue
            NOTIFICATION_CACHE[cache_key] = True

            vol, err = calculate_volume_for_notional(symbol, plan["entry"], AUTO_ORDER_NOTIONAL)
            if err:
                send_error_to_telegram(f"{symbol} 数量計算失敗: {err}")
                continue

            order_id = None
            if AUTO_ORDER_ENABLE:
                ok, err = validate_order(symbol, vol, plan["entry"])
                if not ok:
                    send_error_to_telegram(f"{symbol} 注文検証失敗: {err}")
                    continue
                order_id, err = place_market_short_order(symbol, vol, AUTO_ORDER_LEVERAGE)
                if err:
                    send_error_to_telegram(f"{symbol} 注文失敗: {err}")
                    continue

            send_short_signal(symbol, score, signals, plan, vol, order_id)

    except Exception as e:
        send_error_to_telegram(f"run_analysis失敗:\n{traceback.format_exc()}")

# =====================
# Flaskエンドポイント
# =====================
@app.route("/run_analysis", methods=["GET", "HEAD"])
def run_analysis_route():
    req_secret = request.args.get("secret") or request.headers.get("X-Run-Secret")
    if RUN_SECRET and req_secret != RUN_SECRET:
        return "Forbidden", 403
    if not RUN_LOCK.acquire(blocking=False):
        return "Another run in progress", 409
    try:
        run_analysis()
        return "分析完了", 200
    finally:
        RUN_LOCK.release()

# =====================
# メイン
# =====================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
