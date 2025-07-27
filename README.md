# 📉 Crypto Short Signal Bot - ATH追跡型（OKX + CoinGecko + Groq + Telegram）

このBotは、**OKXのUSDT-SWAP銘柄のうち、急騰した上位10銘柄**の中から、**CoinGeckoの価格履歴データ**を使って「史上最高値（ATH）を直近で更新」したものだけを抽出し、**Groq（LLaMA3-70B）によるAI分析**でショート注文の可否を判定します。結果はTelegramに自動通知されます。

---

## 🔧 使用技術

- OKX API（先物銘柄・24h価格変動取得）
- CoinGecko API（OHLC価格履歴取得）
- Groq API（LLaMA3でのAI分析）
- Flask（Webサーバー）
- Telegram Bot（通知）
- Python + Pandas

---

## 📌 機能概要

1. **OKXで24時間急騰した上位10銘柄を取得**
2. **CoinGeckoのAPIで該当銘柄の過去全価格を取得**
3. **ATH（All Time High）を更新しているかを判定**
4. **Groq（LLaMA3）でAI分析し、ショートすべきか判定**
5. **ショートが有効と判断された銘柄のみTelegramに通知**

---

## 📲 通知形式（Telegram）

