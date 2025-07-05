# VirtualCurrencyAnalysisBot

GPT-4とロジックによる仮想通貨ショート分析Bot（Render無料Webサービス対応）

---

## 🔍 概要

- OKXの全USDT-SWAP銘柄を取得
- RSI > 70 の銘柄を抽出（過熱気味な通貨）
- 当日通知済みの銘柄は除外
- RSIが高い上位3銘柄のみをGPT-4で詳細分析
- GPTの結果において「利益の出る確率」が80%以上のもののみ、チャート画像と共にTelegramへ通知
- 実行時間は毎日 20:30〜23:30、10分間隔

---

## 🔧 使用技術

- Python + Flask（Render Webサービス対応）
- OKX REST API
- OpenAI GPT-4 API
- Telegram Bot API

---

## ⚙️ 必要な環境変数

`.env` または Render の「Environment Variables」に以下を設定：
