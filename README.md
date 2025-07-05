# GPT-4 Short Recommendation Bot

## 概要
- Binanceから仮想通貨チャートを取得
- GPT-4に送ってショートするか分析
- Telegramで通知
- Renderで常時動作（無料）

## 必要な設定
- `.env` に以下を記入（Renderでは環境変数として設定）

## デプロイ方法（Render）
1. GitHubにこのプロジェクトをPush
2. Renderにログイン → "New Web Service"
3. リポジトリを選択
4. Python 3.10 / スタートコマンドは `python main.py`
5. 環境変数に `.env` の値を設定
