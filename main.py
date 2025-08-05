import requests
import json
from datetime import datetime

COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"

def check_coingecko_list():
    url = f"{COINGECKO_BASE_URL}/coins/list"
    print(f"[{datetime.utcnow()}] CoinGecko API `/coins/list` にアクセス中...")

    try:
        response = requests.get(url, timeout=10)
        print(f"[HTTP ステータスコード]: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"[成功] コイン数: {len(data)}")
            print("[最初の5件の例]:")
            for coin in data[:5]:
                print(json.dumps(coin, ensure_ascii=False, indent=2))
        else:
            print(f"[失敗] ステータスコード: {response.status_code}")
            print(f"[レスポンス内容]: {response.text}")

    except Exception as e:
        print(f"[例外発生] {str(e)}")

if __name__ == "__main__":
    check_coingecko_list()
