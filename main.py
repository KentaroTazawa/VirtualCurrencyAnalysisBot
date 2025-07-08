from flask import Flask, request
import os
from analyze import run_analysis  # 分析ロジックを分離

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def handler():
    return run_analysis()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
