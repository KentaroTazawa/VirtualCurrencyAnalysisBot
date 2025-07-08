
from flask import Flask, request

app = Flask(__name__)

@app.route("/", methods=["POST"])
def handler():
    print("[INFO] リクエストを受信しました")
    return "OK", 200
