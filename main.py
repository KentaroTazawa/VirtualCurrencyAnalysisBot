
import os
from flask import Flask, request

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return "OK", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
