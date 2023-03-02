from flask import Flask
from flask import request

app = Flask(__name__)

@app.route("/", methods=['POST'])
def keyword_suggester():
    s = request.json['sentence']
    return s
