from flask import Flask
from flask import request

app = Flask(__name__)

@app.route("/", methods=['GET'])
def keyword_suggester():
    keyword_list = keyword_list_load()
    keyword_count = {}
    s = request.json['sentence']
    for keyword in keyword_list:
        count =  s.count(keyword)
        if count > 0:
            keyword_count[keyword] = count
    return keyword_count

def keyword_list_load():
    list = []
    with open('keywords.txt', "r") as f:
        list = [k.strip() for k in f.readlines()]
    f.close()
    return list

