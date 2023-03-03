from flask import Flask
from flask import request
import MeCab

app = Flask(__name__)

@app.route("/", methods=['GET'])
def keyword_suggester():
    keyword_list = keyword_list_load()
    keyword_count = {}
    s = request.json['sentence']

    # 形態素解析追加
    wakati = MeCab.Tagger("-Owakati")
    node = wakati.parseToNode(s)
    while node:
        if node.feature.split(",")[0] == "動詞":
            # 動詞の時のみ原型も確認
            s += "," + node.feature.split(",")[7]
        node = node.next

    # 一致チェック
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

