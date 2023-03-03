from flask import Flask
from flask import request
import MeCab

app = Flask(__name__)

@app.route("/", methods=['GET'])
def keyword_suggester():
    keyword_list = keyword_list_load()
    katakana_list = katakana_list_load()
    keyword_count = {}
    katakana_count = {}
    s = request.json['sentence']
    s_katakana = ""

    # 形態素解析追加
    wakati = MeCab.Tagger("-Owakati")
    node = wakati.parseToNode(s)
    while node:
        features = node.feature.split(",")
        if len(features) > 6:
            s_katakana += node.feature.split(",")[6]
        if node.feature.split(",")[0] == "動詞":
            # 動詞の時のみ原型も確認
            s += "," + node.feature.split(",")[7]
        node = node.next

    # 一致チェック
    for keyword in keyword_list:
        count =  s.count(keyword)
        if count > 0:
            keyword_count[keyword] = count

    # 読みでチェック
    for (i, katakana) in enumerate(katakana_list):
        if katakana == "":
            continue

        # カタカナに対応するような元のキーワードを選ぶ
        keyword = keyword_list[i]

        # 探す
        count =  s_katakana.count(katakana)
        if count > 0:
            katakana_count[keyword] = count

    # 結果のコンバイン
    res = {**katakana_count, **keyword_count}

    return res

def katakana_list_load():
    return list_load('keywords_katakana.txt')

def keyword_list_load():
    return list_load('keywords.txt')

def list_load(file):
    list = []
    with open(file, "r") as f:
        list = [k.strip() for k in f.readlines()]
    f.close()
    return list
