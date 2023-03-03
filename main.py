from flask import Flask
from flask import request
import MeCab
from transformers import BertJapaneseTokenizer, BertForPreTraining
import torch
import pickle

app = Flask(__name__)

tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-char-whole-word-masking')
model = BertForPreTraining.from_pretrained('cl-tohoku/bert-base-japanese-char-whole-word-masking', output_hidden_states = True,)

keyword_embedding_list = {}
with open('keywords_embedding.pkl', 'rb') as f:
    keyword_embedding_list = pickle.load(f)

@app.route("/", methods=['POST'])
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

@app.route("/suggest", methods=["POST"])
def suggest():
    s = request.json['sentence']
    s_embedding = embedding_avg(s)

    keyword_count = {}

    for keyword, key_embedding in keyword_embedding_list.items():
        keyword_count[keyword] = torch.cosine_similarity(s_embedding, key_embedding, dim = 1).item()

    sorted_count_list = sorted(keyword_count.items(), key = lambda keyword_count : keyword_count[1])

    return dict(sorted_count_list[-20:])

def embedding_avg(s):
    input_ids = torch.tensor(tokenizer.encode(s, add_special_tokens=True, max_length=512)).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids)

    return torch.mean(outputs.hidden_states[-1], dim=1)

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
