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
def get_keyword():
    s = request.json['sentence']
    suggest = keyword_suggester(s)
    match = keyword_matcher(s)

    match_size = min(5, len(match))
    match =  select_from_simuler(match, match_size)
    match = dict([(k, 1) for k in match])
    res =  select_from_simuler({**suggest, **match}, 20)

    return res

@app.route("/match", methods=['POST'])
def get_matcht_keyword():
    s = request.json['sentence']
    return keyword_matcher(s)

@app.route("/suggest", methods=["POST"])
def get_suggest_keyword():
    s = request.json['sentence']
    return keyword_suggester(s)

def keyword_suggester(s):
    s_embedding = embedding_avg(s)

    keyword_count = {}

    for keyword, key_embedding in keyword_embedding_list.items():
        keyword_count[keyword] = torch.cosine_similarity(s_embedding, key_embedding, dim = 1).item()

    return select_from_simuler(keyword_count, 20)

def keyword_matcher(s):
    keyword_list = keyword_list_load()
    katakana_list = katakana_list_load()
    keyword_count = {}
    katakana_count = {}
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

def select_from_simuler(dic, num):
    sorted_list = sorted(dic.items(), key = lambda dic : dic[1])

    return dict(sorted_list[-num:])
