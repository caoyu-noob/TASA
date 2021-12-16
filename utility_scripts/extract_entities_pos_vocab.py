import spacy
import json
from tqdm import tqdm
from nltk.corpus import wordnet as wn

ANSWER_POS = ["ADV", "ADJ", "NOUN", "VERB", "NUM"]

# file_list = ["../data/squad/train-v1.1.json", "../data/squad/dev-v1.1.json", "../data/squad2/train-v2.0.json",
#              "../data/squad2/dev-v2.0.json"]
# file_list = ["../data/newsqa/newsqa_dev.json", "../data/newsqa/newsqa_train.json"]
file_list = ["../data/nq/nq-dev.json", "../data/nq/nq-train.json"]
# ent_output_file = "ent_dicts/ent_dict_newsqa.json"
# pos_vocab_output_file = "ent_dicts/pos_vocab_dict_newsqa.json"
ent_output_file = "../auxiliary_data/ent_dicts/ent_dict_nq.json"
pos_vocab_output_file = "../auxiliary_data/ent_dicts/pos_vocab_dict_nq.json"
nlp = spacy.load("en_core_web_md")
contexts = []
for file in file_list:
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)["data"]
    for doc in data:
        for para in doc["paragraphs"]:
            contexts.append(para["context"])

ent_dict = {}
pos_vocab_dict = {}
processed_contexts = nlp.pipe(contexts)
for context in tqdm(processed_contexts):
    for ent in context.ents:
        if not ent_dict.__contains__(ent.label_):
            ent_dict[ent.label_] = set()
        ent_dict[ent.label_].add(ent.text)
    for token in context:
        if token.pos_ in ANSWER_POS:
            if not pos_vocab_dict.__contains__(token.pos_):
                pos_vocab_dict[token.pos_] = set()
            pos_vocab_dict[token.pos_].add(token.text)

for k, v in ent_dict.items():
    ent_dict[k] = list(v)
for k, v in pos_vocab_dict.items():
    pos_vocab_dict[k] = list(v)
with open(ent_output_file, "w", encoding="utf-8") as f:
    json.dump(ent_dict, f)
with open(pos_vocab_output_file, "w", encoding="utf-8") as f:
    json.dump(pos_vocab_dict, f)
