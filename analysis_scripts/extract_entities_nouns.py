import spacy
import json
from tqdm import tqdm

file_list = ["../data/squad/train-v1.1.json", "../data/squad/dev-v1.1.json", "../data/squad2/train-v2.0.json",
             "../data/squad2/dev-v2.0.json"]
ent_output_file = "ent_dicts/ent_dict_squad.json"
noun_output_file = "ent_dicts/noun_squad.json"
nlp = spacy.load("en_core_web_md")
contexts = []
for file in file_list:
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)["data"]
    for doc in data:
        for para in doc["paragraphs"]:
            contexts.append(para["context"])

ent_dict = {}
noun_list = set()
processed_contexts = nlp.pipe(contexts)
for context in tqdm(processed_contexts):
    for ent in context.ents:
        if not ent_dict.__contains__(ent.label_):
            ent_dict[ent.label_] = set()
        ent_dict[ent.label_].add(ent.text)
    for token in context:
        if token.pos_ == "NOUN":
            noun_list.add(token.lemma_)

for k, v in ent_dict.items():
    ent_dict[k] = list(v)
noun_list = list(noun_list)
with open(ent_output_file, "w", encoding="utf-8") as f:
    json.dump(ent_dict, f)
with open(noun_output_file, "w", encoding="utf-8") as f:
    json.dump(noun_list, f)
