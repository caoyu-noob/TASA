import json
import spacy
import random
import re
import neuralcoref

from tqdm import tqdm
from itertools import chain

nlp = spacy.load("en_core_web_md")
neuralcoref.add_to_pipe(nlp)

with open("./data/squad/dev-v1.1.json", "r", encoding="utf-8") as f:
    data = json.load(f)
pair_data = []
for doc in data["data"]:
    for para in doc["paragraphs"]:
        for qa in para["qas"]:
            pair_data.append({"context": para["context"], "question": qa["question"], "answers": qa["answers"],
                              "id": qa["id"]})

coreferences = []
for pair in tqdm(pair_data):
    processed = nlp(pair["context"])
    cur_coref = []
    if processed._.has_coref:
        cur_coref = []
        for cluster in processed._.coref_clusters:
            coref_pair = []
            for mention in cluster.mentions:
                coref_pair.append([mention.text, mention.start_char, mention.end_char])
            cur_coref.append(coref_pair)
    coreferences.append(cur_coref)
print("1")

### Generate entity replace dict
# with open("ent_dict.json", "r", encoding="utf-8") as f:
#     ent_dict = json.load(f)
# ent_labels = list(ent_dict.keys())
# questions = [x["question"] for x in pair_data]
# nlp = spacy.load("en_core_web_md")
# processed_questions = nlp.pipe(questions, batch_size=100)
# replace_dicts = []
# for i, q in tqdm(enumerate(processed_questions)):
#     replace_dict = {}
#     for ent in q.ents:
#         # entity_candidates = []
#         random.shuffle(ent_labels)
#         while ent_labels[0] == ent.label_:
#             random.shuffle(ent_labels)
#         entity_candidates = ent_dict[ent_labels[0]]
#         # entity_candidates = ent_dict[ent.label_]
#         random.shuffle(entity_candidates)
#         tmp_entity_candidates = entity_candidates[:5]
#         while ent.text in tmp_entity_candidates:
#             random.shuffle(entity_candidates)
#             tmp_entity_candidates = entity_candidates[:5]
#         replace_dict[ent.text] = [e for e in tmp_entity_candidates]
#     replace_dicts.append(replace_dict)
# with open("replace_dicts_no_same_type.json", "w", encoding="utf-8") as f:
#     json.dump(replace_dicts, f)

### Generate new samples
new_data = []
with open("../replace_dicts_no_same_type.json", "r", encoding="utf-8") as f:
    replace_dicts = json.load(f)
for i, pair in enumerate(tqdm(pair_data)):
    replace_dict = replace_dicts[i]
    context = pair["context"]
    answer_starts = [a["answer_start"] for a in pair["answers"]]
    answer_texts = [a["text"] for a in pair["answers"]]
    # answer_ends = [a["answer_start"] + len(a["text"]) for a in pair["answers"]]
    new_contexts = [context] * 5
    new_questions = [pair["question"]] * 5
    for j in range(5):
        get_answer = True
        for k, v in replace_dict.items():
            # new_contexts[j] = new_contexts[j].replace(k, v[j])
            new_questions[j] = new_questions[j].replace(k, v[j])
        new_answers = []
        for a_i in range(len(answer_starts)):
            cur_answer_text = answer_texts[a_i].replace(")", "\)")
            cur_answer_text = cur_answer_text.replace("(", "\(")
            cur_answer_text = cur_answer_text.replace("*", "\*")
            answer_positions = [a.span(0)[0] for a in list(re.finditer(cur_answer_text, new_contexts[j]))]
            if len(answer_positions) == 0:
                get_answer = False
                break
            else:
                best_start, best_margin = answer_positions[0], abs(answer_positions[0] - answer_starts[a_i])
                for ap in answer_positions:
                    if abs(ap - answer_starts[a_i]) < best_margin:
                        best_start = ap
                new_answers.append({"answer_start": best_start, "text": answer_texts[a_i]})

        if get_answer:
           new_data.append({"context": new_contexts[j],
                            "qas": [{"answers": new_answers, "question": new_questions[j], "id": pair["id"] + "-" + str(j)}]})
with open("../dev-v1.1-entity_question_no_type.json", "w", encoding="utf-8") as f:
    json.dump({"data": [{"title": "replace", "paragraphs": new_data}], "version": "1.1", "len": len(new_data)}, f)
print("1")