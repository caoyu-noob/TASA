import json
import pickle
import re
import sys
import traceback
from PWWS.utils_PWWS import ReplaceOp
# from itertools import chain
# from datasets import load_from_disk, load_dataset, load_metric
from transformers import AutoTokenizer
# import spacy_alignments
# import multiprocessing as mp
# # from nltk.corpus import wordnet
# import spacy
# import torch
from datasets import load_metric
import nltk
from tqdm import tqdm

# with open("test.json", "r", encoding="utf-8") as f:
#     test_data = json.load(f)
# source, source1 = 'Tesla', 'Nikola'
# target = ''
# target1 = ''
# for i, p in enumerate(test_data['data'][0]['paragraphs']):
#     context = p['context']
#     context = context.replace(source, target)
#     context = context.replace(source.lower(), target)
#     context = context.replace(source1, target1)
#     context = context.replace(source1.lower(), target1)
#     test_data['data'][0]['paragraphs'][i]['context'] = context



with open("data/test/test.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)
source, source1 = 'Tesla', 'Nikola'
target = ''
target1 = ''
for i, p in enumerate(test_data['data'][0]['paragraphs']):
    for j, q in enumerate(p['qas']):
        question = q['question']
        question = question.replace(source, target)
        question = question.replace(source.lower(), target)
        question = question.replace(source1, target1)
        question = question.replace(source1.lower(), target1)
        test_data['data'][0]['paragraphs'][i]['qas'][j]['question'] = question

with open("data/test/test.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f)

with open("corefs.json", "r", encoding="utf-8") as f:
    corefs = json.load(f)
pron_indices = []
for i, coref in enumerate(tqdm(corefs["corefs"])):
    cur_prons = []
    if len(coref) > 0:
        for j, c in enumerate(coref):
            pos_tags = [nltk.pos_tag(nltk.tokenize.word_tokenize(sub_coref[0])) for sub_coref in c]
            is_pron, no_pron = False, False
            for p in pos_tags:
                for sub_p in p:
                    if sub_p[1] in ["PRP", "PRP$", "WP", "WP$"]:
                        is_pron = True
                    else:
                        no_pron = True
            if is_pron and no_pron:
                cur_prons.append(c)
    pron_indices.append(cur_prons)


with open("dev-v1.1-entity_question_replace.json", "r", encoding="utf-8") as f:
    d = json.load(f)
answers = []
for qa in d["data"][0]["paragraphs"]:
    answer_start = [a["answer_start"] for a in qa["qas"][0]["answers"]]
    answer_text = [a["text"] for a in qa["qas"][0]["answers"]]
    answers.append({"answers": {"answer_start":answer_start, "text": answer_text}, "id": qa["qas"][0]["id"]})
with open("test_predictions_question.json", "r", encoding="utf-8") as f:
    pred = json.load(f)
predictions = []
for k, v in pred.items():
    predictions.append({"id": k, "prediction_text": v})

metric = load_metric("./metrics/squad")
metrics = metric.compute(predictions=predictions, references=answers)

em_cnt, cnt = 0, 0
for k, v in pred.items():
    answer = answer_dict[k]
    for a in answer:
        if v == a:
            em_cnt += 1
    cnt += len(answer)

tokenizer = AutoTokenizer.from_pretrained("debug_squad_roberta")

with open("PWWS_nbest_ops.pickle", "rb") as f:
    ops = pickle.load(f)
with open("test_predictions_attack.json", "r") as f:
    attack_predictions = json.load(f)
with open("test_predictions.json", "r") as f:
    test_predictions = json.load(f)
with open("data/squad/dev-v1.1.json", "r") as f:
    data = json.load(f)
answers = {}
# for doc in data:
#     for paragraph in doc["paragraphs"]:
#         for qa in paragraph["qas"]:
#             question_context_index.append(c_i)
#             questions.append(qa["question"])
ids = []
for k, v in attack_predictions.items():
    orig_id = k.split("-")[0]
    if test_predictions[orig_id] != v:
        ids.append(orig_id)
with open("PWWS_nbest.json", "r") as f:
    attack_data = json.load(f)
with open("PWWS_nbest_ops.pickle", "rb") as f:
    ops = pickle.load(f)

op_cnt, ner_cnt, word_cnt = 0, 0, 0
origs = []
words = []
for op in ops[0]:
    for sub_op in op:
        if sub_op.original_text.lower() != sub_op.original_text:
            ner_cnt += 1
        else:
            if bool(re.search(r'\d', sub_op.original_text)):
                ner_cnt += 1
            else:
                word_cnt += 1
    op_cnt += len(op)

with open("data/squad/dev-v1.1.json", "r") as f:
    d = json.load(f)
print(d["data"][0]["paragraphs"][0]["context"])

with open("PWWS_nbest.json", "r") as f:
    data = json.load(f)

with open("adv_nbest.json", "r") as f:
    data = json.load(f)

with open("data/squad/dev-v1.1.json", "r") as f:
    d1 = json.load(f)

with open("PWWS_adversarial_samples.json", "r", encoding="utf-8") as f:
    d=json.load(f)
for i, p in enumerate(d['data'][0]['paragraphs']):
    answers = d['data'][0]['paragraphs'][i]['qas'][0]["answers"]
    answers = [{"answer_start": a["answer_start"], "text": a["answer_text"]} for a in answers]
    d['data'][0]['paragraphs'][i]['qas'][0]["answers"] = answers
    # d['data'][0]['paragraphs'][i]['qas'] = [d['data'][0]['paragraphs'][i]['qas']]
#     d['data'][0]['paragraphs'][i]['qas'] = [d['data'][0]['paragraphs'][i]['qas']]
with open("PWWS_adversarial_samples.json", "w", encoding="utf-8") as f:
    json.dump(d, f)
# a = spacy_alignments.get_alignments(["å", "BC"], ["abc"])
nlp = spacy.load("en_core_web_md")
# with open("data/squad/train-v1.1.json", "r", encoding="utf-8") as f:
#     data = json.load(f)["data"]
with open("data/squad/dev-v1.1.json", "r", encoding="utf-8") as f:
    data = json.load(f)["data"]
contexts, questions = [], []
c_i = 0
context_results = list(nlp.pipe(contexts, batch_size=256, n_process=1))
question_results = list(nlp.pipe(questions, batch_size=256, n_process=1))
a = []
for x in context_results:
    a.extend(x)
tokenizer = AutoTokenizer.from_pretrained("../bert_base_uncased")
tokenized_contexts = tokenizer(contexts)
tokenized_questions = tokenizer(questions)
align = []
c_cnt, q_cnt = 0, 0
for i in range(len(context_results)):
    if len(context_results[i]) != tokenized_contexts.encodings[i].words[-2] + 1:
        c_cnt += 1
for i in range(len(question_results)):
    if len(question_results[i]) != tokenized_questions.encodings[i].words[-2] + 1:
        q_cnt += 1
context_results = [[t.text for t in c] for c in context_results]
question_results = [[t.text for t in q] for q in question_results]
tokenized_contexts = [c.tokens[1: -1] for c in tokenized_contexts.encodings]
tokenized_questions = [q.tokens[1: -1] for q in tokenized_questions.encodings]
context_spacy_to_bert, context_bert_to_spacy = [], []
for x, y in zip(context_results, tokenized_contexts):
    spacy_to_bert, bert_to_spacy = spacy_alignments.get_alignments(x, y)
    context_spacy_to_bert.append(spacy_to_bert)
    context_bert_to_spacy.append(bert_to_spacy)
question_spacy_to_bert, question_bert_to_spacy = [], []
for x, y in zip(question_results, tokenized_questions):
    spacy_to_bert, bert_to_spacy = spacy_alignments.get_alignments(x, y)
    question_spacy_to_bert.append(spacy_to_bert)
    question_bert_to_spacy.append(bert_to_spacy)

with open("spacy_data.pickle", "wb") as f:
    pickle.dump({"contexts": context_results, "questions": question_results, "index": question_context_index}, f)

ent_dict = {}
for r in results:
    if len(r.ents) > 0:
        for e in r.ents:
            if not ent_dict.__contains__(e.label_):
                ent_dict[e.label_] = set()
            ent_dict[e.label_].add(e.text)
#
# wordnet.synsets("apple")
# t = load_metric("metrics/squad")
with open("extract/extract.json", "r") as f:
    data = json.load(f)
d = load_dataset("data/squad/", ignore_verifications=True, cache_dir="./cache")
dataset = load_dataset('data/squad/squad.py', '')

with open('data/prediction.json', 'r') as f:
    dd = json.load(f)
with open('data/AddSent/AddSent.json', 'r') as f:
    d = json.load(f)
with open('data/AddOneSent.json', 'r') as f:
    d1 = json.load(f)
d = list(chain(*[s['paragraphs'] for s in [x for x in d['data']]]))
d1 = list(chain(*[s['paragraphs'] for s in [x for x in d1['data']]]))
d1_filter = []
for x in d1:
    cur_qas = []
    for qa in x['qas']:
        if 'turk' in qa['id']:
            cur_qas.append(qa)
    if len(cur_qas) > 0:
        x['qas'] = cur_qas
        d1_filter.append(x)
print('111')