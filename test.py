import json
from itertools import chain
from datasets import load_from_disk, load_dataset, load_metric

# t = load_metric("metrics/squad")

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