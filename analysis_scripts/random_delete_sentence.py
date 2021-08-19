import json
import random
from copy import deepcopy

def split_sentences(context):
    sentences = context.split(". ")
    sentences = [s for s in sentences if len(s) != 0]
    for i in range(len(sentences)):
        if sentences[i][-1] != ".":
            sentences[i] = sentences[i] + "."
    offsets = []
    start = 0
    for i, sentence in enumerate(sentences):
        if i != len(sentences) - 1:
            offsets.append((start, start + len(sentence) + 1))
        else:
            offsets.append((start, start + len(sentence)))
        start += (len(sentence) + 1)
    return offsets

def check_overlap(offset, answer_indices):
    for index in answer_indices:
        if index[1] <= offset[0] or index[0] >= offset[1]:
            return False
    return True

ratio = 0.6
with open("../data/squad/dev-v1.1.json", "r", encoding="utf-8") as f:
    data = json.load(f)
new_data_list = []
cannot_delete_cnt = 0
for doc in data["data"]:
    cur_doc = {"title": doc["title"], "paragraphs": []}
    for paragraph in doc["paragraphs"]:
        sentences_offsets = split_sentences(paragraph["context"])
        for qa in paragraph["qas"]:
            context = deepcopy(paragraph["context"])
            answer_indices = [[x["answer_start"], x["answer_start"] + len(x["text"])] for x in qa["answers"]]
            selected_offsets = []
            for offset in sentences_offsets:
                if not check_overlap(offset, answer_indices):
                    selected_offsets.append(offset)
            if len(selected_offsets) == 0:
                cannot_delete_cnt += 1
            else:
                random.shuffle(selected_offsets)
                selected_offsets = selected_offsets[:int(max(len(selected_offsets) * ratio, 1))]
                selected_offsets = sorted(selected_offsets, key=lambda x: x[0])
                char_offset = 0
                for offset in selected_offsets:
                    context = context[0: offset[0] + char_offset] + context[offset[1] + char_offset:]
                    for i, answer_index in enumerate(answer_indices):
                        if answer_index[0] >= offset[1] + char_offset:
                            answer_indices[i] = [answer_index[0] - (offset[1] - offset[0]), answer_index[1] - (offset[1] - offset[0])]
                    char_offset -= (offset[1] - offset[0])
                answers = [{"answer_start": answer_indices[i][0], "text": qa["answers"][i]["text"]} for i in range(len(answer_indices))]
                cur_doc["paragraphs"].append({"context": context, "qas": [{"question": qa["question"],
                                "answers": answers, "id": qa["id"] + "-del"}]})
    if len(cur_doc["paragraphs"]) != 0:
        new_data_list.append(cur_doc)
with open("dev-v1.1_" + str(ratio) + ".json", "w", encoding="utf-8") as f:
    json.dump({"data": new_data_list, "len": data["len"] - cannot_delete_cnt, "version": "1.1"}, f)
print("111")