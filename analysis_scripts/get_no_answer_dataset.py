import json
import random
import argparse

def get_no_answer_samples(data):
    new_data = []
    question_paragraph_idx = []
    questions = []
    for di, doc in enumerate(data):
        for pi, para in enumerate(doc["paragraphs"]):
            for qi, qa in enumerate(para["qas"]):
                question_paragraph_idx.append([di, pi, qi])
                questions.append(qa["question"])
                data[di]["paragraphs"][pi]["qas"][qi]["is_impossible"] = False
    for di, doc in enumerate(data):
        new_doc = {"title": doc["title"], "paragraphs": []}
        for pi, para in enumerate(doc["paragraphs"]):
            new_qas = []
            for qi, qa in enumerate(para["qas"]):
                selected_idx = random.choice(list(range(len(question_paragraph_idx))))
                while question_paragraph_idx[selected_idx][0] == di and question_paragraph_idx[selected_idx][1] == pi:
                    selected_idx = random.choice(list(range(len(question_paragraph_idx))))
                question = questions[selected_idx]
                new_qas.append({"question": question, "answers": [], "id": qa["id"] + "-neg", "is_impossible": True})
            new_doc["paragraphs"].append({"context": para["context"], "qas": new_qas})
        new_data.append(new_doc)
    new_data = data + new_data
    return new_data

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_train_dataset",
    type=str,
    help="The training dataset path to get the training data for the training of answer-determining model "
)
parser.add_argument(
    "--input_dev_dataset",
    type=str,
    help="The dev dataset path to get the training data for the dev of answer-determining model "
)
args = parser.parse_args()

with open(args.input_train_file, "r", encoding="utf-8") as f:
    train_data = json.load(f)
with open(args.input_dev_file, "r", encoding="utf-8") as f:
    dev_data = json.load(f)
new_train_data = get_no_answer_samples(train_data["data"])
new_dev_data = get_no_answer_samples(dev_data["data"])
with open(args.input_train_file + "_no_answer", "w", encoding="utf-8") as f:
    json.dump({"data": new_train_data, "version": train_data["version"], "len": train_data["len"] * 2}, f)
with open(args.input_train_file + "_no_answer", "w", encoding="utf-8") as f:
    json.dump({"data": new_dev_data, "version": dev_data["version"], "len": dev_data["len"] * 2}, f)
