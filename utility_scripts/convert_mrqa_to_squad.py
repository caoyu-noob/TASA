from tqdm import tqdm

import jsonlines
import gzip
import json

def cal_sentence_overlap(question_tokens, sentence_tokens):
    sentence_set = set([token.lower() for token in sentence_tokens])
    overlap_cnt = 0
    for token in question_tokens:
        if sentence_set.__contains__(token[0]):
            overlap_cnt += 1
    return overlap_cnt

def find_highest_overlap(sentence_token_index, answer_token_spans, question_tokens, context_tokens):
    max_over_lap = 0
    index = 0
    for i in range(len(answer_token_spans)):
        for sentence in sentence_token_index:
            if sentence[0] <= answer_token_spans[i][0] and sentence[1] >= answer_token_spans[i][1]:
                sentence_tokens = [t[0] for t in context_tokens[sentence[0]:sentence[1] + 1]]
                overlap_cnt = cal_sentence_overlap(question_tokens, sentence_tokens)
                if overlap_cnt > max_over_lap:
                    max_over_lap = overlap_cnt
                    index = i
                break
    return index, max_over_lap

def split_sentence(context_tokens):
    sentence_index = []
    start_index = 0
    for i in range(0, len(context_tokens)):
        if context_tokens[i][0] == ".":
            sentence_index.append([start_index, i])
            start_index = i + 1
    if start_index < len(context_tokens) - 1:
        sentence_index.append([start_index, len(context_tokens) - 1])
    return sentence_index

def convert_mrqa_to_squad(input_file_name):
    data_list = []
    sample_num = 0
    with gzip.open(input_file_name, "r") as f:
        jsonl_tqdm = tqdm(jsonlines.Reader(f))
        for l in jsonl_tqdm:
            if l.__contains__("qas"):
                paragraph = {}
                paragraph["context"] = l["context"]
                paragraph["qas"] = []
                for qa in l["qas"]:
                    if len(qa["detected_answers"][0]["char_spans"]) == 1:
                        answer_start = qa["detected_answers"][0]["char_spans"][0][0]
                        text = l["context"][answer_start: answer_start + len(qa["detected_answers"][0]["text"])]
                        answers = [{"answer_start": answer_start, "text": text}]
                    else:
                        sentence_token_index = split_sentence(l["context_tokens"])
                        best_answer_index, _ = find_highest_overlap(sentence_token_index,
                                                                    qa["detected_answers"][0]["token_spans"],
                                                                    qa["question_tokens"],
                                                                    l["context_tokens"])
                        answer_start = qa["detected_answers"][0]["char_spans"][best_answer_index][0]
                        text = l["context"][answer_start: answer_start + len(qa["detected_answers"][0]["text"])]
                        answers = [{"answer_start": answer_start, "text": text}]
                    qas_item = {"question": qa["question"], "id": qa["id"], "answers": answers}
                    paragraph["qas"].append(qas_item)
                    sample_num += 1
                data_list.append(paragraph)
    # for data in data_list:
    #     for qa in data["qas"]:
    #         for answer in qa["answers"]:
    #             answer_start = answer["answer_start"]
    #             text = answer["text"]
    #             assert data["context"][answer_start: answer_start + len(text)].lower() == text.lower()
    return data_list, sample_num

if __name__ == "__main__":
    input_file_name = "../corpus/HotpotQA-train.jsonl.gz"
    output_file_name = "../corpus/hotpot_train.json"
    data_list, sample_num = convert_mrqa_to_squad(input_file_name)
    with open(output_file_name, "w", encoding="utf-8") as f:
        json.dump({"data": [{"title": "all", "paragraphs": data_list}], "len": sample_num, "version": "1.0"}, f)