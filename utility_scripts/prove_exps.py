import json
import nltk
from tqdm import tqdm
import spacy
import random

from copy import deepcopy

TOKEN_POS = ["VERB", "NOUN", "ADJ", "ADV"]

def add_gold_sent(samples):
    miss_cnt = 0
    for i, sample in enumerate(tqdm(samples, desc="get gold sent")):
        answer_starts = [a["answer_start"] for a in sample["answers"]]
        answer_ends = [a["answer_start"] + len(a["text"]) for a in sample["answers"]]
        sent_hits = [0] * len(sample["context_sents"])
        sent_correspond_answer_index = [[] for _ in range(len(sample["context_sents"]))]
        for j, offset in enumerate(sample["context_sents_offsets"]):
            answer_hits = 0
            for a_i in range(len(answer_starts)):
                if answer_starts[a_i] >= offset[0] and answer_ends[a_i] <= offset[1]:
                    answer_hits += 1
                    sent_correspond_answer_index[j].append(a_i)
            sent_hits[j] = answer_hits
        sorted_sent_index = sorted(range(len(sent_hits)), key=lambda k: sent_hits[k], reverse=True)
        if sent_hits[sorted_sent_index[0]] > 0:
            gold_sent = sample["context_sents"][sorted_sent_index[0]]
            gold_sent_offset = sample["context_sents_offsets"][sorted_sent_index[0]]
            selected_answers = [sample["answers"][a_i] for a_i in
                                sent_correspond_answer_index[sorted_sent_index[0]]]
            best_answer = max(selected_answers, key=selected_answers.count)
            best_answer["answer_end"] = best_answer["answer_start"] + len(best_answer["text"])
            answer_start_in_gold = best_answer["answer_start"] - gold_sent_offset[0]
            new_start, new_end = best_answer["answer_start"] - gold_sent_offset[0], best_answer["answer_end"] - \
                                 gold_sent_offset[0]
            if gold_sent[new_start:new_end] != best_answer["text"]:
                print("Error when obtaining gold sent")
                sample["best_answer"] = best_answer
                samples[i] = sample
                miss_cnt += 1
            else:
                # assert gold_sent[new_start:new_end] == best_answer["text"]
                sample["gold_sent"] = sample["context_sents"][sorted_sent_index[0]]
                sample["gold_sent_index"] = sorted_sent_index[0]
                sample["best_answer"] = best_answer
                sample["gold_sent_front_offset"] = sample["context_sents_offsets"][sample["gold_sent_index"]][0]
                samples[i] = sample
        else:
            best_answer = max(sample["answers"], key=selected_answers.count)
            best_answer["answer_end"] = best_answer["answer_start"] + len(best_answer["text"])
            sample["best_answer"] = best_answer
            samples[i] = sample
            miss_cnt += 1
    return samples, miss_cnt

def add_overlap_tokens(samples):
    for i, sample in enumerate(samples):
        if sample.__contains__("gold_sent"):
            q_tokens = nltk.word_tokenize(sample["question"])
            s_tokens = nltk.word_tokenize(sample["gold_sent"])
            overlapped = set(q_tokens) & set(s_tokens)
            samples[i]["adv_question"] = " ".join(list(overlapped))
    return samples

def remove_entity(text, nlp_spacy):
    spacy_text = list(nlp_spacy.pipe([text]))[0]
    new_question = []
    remove_cnt = 0
    for token in spacy_text:
        if token.ent_type_ == "":
            new_question.append(token.text)
        else:
            remove_cnt += 1
    if remove_cnt > 0:
        return " ".join(new_question)
    return None

def token_perturbation(question):
    tokens = nltk.word_tokenize(question)
    random.shuffle(tokens)
    return " ".join(tokens)

def perturb_context_tokens(samples, data):
    new_context_dict = {}
    for i, sample in enumerate(samples):
        if sample.__contains__("gold_sent"):
            gold_sent = sample["gold_sent"]
            tokens_before_answer = nltk.word_tokenize(
                gold_sent[:sample["best_answer"]["answer_start"] - sample["gold_sent_front_offset"]])
            tokens_after_answer = nltk.word_tokenize(
                gold_sent[sample["best_answer"]["answer_end"] - sample["gold_sent_front_offset"]:])
            tokens = tokens_before_answer + tokens_after_answer + [sample["best_answer"]["text"]]
            random.shuffle(tokens)
            new_gold_sent = " ".join(tokens)
            sample["context_sents"][sample["gold_sent_index"]] = new_gold_sent
            new_context_dict[sample["id"]] = " ".join(sample["context_sents"])
    new_data = []
    for di, doc in enumerate(data):
        new_doc = {"title": doc["title"], "paragraphs": []}
        for pi, para in enumerate(doc["paragraphs"]):
            for qa in para["qas"]:
                if new_context_dict.__contains__(qa["id"]):
                    new_para = {"context": new_context_dict[qa["id"]], "qas": [qa]}
                    new_doc["paragraphs"].append(new_para)
        if len(new_doc["paragraphs"]) > 0:
            new_data.append(new_doc)
    return new_data, len(new_context_dict)

def remove_entities(samples, data, remove_question=False, remove_gold_sent=True, remain=False):
    nlp_spacy = spacy.load("en_core_web_md")
    new_context_dict = {}
    new_question_dict = {}
    pos_tag_num = {}
    for i, sample in enumerate(tqdm(samples)):
        if sample.__contains__("gold_sent"):
            spacy_question = list(nlp_spacy.pipe([sample["question"]]))[0]
            if len(spacy_question.ents) > 0:
                question_entities = []
                for ent in spacy_question.ents:
                    question_entities.append(ent.text.lower())
                if remove_question:
                    tokens = []
                    for token in spacy_question:
                        if token.ent_type_ == "":
                            tokens.append(token.text)
                    new_question_dict[sample["id"]] = " ".join(tokens)
                else:
                    new_question_dict[sample["id"]] = sample["question"]
                if remove_gold_sent:
                    spacy_context = list(nlp_spacy.pipe([sample["gold_sent"]]))[0]
                    new_context = sample["gold_sent"]
                    best_answer_pos = [sample["best_answer"]["answer_start"] - sample["gold_sent_front_offset"],
                                       sample["best_answer"]["answer_end"] - sample["gold_sent_front_offset"]]
                else:
                    spacy_context = list(nlp_spacy.pipe([sample["context"]]))[0]
                    new_context = sample["context"]
                    best_answer_pos = [sample["best_answer"]["answer_start"], sample["best_answer"]["answer_end"]]
                remove_spans = []
                for ent in spacy_context.ents:
                    for text in question_entities:
                        if ent.text.lower() in text:
                            remove_spans.append([ent.start_char, ent.end_char])
                            break
                offset = 0
                if len(remove_spans) == 0:
                    continue
                if not remain:
                    for span in remove_spans:
                        if span[0] >= best_answer_pos[1] or span[1] < best_answer_pos[1] <= best_answer_pos[0]:
                            new_context = new_context[:span[0] - offset] + " " + new_context[span[1] - offset:]
                            offset += (span[1] - span[0])
                else:
                    front_context, read_context = [], []
                    for span in remove_spans:
                        if span[1] <= best_answer_pos[0]:
                            front_context.append(new_context[span[0]: span[1]])
                        if span[0] >= best_answer_pos[1]:
                            read_context.append(new_context[span[0]: span[1]])
                    new_context = " ".join(front_context + [sample["best_answer"]["text"]] + read_context) + "."
                if remove_gold_sent:
                    new_context_sents = sample["context_sents"]
                    new_context_sents[sample["gold_sent_index"]] = new_context
                    new_context_dict[sample["id"]] = " ".join(new_context_sents)
                else:
                    new_context_dict[sample["id"]] = new_context

    new_data = []
    for di, doc in enumerate(data):
        new_doc = {"title": doc["title"], "paragraphs": []}
        for pi, para in enumerate(doc["paragraphs"]):
            for qa in para["qas"]:
                if new_context_dict.__contains__(qa["id"]):
                    qa["question"] = new_question_dict[qa["id"]]
                    new_para = {"context": new_context_dict[qa["id"]], "qas": [qa]}
                    new_doc["paragraphs"].append(new_para)
        if len(new_doc["paragraphs"]) > 0:
            new_data.append(new_doc)
    return new_data, len(new_context_dict)


def remove_tokens(samples, data, remove_question=False, remove_context=False, remain=False, remove_gold_sent=True):
    nlp_spacy = spacy.load("en_core_web_md")
    new_context_dict = {}
    new_question_dict = {}
    pos_tag_num = {}
    for i, sample in enumerate(tqdm(samples)):
        if sample.__contains__("gold_sent"):
            spacy_question = list(nlp_spacy.pipe([sample["question"]]))[0]
            # for token in spacy_question:
            #     if not pos_tag_num.__contains__(token.pos_):
            #         pos_tag_num[token.pos_] = 0
            #     pos_tag_num[token.pos_] += 1
            remove_tokens = []
            best_answer_pos = [sample["best_answer"]["answer_start"] - sample["gold_sent_front_offset"],
                               sample["best_answer"]["answer_end"] - sample["gold_sent_front_offset"]]
            for token in spacy_question:
                if token.pos_ not in TOKEN_POS and token.ent_type_ == "":
                    remove_tokens.append(token.lemma_)
            # for token in spacy_question:
            #     if token.pos_ in TOKEN_POS and token.ent_type_ == "":
            #         remove_tokens.append(token.lemma_)
            if len(remove_tokens) == 0:
                continue
            spacy_context = list(nlp_spacy.pipe([sample["gold_sent"]]))[0]
            remove_spans = []
            for token in spacy_context:
                if token.lemma_ in remove_tokens:
                    remove_spans.append([token.idx, token.idx + len(token)])
            if len(remove_spans) == 0:
                continue
            offset = 0
            new_context = sample["gold_sent"]
            if not remain:
                for span in remove_spans:
                    if span[0] >= best_answer_pos[1] or span[1] < best_answer_pos[1] <= best_answer_pos[0]:
                        new_context = new_context[:span[0] - offset] + new_context[span[1] - offset:]
                        offset += (span[1] - span[0])
            else:
                front_context, read_context = [], []
                for span in remove_spans:
                    if span[1] <= best_answer_pos[0]:
                        front_context.append(new_context[span[0]: span[1]])
                    if span[0] >= best_answer_pos[1]:
                        read_context.append(new_context[span[0]: span[1]])
                new_context = " ".join(front_context + [sample["best_answer"]["text"]] + read_context) + "."

            if remove_gold_sent:
                new_context_sents = sample["context_sents"]
                new_context_sents[sample["gold_sent_index"]] = new_context
                new_context_dict[sample["id"]] = " ".join(new_context_sents)
            else:
                new_context_dict[sample["id"]] = new_context

    new_data = []
    for di, doc in enumerate(data):
        new_doc = {"title": doc["title"], "paragraphs": []}
        for pi, para in enumerate(doc["paragraphs"]):
            for qa in para["qas"]:
                if new_context_dict.__contains__(qa["id"]):
                    # qa["question"] = new_question_dict[qa["id"]]
                    new_para = {"context": new_context_dict[qa["id"]], "qas": [qa]}
                    new_doc["paragraphs"].append(new_para)
        if len(new_doc["paragraphs"]) > 0:
            new_data.append(new_doc)
    return new_data, len(new_context_dict)

            # overlap_lex_tokens, overlap_func_tokens = [], []
            # for token in spacy_question:
            #     if token.pos_ not in TOKEN_POS and token.ent_type_ != "":
            #         remove_tokens.append(token.lemma_)
            #     else:
            #         tokens.append(token.text)
            # if len(remove_tokens) == 0:
            #     continue
            # new_question_dict[sample["id"]] = " ".join(tokens)
            # spacy_context = list(nlp_spacy.pipe([sample["gold_sent"]]))[0]
            # tokens = []
            # remove_cnt = 0
            # for token in spacy_context:
            #     if token.lemma_ not in remove_tokens:
            #         tokens.append(token.text)
            #     else:
            #         remove_cnt += 1
            # if remove_cnt > 0:
            #     new_context_dict[sample["id"]] = " ".join(tokens)

    # new_data_question, new_data_context, new_data_both = [], [], []
    # for di, doc in enumerate(data):
    #     new_doc_question = {"title": doc["title"], "paragraphs": []}
    #     new_doc_context = {"title": doc["title"], "paragraphs": []}
    #     new_doc_both = {"title": doc["title"], "paragraphs": []}
    #     for pi, para in enumerate(doc["paragraphs"]):
    #         new_qas = []
    #         for qa in para["qas"]:
    #             if new_context_dict.__contains__(qa["id"]):
    #                 new_qa = deepcopy(qa)
    #                 new_qa["question"] = new_question_dict[qa["id"]]
    #                 new_para = {"context": new_context_dict[qa["id"]], "qas": [new_qa]}
    #                 new_doc_both["paragraphs"].append(new_para)
    #                 new_context_para = {"context": new_context_dict[qa["id"]], "qas": [qa]}
    #                 new_doc_context["paragraphs"].append(new_context_para)
    #             if new_question_dict.__contains__(qa["id"]):
    #                 new_qa = deepcopy(qa)
    #                 new_qa["question"] = new_question_dict[qa["id"]]
    #                 new_qas.append(new_qa)
    #         if len(new_qas) > 0:
    #             new_doc_question["paragraphs"].append({"context": para["context"], "qas": new_qas})
    #     if len(new_doc_both["paragraphs"]) > 0:
    #         new_data_both.append(new_doc_both)
    #     if len(new_doc_context["paragraphs"]) > 0:
    #         new_data_context.append(new_doc_context)
    #     new_data_question.append(new_doc_question)
    # return new_data_both, new_data_context, new_data_question, len(new_context_dict), len(new_question_dict)

def remain_gold_answer(samples, data):
    new_context_dict = {}
    for i, sample in enumerate(tqdm(samples)):
        if sample.__contains__("gold_sent"):
            new_context_dict[sample["id"]] = sample["gold_sent"]

    new_data = []
    for di, doc in enumerate(data):
        new_doc = {"title": doc["title"], "paragraphs": []}
        for pi, para in enumerate(doc["paragraphs"]):
            for qa in para["qas"]:
                if new_context_dict.__contains__(qa["id"]):
                    new_para = {"context": new_context_dict[qa["id"]], "qas": [qa]}
                    new_doc["paragraphs"].append(new_para)
        if len(new_doc["paragraphs"]) > 0:
            new_data.append(new_doc)
    return new_data, len(new_context_dict)

if __name__ == "__main__":
    with open("../data/squad/dev-v1.1.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    data = raw_data["data"]

    # nlp_spacy = spacy.load("en_core_web_md")
    # sample_cnt = 0
    # for d_i, doc in tqdm(enumerate(data)):
    #     for p_i, paragraph in enumerate(doc["paragraphs"]):
    #         new_qas = []
    #         for q_i, qa in enumerate(paragraph["qas"]):
    #             # new_question = remove_entity(qa["question"], nlp_spacy)
    #             # new_question = token_perturbation(qa["question"])
    #             if new_question is not None:
    #                 qa["question"] = new_question
    #                 new_qas.append(qa)
    #         sample_cnt += len(new_qas)
    #         data[d_i]["paragraphs"][p_i]["qas"] = new_qas
    # with open("perturb_token_question_dev.json", "w", encoding="utf-8") as f:
    #     json.dump({"data": data, "len": sample_cnt, "version": "1.1"}, f)

    samples = []
    for d_i, doc in enumerate(data):
        for p_i, paragraph in enumerate(doc["paragraphs"]):
            context = paragraph["context"]
            context_sents = nltk.sent_tokenize(context)
            sents_offsets = []
            index, sent_index = 0, 0
            while sent_index != len(context_sents):
                while context[index: index + len(context_sents[sent_index])] != context_sents[sent_index]:
                    index += 1
                sents_offsets.append([index, index + len(context_sents[sent_index])])
                index += len(context_sents[sent_index])
                sent_index += 1
            if len(sents_offsets) == 0:
                print("1")
            for qas in paragraph["qas"]:
                samples.append({"context": paragraph["context"],
                                "context_sents": context_sents,
                                "context_sents_offsets": sents_offsets,
                                "question": qas["question"].strip(),
                                "answers": qas["answers"],
                                "id": qas["id"],
                                "title": doc["title"]})


    samples, _ = add_gold_sent(samples)
    new_data, data_size = remove_tokens(samples, data)
    # new_data, data_size = remove_entities(samples, data)
    # new_data, data_size = remain_gold_answer(samples, data)
    with open("../data/squad_adv/remove_func_token_dev.json", "w", encoding="utf-8") as f:
        json.dump({"data": new_data, "len": data_size, "version": "1.1"}, f)


    # new_data_both, new_data_context, new_data_question, context_size, question_size = remove_tokens(samples, data)
    # # new_data, size = remove_entities(samples, data)
    # # new_data, size = perturb_context_tokens(samples, data)
    # with open("remove_token_context_question_dev.json", "w", encoding="utf-8") as f:
    #     json.dump({"data": new_data_both, "len": context_size, "version": "1.1"}, f)
    # with open("remove_token_context_dev.json", "w", encoding="utf-8") as f:
    #     json.dump({"data": new_data_context, "len": context_size, "version": "1.1"}, f)
    # with open("remove_token_question_dev.json", "w", encoding="utf-8") as f:
    #     json.dump({"data": new_data_question, "len": question_size, "version": "1.1"}, f)

    # samples = add_overlap_tokens(samples)
    # adv_q_dict = {}
    # for sample in samples:
    #     if sample.__contains__("adv_question"):
    #         adv_q_dict[sample["id"]] = sample["adv_question"]
    # cnt = 0
    # for d_i, doc in enumerate(data):
    #     new_paragraphs = []
    #     for p_i, paragraph in enumerate(doc["paragraphs"]):
    #         new_qas = []
    #         for qa in paragraph["qas"]:
    #             if adv_q_dict.__contains__(qa["id"]):
    #                 qa["question"] = adv_q_dict[qa["id"]]
    #                 new_qas.append(qa)
    #         if len(new_qas) > 0:
    #             paragraph["qas"] = new_qas
    #             new_paragraphs.append(paragraph)
    #             cnt += len(new_qas)
    #     data[d_i]["paragraphs"] = new_paragraphs

    print("1")