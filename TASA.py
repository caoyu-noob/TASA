import copy
import json
import nltk
import os
import spacy
import re
import torch
import pickle
import random
import pyinflect
import numpy as np
import argparse

from tqdm import tqdm
from itertools import chain
from copy import deepcopy
from nltk.corpus import wordnet as wn
from model.evaluator import QAEvaluator
from model.ppl_evaluator import PPLEvaluator
from model.use_evaluator import USEEvaluator
from utils.common_utils import config_logger

REAR_MODE = "rear"
AFTER_GOLD = "after"
EDIT_CANDIDATE_POS = ["VERB", "NOUN", "ADJ", "ADV"]
ANSWER_POS = ["ADV", "ADJ", "NOUN", "VERB", "NUM"]
POS_ALIGN = ["VERB", "NOUN", "ADJ"]
MASK_SYMBOLS = {"bert": "[UNK]", "bidaf": "@@UNKNOWN@@", "spanbert": "[UNK]"}
ADVERSARIAL_MODE_NO = "NO"
ADVERSARIAL_MODE_EDIT = "EDIT"
ADVERSARIAL_MODE_ADD = "ADD"
ADVERSARIAL_MODE_EDIT_AND_ADD = "EDIT+ADD"
NOUN_POS_IN_WORDNET = "noun"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_dataset_file",
        type=str,
        help="The dataset file name of attack target",
    )
    parser.add_argument(
        "--target_dataset_type",
        type=str,
        help="The type of target dataset",
    )
    parser.add_argument(
        "--target_model",
        type=str,
        help="The path of target model",
    )
    parser.add_argument(
        "--target_model_type",
        type=str,
        help="The type for the target model, bert/bidaf",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="The directory for the new json file of adversarial samples",
    )
    parser.add_argument(
        "--ent_dict_file",
        type=str,
        default="ent_dict.json",
        help="The path for named entity dict file used in adding distractor",
    )
    parser.add_argument(
        "--coreference_file",
        type=str,
        default="corefs.json",
        help="The path for file that contains the positions that need to be replaced by coreference ",
    )
    parser.add_argument(
        "--pos_vocab_dict_file",
        type=str,
        default="pos_vocab_dict.json",
        help="The path for the dict of pos tags and vocabulary",
    )
    parser.add_argument(
        "--ppdb_synonym_dict_file",
        type=str,
        default="./auxiliary_data/ppdb_synonym_dict.json",
        help="The path for the processed ppdb synonym dict file",
    )
    parser.add_argument(
        "--raw_ppdb_synonym_dict_file",
        type=str,
        default="./auxiliary_data/ppdb_synonyms.txt",
        help="The path for the raw ppdb synonym text file",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=3,
        help="The beam search size during generating adversarial samples",
    )
    parser.add_argument(
        "--remain_size_per_sample",
        type=int,
        default=1,
        help="The final remaining size in the last step during generating (if there is a possibility for attacking,"
             " how many adversarial samples remaining per original sample",
    )
    parser.add_argument(
        "--ent_noun_sample_num",
        type=int,
        default=20,
        help="The number for sampling named entity when generating distractor",
    )
    parser.add_argument(
        "--max_edit_num",
        type=int,
        default=5,
        help="The maximum number for editing overlapped tokens in the gold sentence",
    )
    parser.add_argument(
        "--max_edit_num_in_add",
        type=int,
        default=3,
        help="The maximum number for editing named entities/nouns to obtain a distractor",
    )
    parser.add_argument(
        "--distractor_insert_mode",
        type=str,
        default="rear",
        help="the place to insert the distractor sentence, rear: the rear of the cotnext, "
             "front: the front of the context, random: insert randomly",
    )

    parser.add_argument(
        "--USE_model_path",
        type=str,
        default="../model_hubs/USE_model",
        help="the path to save the USE model",
    )

    parser.add_argument(
        "--ppl_model_path",
        type=str,
        default="../model_hubs/gpt2_small",
        help="the path to save the model for calculating PPL",
    )

    parser.add_argument(
        "--determine_model_path",
        type=str,
        default="../model_hubs/roberta_base_squad2",
        help="the path to save the model for determining whether an answer exists",
    )

    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=8,
        help="the batch size during evaluating by the target model",
    )

    parser.add_argument(
        "--filter_th",
        type=float,
        default=-2,
        help="the threshold for filtering the edited sentence, value larger than it will be remained",
    )

    parser.add_argument(
        "--beam_search_th",
        type=float,
        default=0.2,
        help="the threshold for filtering the edited sentence, value larger than it will be remained",
    )

    parser.add_argument(
        "--sample_start_index",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--sample_end_index",
        type=int,
        default=-1,
    )
    return parser.parse_args()

class QAAttacker:
    def __init__(self, args, model_evaluator, has_answer_evaluator=None, ppl_evaluator=None, use_evaluator=None,
                 logger=None):
        self.beam_size = args.beam_size
        self.remain_size_per_sample = args.remain_size_per_sample
        self.ent_noun_sample_num = args.ent_noun_sample_num
        self.max_edit_num = args.max_edit_num
        self.max_edit_num_in_add = args.max_edit_num_in_add
        self.distractor_insert_mode = args.distractor_insert_mode
        self.dataset_type = args.target_dataset_type
        self.filter_th = args.filter_th
        self.beam_search_th = args.beam_search_th
        self.spacy_nlp = spacy.load("en_core_web_md")
        self.model_type = args.target_model_type
        self.logger = logger

        '''Load the synonym dict from PPDB'''
        if not os.path.exists(args.ppdb_synonym_dict_file):
            ppdb_synonym_dict = self.get_ppdb_synonyms(args.raw_ppdb_synonym_dict_file, self.spacy_nlp)
            with open(args.ppdb_synonym_dict_file, "w", encoding="utf-8") as f:
                json.dump(ppdb_synonym_dict, f)
        with open(args.ppdb_synonym_dict_file, "r", encoding="utf-8") as f:
            ppdb_synonym_dict = json.load(f)
        self.ppdb_synonym_dict = ppdb_synonym_dict

        '''Load the named entity dict from json file'''
        with open(args.ent_dict_file, "r", encoding="utf-8") as f:
            self.ent_dict = json.load(f)
        '''Load the POS tag and vocab dict from json file'''
        with open(args.pos_vocab_dict_file, "r", encoding="utf-8") as f:
            self.pos_vocab_dict = json.load(f)

        self.model_evaluator = model_evaluator
        self.has_answer_evaluator = has_answer_evaluator
        self.ppl_evaluator = ppl_evaluator
        self.use_evaluator = use_evaluator

    def get_ppdb_synonyms(self, ppdb_file_path, spacy_nlp):
        with open(ppdb_file_path, "r", encoding="utf-8") as f:
            ppdb_synonyms_lines = f.readlines()
        ppdb_synonyms_dict = {}
        for line in tqdm(ppdb_synonyms_lines, desc="get ppdb synonyms"):
            pair = line.strip().split()
            pair = [s[0].lemma_ for s in spacy_nlp.pipe(pair)]
            if pair[0] != pair[1]:
                if not ppdb_synonyms_dict.__contains__(pair[0]):
                    ppdb_synonyms_dict[pair[0]] = set()
                ppdb_synonyms_dict[pair[0]].add(pair[1])
                if not ppdb_synonyms_dict.__contains__(pair[1]):
                    ppdb_synonyms_dict[pair[1]] = set()
                ppdb_synonyms_dict[pair[1]].add(pair[0])
        for k, v in ppdb_synonyms_dict.items():
            ppdb_synonyms_dict[k] = list(v)
        return ppdb_synonyms_dict

    def add_gold_sent(self, samples):
        miss_cnt = 0
        for i, sample in enumerate(tqdm(samples, desc="get gold sent")):
            answer_starts = [a["answer_start"] for a in sample["answers"]]
            sample["adversarial_mode"] = ADVERSARIAL_MODE_NO
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
                if len(sample["coreferences"]) > 0:
                    all_corefs = []
                    for coreference in sample["coreferences"]:
                        for target in coreference["targets"]:
                            if target[1] >= gold_sent_offset[0] and target[2] <= gold_sent_offset[1]:
                                all_corefs.append([coreference["text"],
                                                   target[1] - gold_sent_offset[0], target[2] - gold_sent_offset[0]])
                    if len(all_corefs) > 0:
                        offset = 0
                        all_corefs = sorted(all_corefs, key=lambda x: x[1])
                        for coref in all_corefs:
                            gold_sent = gold_sent[:coref[1] + offset] + coref[0] + gold_sent[coref[2] + offset:]
                            offset += len(coref[0]) - (coref[2] - coref[1])
                            if coref[2] <= answer_start_in_gold:
                                best_answer["answer_start"] += len(coref[0]) - (coref[2] - coref[1])
                                best_answer["answer_end"] += len(coref[0]) - (coref[2] - coref[1])
                new_start, new_end = best_answer["answer_start"] - gold_sent_offset[0], best_answer["answer_end"] - gold_sent_offset[0]
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

    def add_overlap_token(self, samples):
        spacy_nlp = self.spacy_nlp
        miss_edit_num, miss_add_num = 0, 0
        for i, sample in enumerate(tqdm(samples, desc="add overlap token")):
            if not sample.__contains__("gold_sent"):
                continue
            spacy_question, spacy_gold_sent = spacy_nlp.pipe([sample["question"], sample["gold_sent"]])
            question_ents_text = [ent.text for ent in spacy_question.ents]
            question_nouns_text = [token.lower_ for token in spacy_question if token.pos_ == "NOUN"]
            question_token_candidates_lemma_dict = {}
            for token in spacy_question:
                if token.ent_type == 0 and (not token.is_punct) and (token.pos_ in EDIT_CANDIDATE_POS):
                    if not question_token_candidates_lemma_dict.__contains__(token.lemma_):
                        question_token_candidates_lemma_dict[token.lemma_] = []
                    question_token_candidates_lemma_dict[token.lemma_].append(token)
            edit_candidates = []
            edit_candidates_set = set()
            for token in spacy_gold_sent:
                if question_token_candidates_lemma_dict.__contains__(token.lemma_) and not edit_candidates_set.__contains__(token.lemma_):
                    edit_candidates.append([token, question_token_candidates_lemma_dict[token.lemma_]])
                    edit_candidates_set.add(token.lemma_)
            add_ent_candidates, add_noun_candidates = [], []
            answer_start_in_gold, answer_end_in_gold = \
                sample["best_answer"]["answer_start"] - sample["gold_sent_front_offset"], \
                sample["best_answer"]["answer_end"] - sample["gold_sent_front_offset"]
            for ent in spacy_gold_sent.ents:
                if ent.text in question_ents_text and \
                        (ent.end_char < answer_start_in_gold or ent.start_char > answer_end_in_gold):
                    add_ent_candidates.append(ent)
            if len(add_ent_candidates) > 0:
                add_ent_candidates = sorted(add_ent_candidates, key=lambda x: abs(x.start - answer_start_in_gold))
            for token in spacy_gold_sent:
                if token.lower_ in question_nouns_text and \
                        (token.idx + len(token) < answer_start_in_gold or token.idx > answer_end_in_gold):
                    add_noun_candidates.append(token)
            if len(add_noun_candidates) > 0:
                add_noun_candidates = sorted(add_noun_candidates, key=lambda x: abs(x.idx - answer_start_in_gold))
            samples[i]["edit_candidates"] = edit_candidates
            samples[i]["add_candidates"] = {"ents": add_ent_candidates,
                                            "nouns": add_noun_candidates}
            if len(edit_candidates) == 0:
                miss_edit_num += 1
            if len(add_ent_candidates) == 0 and len(add_noun_candidates) == 0:
                miss_add_num += 1
            samples[i]["spacy_gold_sent"] = spacy_gold_sent
            samples[i]["spacy_question"] = spacy_question
        return samples, miss_edit_num, miss_add_num

    def evaluate_edit_importance(self, samples):
        '''each pos in indices indicates 0: the index of original sample, 1: the index of edit candidate,
        2: the index in masked logits'''
        masked_contexts, indices, original_contexts = [], [], []
        masked_questions, original_questions = [], []
        masked_answer_starts, masked_answer_ends, answer_starts, answer_ends = [], [], [], []
        mask_index = 0
        mask_symbol = MASK_SYMBOLS[self.model_type]
        for i, sample in enumerate(samples):
            context_sents = sample["context_sents"]
            original_contexts.append(" ".join(context_sents))
            original_questions.append(sample["question"])
            answer_starts.append(sample["best_answer"]["answer_start"])
            answer_ends.append(sample["best_answer"]["answer_end"])
            if sample.__contains__("gold_sent"):
                answer_start_char_in_gold = sample["best_answer"]["answer_start"] - sample["gold_sent_front_offset"]
                for j, edit_candidate in enumerate(sample["edit_candidates"]):
                    new_answer_start, new_answer_end = sample["best_answer"]["answer_start"], sample["best_answer"][
                        "answer_end"]
                    if edit_candidate[0].idx < answer_start_char_in_gold:
                        new_answer_start += (len(mask_symbol) - len(edit_candidate[0]))
                        new_answer_end += (len(mask_symbol) - len(edit_candidate[0]))
                    masked_answer_starts.append(new_answer_start)
                    masked_answer_ends.append(new_answer_end)
                    masked_gold_sent = sample["gold_sent"][:edit_candidate[0].idx] + mask_symbol + \
                                       sample["gold_sent"][edit_candidate[0].idx + len(edit_candidate[0]):]
                    masked_contexts.append(" ".join(context_sents[:sample["gold_sent_index"]] + [masked_gold_sent] +
                                                    context_sents[sample["gold_sent_index"] + 1:]))
                    masked_questions.append(sample["question"])
                    indices.append([i, j, mask_index])
                    mask_index += 1
            else:
                indices.append([i, -1, -1])

        original_start_logits, original_end_logits, start_positions, end_positions, best_start_positions, \
                best_end_positions = \
            self.model_evaluator.evaluate_sample_logits(original_contexts,
                original_questions, answer_starts, answer_ends, single_question=False, return_best_position=True,
                                                        desc="Get edit importance")
        masked_start_logits, masked_end_logits = self.model_evaluator.evaluate_sample_logits(masked_contexts,
                masked_questions, masked_answer_starts, masked_answer_ends, single_question=False)
        for i in tqdm(range(len(indices)), desc="Rerank edit candidates"):
            original_index = indices[i][0]
            if not samples[original_index].__contains__("original_logits"):
                samples[original_index]["original_logits"] = [original_start_logits[original_index].item(),
                                                              original_end_logits[original_index].item()]
            samples[original_index]["pred_start_position"] = best_start_positions[original_index].item()
            samples[original_index]["pred_end_position"] = best_end_positions[original_index].item()
            if indices[i][1] != -1:
                edit_candidate_index = indices[i][1]
                mask_index = indices[i][2]
                if not samples[original_index].__contains__("masked_logits"):
                    samples[original_index]["masked_logits"] = \
                        [[0 for _ in range(len(samples[original_index]["edit_candidates"]))],
                         [0 for _ in range(len(samples[original_index]["edit_candidates"]))]]
                samples[original_index]["masked_logits"][0][edit_candidate_index] = masked_start_logits[mask_index].item()
                samples[original_index]["masked_logits"][1][edit_candidate_index] = masked_end_logits[mask_index].item()

                sample = samples[original_index]
                sorted_indices = np.argsort([
                    sample["masked_logits"][0][m] + sample["masked_logits"][1][m] -
                    sample["original_logits"][0] - sample["original_logits"][1]
                    for m in range(len(sample["edit_candidates"]))]).tolist()
                sorted_edit_candidates = []
                for index in sorted_indices:
                    sorted_edit_candidates.append(sample["edit_candidates"][index])
                sample["edit_candidates"] = sorted_edit_candidates
                samples[original_index] = sample
        return samples

    def get_synonyms(self, samples):
        ppdb_synonym_dict = self.ppdb_synonym_dict
        for i, sample in enumerate(tqdm(samples, desc="get edit synonyms")):
            if not sample.__contains__("gold_sent"):
                continue
            synonym_candidates = []
            for candidate in sample["edit_candidates"]:
                cur_replacement = set()
                if ppdb_synonym_dict.__contains__(candidate[0].lemma_):
                    cur_replacement = set(ppdb_synonym_dict[candidate[0].lemma_])
                wordnet_synonyms = set()
                for synonym in wn.synsets(candidate[0].lemma_):
                    for lemma in synonym.lemma_names():
                        if lemma != candidate[0].lemma_:
                            wordnet_synonyms.add(lemma)
                cur_replacement = cur_replacement | wordnet_synonyms
                synonym_candidates.append(list(cur_replacement))
            samples[i]["edit_synonyms"] = synonym_candidates
        return samples

    def replace_text_given_positions(self, string, positions, target, answer_start=-1, answer_end=-1, answer_offset=-1,
                                     all_answer_starts=[]):
        char_offset = 0
        res = deepcopy(string)
        for position in positions:
            res = res[:position[0] + char_offset] + target + res[position[1] + char_offset:]
            tmp_offset = len(target) - (position[1] - position[0])
            char_offset += tmp_offset
            if position[1] < answer_start - answer_offset:
                answer_start += tmp_offset
            if position[1] < answer_end - answer_offset:
                answer_end += tmp_offset
            for i, start in enumerate(all_answer_starts):
                if position[1] < start - answer_offset:
                    all_answer_starts[i] += tmp_offset
        return res, answer_start, answer_end, all_answer_starts

    def _get_inflect_aligned_tokens(self, token_candidates, edit_target):
        aligned_tokens = []
        for synonym in self.spacy_nlp.pipe(token_candidates):
            success_cnt = 0
            aligned_result = []
            for token in synonym:
                aligned_token = token._.inflect(edit_target.tag_)
                if aligned_token is not None:
                    success_cnt += 1
                    aligned_result.append(aligned_token)
                else:
                    aligned_result.append(token.text)
            if success_cnt > 0:
                aligned_tokens.append(" ".join(aligned_result))
        return aligned_tokens

    def get_edited_sentences(self, samples):
        cnt = []
        for i, sample in enumerate(tqdm(samples, desc="get edited samples")):
            if not sample.__contains__("gold_sent"):
                continue
            if len(sample["edit_synonyms"]) == 0:
                sample["edited_gold_sents"] = []
                sample["edited_answer_starts"] = []
                samples[i] = sample
            elif len(sample["edit_synonyms"]) > 0:
                edited_gold_sents = [sample["gold_sent"]]
                answer_start, answer_end = sample["best_answer"]["answer_start"], sample["best_answer"]["answer_end"]
                answer_starts = [answer_start]
                answer_ends = [answer_end]
                all_answer_starts = [[a["answer_start"] for a in sample["answers"]]]
                edit_cnt = 0
                for edit_target, synonyms in zip(sample["edit_candidates"], sample["edit_synonyms"]):
                    if edit_cnt >= self.max_edit_num:
                        break
                    synonyms = [s.replace("_", " ") for s in synonyms]
                    if edit_target[0].pos_ in POS_ALIGN:
                        aligned_synonyms = self._get_inflect_aligned_tokens(synonyms, edit_target[0])
                    else:
                        aligned_synonyms = synonyms
                    # spacy_synonyms = [s[0]._.inflect(edit_target[0].tag_) for s in spacy_nlp.pipe(synonyms)]
                    if len(aligned_synonyms) > 0:
                        answer_offsets = len(" ".join(sample["context_sents"][:sample["gold_sent_index"]])) + 1
                        replaced_gold_sents, replaced_answer_starts, replaced_answer_ends, replaced_all_answer_starts = \
                            [], [], [], []
                        for s_i, gold_sent in enumerate(edited_gold_sents):
                            answer_start, answer_end = answer_starts[s_i], answer_ends[s_i]
                            matched_positions = [m.span(0) for m in re.finditer(edit_target[0].text, gold_sent)]
                            for synonym in aligned_synonyms:
                                if "_" in synonym:
                                    synonym = " ".join(synonym.split("_"))
                                replaced_sent, new_answer_start, new_answer_end, new_all_answer_starts = \
                                    self.replace_text_given_positions(gold_sent, matched_positions, synonym,
                                            answer_start, answer_end, answer_offsets, all_answer_starts[s_i])
                                replaced_gold_sents.append(replaced_sent)
                                replaced_answer_starts.append(new_answer_start)
                                replaced_answer_ends.append(new_answer_end)
                                replaced_all_answer_starts.append(new_all_answer_starts)
                        new_contexts = []
                        for gold_sent in replaced_gold_sents:
                            sample["context_sents"][sample["gold_sent_index"]] = gold_sent
                            new_contexts.append(" ".join(sample["context_sents"]))
                        answer_start_logits, answer_end_logits = self.model_evaluator.evaluate_sample_logits(
                                new_contexts, sample["question"], replaced_answer_starts, replaced_answer_ends)
                        if answer_start_logits.size(0) < self.beam_size:
                            best_indices = list(range(answer_start_logits.size(0)))
                        else:
                            best_indices = torch.topk(answer_start_logits + answer_end_logits, k=self.beam_size,
                                                      largest=False)[1].tolist()
                        edited_gold_sents = [replaced_gold_sents[index] for index in best_indices]
                        answer_starts = [replaced_answer_starts[index] for index in best_indices]
                        answer_ends = [replaced_answer_ends[index] for index in best_indices]
                        all_answer_starts = [replaced_all_answer_starts[index] for index in best_indices]
                        edit_cnt += 1
                        min_beam_score = (answer_start_logits[best_indices[-1]] + answer_end_logits[best_indices[-1]]) \
                                            - sum(sample["original_logits"])
                        if min_beam_score >= self.beam_search_th:
                            break
                filtered_gold_sents, filtered_answer_starts = \
                    self.filter_edited_sents(sample, edited_gold_sents, answer_starts)
                sample["edited_gold_sents"] = filtered_gold_sents
                sample["edited_answer_starts"] = filtered_answer_starts
                if len(filtered_gold_sents) > 0:
                    sample["adversarial_mode"] = ADVERSARIAL_MODE_EDIT
                else:
                    sample["adversarial_mode"] = ADVERSARIAL_MODE_NO
                cnt.append(len(filtered_gold_sents))
                samples[i] = sample
        return samples

    def filter_edited_sents(self, sample, edited_gold_sents, answer_starts):
        new_contexts = []
        for gold_sent in edited_gold_sents:
            sample["context_sents"][sample["gold_sent_index"]] = gold_sent
            new_contexts.append(" ".join(sample["context_sents"]))
        has_answer = self.has_answer_evaluator.evaluate_whether_has_answer(new_contexts, sample["question"]).tolist()
        filtered_gold_sents = [item for (item, has) in zip(edited_gold_sents, has_answer) if has]
        filtered_answer_starts = [item for (item, has) in zip(answer_starts, has_answer) if has]
        if len(filtered_gold_sents) == 0:
            return filtered_gold_sents, filtered_answer_starts
        norm_ppls = self.ppl_evaluator.calculate_norm_ppl(sample["gold_sent"], filtered_gold_sents)
        cosine_sim = self.use_evaluator.get_semantic_sim_by_ref(sample["gold_sent"], filtered_gold_sents)
        filter_scores = cosine_sim - norm_ppls
        filtered_gold_sents = [item for (item, score) in zip(filtered_gold_sents, filter_scores) if
                               score >= self.filter_th]
        filtered_answer_starts = [item for (item, score) in zip(filtered_answer_starts, filter_scores) if
                               score >= self.filter_th]
        return filtered_gold_sents, filtered_answer_starts

    def get_answer_spacy_token_index(self, spacy_gold_sent, answer_start, answer_end):
        token_start_index, token_end_index = 0, len(spacy_gold_sent) - 1
        while token_start_index < len(spacy_gold_sent) and spacy_gold_sent[token_start_index].idx <= answer_start:
            token_start_index += 1
        while spacy_gold_sent[token_end_index].idx + len(spacy_gold_sent[token_end_index]) >= answer_end and \
                token_end_index > 0:
            token_end_index -= 1
        return token_start_index - 1, token_end_index + 1

    def _replace_texts_with_candidates(self, texts, target, candidates, answer_starts_in_text=[], answer_ends_in_text=[]):
        replaced_texts, replace_answer_starts_in_gold, replace_answer_ends_in_gold = [], [], []
        for i in range(len(texts)):
            tmp_texts = []
            try:
                matched_positions = [m.span(0) for m in re.finditer(target, texts[i])]
            except:
                print("Error when using re to find matched positions")
                try:
                    pos = texts[i].index(target)
                    matched_positions = [(pos, pos + len(target))]
                except:
                    print("Cannot using index to find matched positions")
                    continue
            if len(answer_starts_in_text) == 0:
                answer_start, answer_end = 0, 0
            else:
                answer_start, answer_end = answer_starts_in_text[i], answer_ends_in_text[i]
            for candidate in candidates:
                replaced_disractor, tmp_answer_start, tmp_answer_end, _ = \
                    self.replace_text_given_positions(texts[i], matched_positions, candidate, answer_start=answer_start,
                                                 answer_end=answer_end)
                tmp_texts.append(replaced_disractor)
                replace_answer_starts_in_gold.append(tmp_answer_start)
                replace_answer_ends_in_gold.append(tmp_answer_end)
            replaced_texts.append(tmp_texts)
        return replaced_texts, replace_answer_starts_in_gold, replace_answer_ends_in_gold

    def get_new_contexts_from_distractor(self, sample, replaced_distractors, base_context_sents):
        new_contexts, all_distractors, all_context_sents = [], [], []
        for distractor in replaced_distractors:
            for context_sents in base_context_sents:
                all_distractors.append(distractor)
                all_context_sents.append(context_sents)
                if self.distractor_insert_mode == REAR_MODE:
                    tmp_context_sents = deepcopy(context_sents)
                    tmp_context_sents[sample["gold_sent_index"] + 1] = distractor
                    new_contexts.append(" ".join(tmp_context_sents))
        return new_contexts, all_distractors, all_context_sents

    def _replace_text_and_evaluate(self, sample, distractors, target_text, replace_candidates, answer_starts, answer_ends,
                                   answer_starts_in_distractor, answer_ends_in_distractor, contexts_with_distractor,
                                   distractor_starts, distractor_ends):
        replaced_distractors, answer_starts_in_distractor, answer_ends_in_distractor = \
            self._replace_texts_with_candidates(distractors, target_text, replace_candidates, answer_starts_in_distractor,
                                                answer_ends_in_distractor)
        new_contexts, new_distractor_starts, new_distractor_ends = [], [], []
        new_answer_starts, new_answer_ends = [], []
        for j in range(len(contexts_with_distractor)):
            for replaced_distractor in replaced_distractors[j]:
                new_contexts.append(contexts_with_distractor[j][:distractor_starts[j]] + replaced_distractor +
                                    contexts_with_distractor[j][distractor_ends[j]:])
                new_distractor_starts.append(distractor_starts[j])
                new_distractor_ends.append(distractor_starts[j] + len(replaced_distractor))
            new_answer_starts.extend([answer_starts[j]] * len(replaced_distractors[j]))
            new_answer_ends.extend([answer_ends[j]] * len(replaced_distractors[j]))
        answer_start_logits, answer_end_logits = \
            self.model_evaluator.evaluate_sample_logits(new_contexts, sample["question"], new_answer_starts,
                                                        new_answer_ends)
        if answer_start_logits.size(0) < self.beam_size:
            best_indices = list(range(answer_start_logits.size(0)))
        else:
            best_indices = \
                torch.topk(answer_start_logits + answer_end_logits, k=self.beam_size, largest=False)[1].tolist()
        contexts_with_distractor = [new_contexts[index] for index in best_indices]
        answer_starts = [new_answer_starts[index] for index in best_indices]
        answer_ends = [new_answer_ends[index] for index in best_indices]
        distractor_starts = [new_distractor_starts[index] for index in best_indices]
        distractor_ends = [new_distractor_ends[index] for index in best_indices]
        answer_starts_in_distractor = [answer_starts_in_distractor[index] for index in best_indices]
        answer_ends_in_distractor = [answer_ends_in_distractor[index] for index in best_indices]
        distractors = [contexts_with_distractor[i][distractor_starts[i]: distractor_ends[i]]
                       for i in range(len(contexts_with_distractor))]
        return distractors, contexts_with_distractor, answer_starts, answer_ends, distractor_starts, distractor_ends, \
               answer_starts_in_distractor, answer_ends_in_distractor

    def _sample_token_same_level_based_on_pos(self, token, pos):
        possible_candidates = []
        synsets = wn.synsets(token)
        if len(synsets) == 0:
            return possible_candidates
        same_synsets = []
        for synset in synsets:
            if synset.lemma_names()[0] == token and synset.lexname().split(".")[0] == pos:
                same_synsets.append(synset)
        possible_candidates = []
        for synset in same_synsets:
            hypernyms = synset.hypernyms()
            for hyper in hypernyms:
                hypos = hyper.hyponyms()
                hypos = [h.lemma_names()[0].replace("_", " ") for h in hypos if h.lemma_names()[0] != token]
                possible_candidates.extend(hypos)
        if len(possible_candidates) > self.ent_noun_sample_num:
            sampled_candidates = random.sample(possible_candidates, self.ent_noun_sample_num)
        else:
            sampled_candidates = possible_candidates
        return sampled_candidates

    def edit_distractors(self, sample, ent_dict, ents=True):
        contexts_with_distractor, distractor_starts, distractor_ends = [], [], []
        if sample["adversarial_mode"] == ADVERSARIAL_MODE_EDIT:
            gold_sent_candidates = sample["edited_gold_sents"]
            answer_starts = sample["edited_answer_starts"]
        elif sample["adversarial_mode"] == ADVERSARIAL_MODE_NO:
            gold_sent_candidates = [sample["gold_sent"]]
            answer_starts = [sample["best_answer"]["answer_start"]]
        answer_ends = [s + len(sample["best_answer"]["text"]) for s in answer_starts]

        for gold_sent in gold_sent_candidates:
            tmp_context_sents = deepcopy(sample["context_sents"])
            tmp_context_sents[sample["gold_sent_index"]] = gold_sent
            if self.distractor_insert_mode == REAR_MODE:
                distractor_start = sum([len(c) + 1 for c in tmp_context_sents])
                tmp_context_sents.append(sample["gold_sent"])
                distractor_starts.append(distractor_start)
                distractor_ends.append(distractor_start + len(sample["gold_sent"]))
            if self.distractor_insert_mode == AFTER_GOLD:
                tmp_context_sents.insert(sample["gold_sent_index"] + 1, sample["gold_sent"])
                distractor_start = sum([len(c) + 1 for c in tmp_context_sents[:sample["gold_sent_index"] + 1]])
                distractor_starts.append(distractor_start)
                distractor_ends.append(distractor_start + len(sample["gold_sent"]))
            contexts_with_distractor.append(" ".join(tmp_context_sents))

        answer_starts_in_distractor, answer_ends_in_distractor = \
                [sample["best_answer"]["answer_start"] - sample["gold_sent_front_offset"]] * len(contexts_with_distractor),\
                [sample["best_answer"]["answer_end"] - sample["gold_sent_front_offset"]] * len(contexts_with_distractor)
        distractors = [sample["gold_sent"]] * len(contexts_with_distractor)
        edit_cnt = 0
        if ents:
            for i, target_ent in enumerate(sample["add_candidates"]["ents"]):
                if edit_cnt > self.max_edit_num_in_add:
                    break
                sampled_ents = random.sample(ent_dict[target_ent.label_], self.ent_noun_sample_num)
                distractors, contexts_with_distractor, answer_starts, answer_ends, distractor_starts, distractor_ends, \
                    answer_starts_in_distractor, answer_ends_in_distractor = \
                        self._replace_text_and_evaluate(sample, distractors, target_ent.text, sampled_ents, answer_starts,
                                answer_ends, answer_starts_in_distractor, answer_ends_in_distractor,
                                contexts_with_distractor, distractor_starts, distractor_ends)
                edit_cnt += 1
        else:
            for i, target_noun in enumerate(sample["add_candidates"]["nouns"]):
                if edit_cnt > self.max_edit_num_in_add:
                    break
                sampled_nouns = self._sample_token_same_level_based_on_pos(target_noun.lemma_.lower(), NOUN_POS_IN_WORDNET)
                if len(sampled_nouns) == 0:
                    continue
                inflected_sampled_nouns = self._get_inflect_aligned_tokens(sampled_nouns, target_noun)
                if len(inflected_sampled_nouns) == 0:
                    inflected_sampled_nouns = sampled_nouns
                distractors, contexts_with_distractor, answer_starts, answer_ends, distractor_starts, distractor_ends, \
                answer_starts_in_distractor, answer_ends_in_distractor = \
                    self._replace_text_and_evaluate(sample, distractors, target_noun.text, inflected_sampled_nouns,
                                                    answer_starts, answer_ends, answer_starts_in_distractor,
                                                    answer_ends_in_distractor, contexts_with_distractor,
                                                    distractor_starts, distractor_ends)
                edit_cnt += 1
        return contexts_with_distractor, answer_starts, answer_ends, distractor_starts, distractor_ends, \
               answer_starts_in_distractor, answer_ends_in_distractor, edit_cnt


    def replace_answer_with_new_ents_or_tokens(self, contexts_with_distractors, sample, answers, target_text, candidate_texts,
            answer_starts, answer_ends, distractor_starts, distractor_ends, answer_starts_in_distractor,
            answer_ends_in_distractor):
        replaced_answers, _, _ = self._replace_texts_with_candidates(answers, target_text, candidate_texts)
        new_contexts, new_distractor_starts, new_distractor_ends, new_answer_starts_in_distractor, \
                new_answer_ends_in_distractor = [], [], [], [], []
        new_answer_starts, new_answer_ends = [], []
        if len(replaced_answers) > 0:
            for i in range(len(contexts_with_distractors)):
                for replaced_answer in replaced_answers[i]:
                    cur_context_with_distractor = \
                            contexts_with_distractors[i][:distractor_starts[i] + answer_starts_in_distractor[i]] + \
                            replaced_answer + \
                            contexts_with_distractors[i][distractor_starts[i] + answer_ends_in_distractor[i]:]
                    new_contexts.append(cur_context_with_distractor)
                    offset = len(replaced_answer) - (answer_ends_in_distractor[i] - answer_starts_in_distractor[i])
                    new_distractor_starts.append(distractor_starts[i])
                    new_distractor_ends.append(distractor_ends[i] + offset)
                    new_answer_starts_in_distractor.append(answer_starts_in_distractor[i])
                    new_answer_ends_in_distractor.append(answer_ends_in_distractor[i] + offset)
                new_answer_starts.extend([answer_starts[i]] * len(replaced_answers[i]))
                new_answer_ends.extend([answer_ends[i]] * len(replaced_answers[i]))
            answer_start_logits, answer_end_logits = \
                self.model_evaluator.evaluate_sample_logits(new_contexts, sample["question"], new_answer_starts, new_answer_ends)
            if answer_start_logits.size(0) < self.beam_size:
                best_indices = list(range(answer_start_logits.size(0)))
            else:
                best_indices = \
                    torch.topk(answer_start_logits + answer_end_logits, k=self.beam_size, largest=False)[1].tolist()
            contexts_with_distractors = [new_contexts[index] for index in best_indices]
            replaced_answers = list(chain(*replaced_answers))
            answers = [replaced_answers[index] for index in best_indices]
            answer_starts = [new_answer_starts[index] for index in best_indices]
            answer_ends = [new_answer_ends[index] for index in best_indices]
            distractor_starts = [new_distractor_starts[index] for index in best_indices]
            distractor_ends = [new_distractor_ends[index] for index in best_indices]
            answer_starts_in_distractor = [new_answer_starts_in_distractor[index] for index in best_indices]
            answer_ends_in_distractor = [new_answer_ends_in_distractor[index] for index in best_indices]
        return contexts_with_distractors, answers, answer_starts, answer_ends, distractor_starts, distractor_ends, \
               answer_starts_in_distractor, answer_ends_in_distractor

    def edit_answer_in_distractors(self, spacy_answer, contexts_with_distractors, sample, ent_dict, pos_vocab_dict, answer_starts,
                                   answer_ends, answer_starts_in_distractor, answer_ends_in_distractor, distractor_starts,
                                   distractor_ends):
        answers = [spacy_answer.text] * len(contexts_with_distractors)
        if len(spacy_answer.ents) > 0:
            for target_ent in spacy_answer.ents:
                sampled_ents = random.sample(ent_dict[target_ent.label_], self.ent_noun_sample_num)
                contexts_with_distractors, answers, answer_starts, answer_ends, distractor_starts, distractor_ends, \
                    answer_starts_in_distractor, answer_ends_in_distractor = \
                    self.replace_answer_with_new_ents_or_tokens(
                        contexts_with_distractors, sample, answers, target_ent.text, sampled_ents,
                        answer_starts, answer_ends, distractor_starts, distractor_ends, answer_starts_in_distractor,
                        answer_ends_in_distractor)
        else:
            for token in spacy_answer:
                if token.pos_ in ANSWER_POS:
                    if token.pos_ == "NUM":
                        sampled_tokens = random.sample(pos_vocab_dict[token.pos_], self.ent_noun_sample_num)
                    else:
                        sampled_tokens = self._sample_token_same_level_based_on_pos(token.lemma_.lower(), token.pos_.lower())
                        if len(sampled_tokens) == 0:
                            sampled_tokens = random.sample(pos_vocab_dict[token.pos_], self.ent_noun_sample_num)
                    if token.pos_ in POS_ALIGN:
                        aligned_sampled_tokens = self._get_inflect_aligned_tokens(sampled_tokens, token)
                        if len(aligned_sampled_tokens) == 0:
                            aligned_sampled_tokens = sampled_tokens
                        else:
                            sampled_tokens = aligned_sampled_tokens
                    contexts_with_distractors, answers, answer_starts, answer_ends, distractor_starts, distractor_ends, \
                    answer_starts_in_distractor, answer_ends_in_distractor = \
                        self.replace_answer_with_new_ents_or_tokens(
                            contexts_with_distractors, sample, answers, token.text, sampled_tokens,
                            answer_starts, answer_ends, distractor_starts, distractor_ends, answer_starts_in_distractor,
                            answer_ends_in_distractor)
        new_distractors = []
        for i in range(len(contexts_with_distractors)):
            new_distractors.append(contexts_with_distractors[i][distractor_starts[i]: distractor_ends[i]])
        return contexts_with_distractors, new_distractors, answers

    def add_distractor_sentences(self, samples):
        ent_dict = self.ent_dict
        pos_vocab_dict = self.pos_vocab_dict
        for i, sample in enumerate(tqdm(samples, desc="add distractors")):
            if not sample.__contains__("gold_sent"):
                continue
            edit_cnt = 0
            # try:
            if len(sample["add_candidates"]["ents"]) > 0:
                '''First replace all possible named entities in the gold sentence as the distractor'''
                contexts_with_distractor, answer_starts, answer_ends, distractor_starts, distractor_ends, \
                answer_starts_in_distractor, answer_ends_in_distractor, edit_cnt = self.edit_distractors(
                    sample, ent_dict)
            if len(sample["add_candidates"]["nouns"]) > 0:
                '''First replace all possible nouns in the gold sentence as the distractor if there is no entity'''
                contexts_with_distractor, answer_starts, answer_ends, distractor_starts, distractor_ends, \
                answer_starts_in_distractor, answer_ends_in_distractor, edit_cnt = self.edit_distractors(
                    sample, ent_dict, ents=False)
            if edit_cnt > 0:
                '''If it is possible that a distractor can be generated, we replace the answer in the distrator'''
                answer_start, answer_end = sample["best_answer"]["answer_start"], sample["best_answer"]["answer_end"]
                answer_former_char_offset = sample["gold_sent_front_offset"]
                answer_token_start, answer_token_end = \
                    self.get_answer_spacy_token_index(sample["spacy_gold_sent"],
                                                      answer_start - answer_former_char_offset,
                                                      answer_end - answer_former_char_offset)
                spacy_answer = sample["spacy_gold_sent"][answer_token_start: answer_token_end + 1]
                contexts_with_distractor, distractors, answers = self.edit_answer_in_distractors(
                        spacy_answer, contexts_with_distractor, sample, ent_dict, pos_vocab_dict, answer_starts,
                        answer_ends, answer_starts_in_distractor, answer_ends_in_distractor, distractor_starts,
                        distractor_ends)
                sample["distractors"] = distractors
                sample["contexts_with_distractor"] = contexts_with_distractor
                sample["edited_answers"] = answers
                sample["edited_answer_starts"] = answer_starts
                if sample["adversarial_mode"] == ADVERSARIAL_MODE_EDIT:
                    sample["adversarial_mode"] = ADVERSARIAL_MODE_EDIT_AND_ADD
                else:
                    sample["adversarial_mode"] = ADVERSARIAL_MODE_ADD
                samples[i] = sample
        return samples

    def construct_new_dataset(self, samples):
        new_data = []
        sample_cnt = 0
        miss_cnt = 0
        cur_doc = {"title": samples[0]["title"], "paragraphs": []}
        for sample in samples:
            if sample["title"] != cur_doc["title"]:
                new_data.append(cur_doc)
                cur_doc = {"title": sample["title"], "paragraphs": []}
            adv_sample_index = 1
            if sample.__contains__("contexts_with_distractor") and len(sample["contexts_with_distractor"]) > 0:
                for i, context in enumerate(sample["contexts_with_distractor"][:self.remain_size_per_sample]):
                    new_answers = \
                        [{"answer_start": sample["edited_answer_starts"][i], "text": sample["best_answer"]["text"]}] * 3
                    cur_doc["paragraphs"].append({"context": context,
                                                  "qas": [{"question": sample["question"],
                                                           "answers": new_answers,
                                                           "id": sample["id"] + "-" + str(adv_sample_index)}],
                                                  "mode": sample["adversarial_mode"]})
                    adv_sample_index += 1
                    sample_cnt += 1
            elif sample.__contains__("edited_gold_sents") and len(sample["edited_gold_sents"]) > 0:
                context_sents = deepcopy(sample["context_sents"])
                for i, gold_sent in enumerate(sample["edited_gold_sents"][:self.remain_size_per_sample]):
                    new_answers = \
                        [{"answer_start": sample["edited_answer_starts"][i], "text": sample["best_answer"]["text"]}] * 3
                    context_sents[sample["gold_sent_index"]] = gold_sent
                    cur_doc["paragraphs"].append({"context": " ".join(context_sents),
                                                  "qas": [{"question": sample["question"],
                                                           "answers": new_answers,
                                                           "id": sample["id"] + "-" + str(adv_sample_index)}],
                                                  "mode": sample["adversarial_mode"]})
                    adv_sample_index += 1
                    sample_cnt += 1
            else:
                miss_cnt += 1
        new_data.append(cur_doc)
        return new_data, sample_cnt, miss_cnt

    def get_adversarial_samples(self, samples):
        '''Add gold sentence and remove its coreference relationship for each sample'''
        samples, miss_gold_sent = self.add_gold_sent(samples)
        self.logger.info("Missing gold sent number: %d", miss_gold_sent)

        '''Add overlap tokens between gold sentence and question'''
        samples, miss_edit_num, miss_add_num = self.add_overlap_token(samples)
        self.logger.info("Missing edit candidate number: %d", miss_edit_num)
        self.logger.info("Missing add candidate number: %d", miss_add_num)

        '''Add the importance of each overlapped tokens in the gold sentence'''
        samples = self.evaluate_edit_importance(samples)

        '''Add replacement synonym candidates for each possible editable token in the gold sentence'''
        samples = self.get_synonyms(samples)

        '''Edit gold sentences to get new samples'''
        samples = self.get_edited_sentences(samples)

        '''Add distractor sentences to the new samples'''
        samples = self.add_distractor_sentences(samples)

        '''Construct the adversarial dataset based on the edited context and distractors'''
        new_data, sample_cnt, miss_cnt = self.construct_new_dataset(samples)
        return samples, new_data, sample_cnt, miss_cnt


def main():
    args = get_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    logger = config_logger(os.path.join(args.output_dir, "output.log"))
    for k, v in args.__dict__.items():
        logger.info("{} : {}".format(k, v))

    '''Get the model evaluator of the target model'''
    model_evaluator = QAEvaluator(args.target_model, "metrics/squad", args.target_model_type, args.eval_batch_size)

    '''Get the evaluator to determine whether has an answer'''
    has_answer_evaluator = QAEvaluator(args.determine_model_path, "metrics/squad", "bert", args.eval_batch_size)

    '''Get the evaluator to calculate the PPL'''
    ppl_evaluator = PPLEvaluator(args.ppl_model_path, args.eval_batch_size)

    '''Get the evaluator for the USE similarity'''
    use_evaluator = USEEvaluator(args.USE_model_path)

    '''Get the attacker class'''
    qa_attacker = QAAttacker(args, model_evaluator, has_answer_evaluator, ppl_evaluator, use_evaluator, logger)

    '''Flatten the raw data and coreference relationship into each sample'''
    with open(args.target_dataset_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    data = raw_data["data"]
    with open(args.coreference_file, "r", encoding="utf-8") as f:
        coreferences = json.load(f)["corefs"]
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
                                "title": doc["title"],
                                "coreferences": coreferences[d_i][p_i]})

    if args.sample_start_index != -1 and args.sample_end_index == -1:
        samples = samples[args.sample_start_index:]
    elif args.sample_start_index == -1 and args.sample_end_index != -1:
        samples = samples[:args.sample_end_index]
    elif args.sample_start_index != -1 and args.sample_end_index != -1:
        samples = samples[args.sample_start_index:args.sample_end_index]

    '''Generate adversarial samples for the given dataset'''
    adversarial_samples, new_data, sample_cnt, miss_cnt = qa_attacker.get_adversarial_samples(samples)

    '''Save the adversarial dataset'''
    with open(os.path.join(args.output_dir, "adv_dataset.json"), "w", encoding="utf-8") as f:
        json.dump({"data": new_data, "len": sample_cnt, "version": raw_data["version"]}, f)

    logger.info("Saved adversarial samples!")
    logger.info("The number of generated adversarial samples: " + str(sample_cnt))
    logger.info("The original sample numbers that cannot generate any adversarial sample: " + str(miss_cnt))

    for i in range(len(adversarial_samples)):
        sample = adversarial_samples[i]
        if not sample.__contains__("gold_sent"):
            continue
        edit_candidates = []
        for edit_candidate in sample["edit_candidates"]:
            edit_candidates.append({"context_index": edit_candidate[0].i, "question_index": [cc.i for cc in edit_candidate[1]]})
        sample["edit_candidates"] = edit_candidates

        add_candidates = {"nouns": [], "ents": []}
        for add_candidate in sample["add_candidates"]["nouns"]:
            add_candidates["nouns"].append([add_candidate.i])
        for add_candidate in sample["add_candidates"]["ents"]:
            add_candidates["ents"].append([add_candidate.start, add_candidate.end])
        sample["add_candidates"] = add_candidates
        adversarial_samples[i] = sample
    with open(os.path.join(args.output_dir, "adv_track.pickle"), "wb") as f:
        pickle.dump(adversarial_samples, f)

if __name__ == "__main__":
    main()