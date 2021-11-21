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
import tensorflow as tf
import tensorflow_hub as hub

REAR_MODE = "rear"
candidate_pos = ["VERB", "NOUN", "ADJ", "ADV"]
ANSWER_POS = ["ADV", "ADJ", "NOUN", "VERB"]
MASK_SYMBOL = "[UNK]"

def get_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Question Answering task")
    parser.add_argument(
        "--target_dataset_file",
        type=str,
        help="The dataset file name of attack target",
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
        "--output_dataset_file",
        type=str,
        help="The path of the new json file of adversarial samples",
    )
    parser.add_argument(
        "--output_track_file",
        type=str,
        default="edit_track.pickle",
        help="The path of the pickle file that tracks the generating of adversarial samples",
    )
    parser.add_argument(
        "--ent_dict_file",
        type=str,
        default="ent_dict.json",
        help="The path for named entity dict file",
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
        default="ppdb_synonym_dict.json",
        help="The path for the processed ppdb synonym dict file",
    )
    parser.add_argument(
        "--raw_ppdb_synonym_dict_file",
        type=str,
        default="ppdb_synonyms.txt",
        help="The path for the raw ppdb synonym text file",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=3,
        help="The beam search size during generating adversarial samples",
    )
    parser.add_argument(
        "--ent_sample_num",
        type=int,
        default=20,
        help="The number for sampling named entity when generating distractor",
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
    return parser.parse_args()

class USE(object):
    def __init__(self, cache_path):
        super(USE, self).__init__()
        self.embed = hub.load(cache_path)

    def semantic_sim(self, sents1, sents2):
        with tf.device("/gpu:0"):
            embed1 = self.embed(sents1)["outputs"]
            embed2 = self.embed(sents2)["outputs"]
            embed1 = tf.stop_gradient(tf.nn.l2_normalize(embed1, axis=1))
            embed2 = tf.stop_gradient(tf.nn.l2_normalize(embed2, axis=1))
            cosine_similarity = tf.reduce_sum(tf.multiply(embed1, embed2), axis=1)
            cosine_similarity = tf.clip_by_value(cosine_similarity, -1.0, 1.0)
            return (1.0 - tf.acos(cosine_similarity)).cpu().numpy()

class QAAttacker:
    def __init__(self, args, model_evaluator):
        self.beam_size = args.beam_size
        self.ent_sample_num = args.ent_sample_num
        self.distractor_insert_mode = args.distractor_insert_mode
        self.spacy_nlp = spacy.load("en_core_web_md")

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
        print("1")

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
        new_samples = []
        for i, sample in enumerate(tqdm(samples, desc="get gold sent")):
            char_pos = 0
            answer_starts = [a["answer_start"] for a in sample["answers"]]
            answer_ends = [a["answer_start"] + len(a["text"]) for a in sample["answers"]]
            sent_hits = [0] * len(sample["context_sents"])
            sent_correspond_answer_index = [[]] * len(sample["context_sents"])
            for j, sent in enumerate(sample["context_sents"]):
                answer_hits = 0
                for a_i in range(len(answer_starts)):
                    if answer_starts[a_i] >= char_pos and answer_ends[a_i] <= char_pos + len(sent):
                        answer_hits += 1
                        sent_correspond_answer_index[j].append(a_i)
                sent_hits[j] = answer_hits
                char_pos += (len(sent) + 1)
            sorted_sent_index = sorted(range(len(sent_hits)), key=lambda k: sent_hits[k], reverse=True)
            if sent_hits[sorted_sent_index[0]] > 0:
                sample["gold_sent"] = sample["context_sents"][sorted_sent_index[0]]
                sample["gold_sent_index"] = sorted_sent_index[0]
                selected_answers = [sample["answers"][a_i] for a_i in sent_correspond_answer_index[sorted_sent_index[0]]]
                best_answer = max(selected_answers, key=selected_answers.count)
                best_answer["answer_end"] = best_answer["answer_start"] + len(best_answer["text"])
                sample["best_answer"] = best_answer
                sample["gold_sent_front_offset"] = sum([len(c) + 1 for c in sample["context_sents"][:sorted_sent_index[0]]])
                new_samples.append(sample)
            else:
                print("Missing gold sentence")
        return new_samples

    def add_overlap_token(self, samples):
        spacy_nlp = self.spacy_nlp
        for i, sample in enumerate(tqdm(samples, desc="add overlap token")):
            if sample.__contains__("gold_sent"):
                spacy_question, spacy_gold_sent = spacy_nlp.pipe([sample["question"], sample["gold_sent"]])
                question_ents_text = [ent.text for ent in spacy_question.ents]
                question_token_candidates_lemma_dict = {}
                for token in spacy_question:
                    if token.ent_type == 0 and (not token.is_punct) and (token.pos_ in candidate_pos):
                        if not question_token_candidates_lemma_dict.__contains__(token.lemma_):
                            question_token_candidates_lemma_dict[token.lemma_] = []
                        question_token_candidates_lemma_dict[token.lemma_].append(token)
                edit_candidates = []
                edit_candidates_set = set()
                for token in spacy_gold_sent:
                    if question_token_candidates_lemma_dict.__contains__(token.lemma_) and not edit_candidates_set.__contains__(token.lemma_):
                        edit_candidates.append([token, question_token_candidates_lemma_dict[token.lemma_]])
                        edit_candidates_set.add(token.lemma_)
                add_edit_candidates = []
                answer_start_in_gold, answer_end_in_gold = \
                    sample["best_answer"]["answer_start"] - sample["gold_sent_front_offset"], \
                    sample["best_answer"]["answer_end"] - sample["gold_sent_front_offset"]
                for ent in spacy_gold_sent.ents:
                    if ent.text in question_ents_text and \
                            (ent.end_char < answer_start_in_gold or ent.start_char > answer_end_in_gold):
                        add_edit_candidates.append(ent)
                samples[i]["edit_candidates"] = edit_candidates
                samples[i]["add_edit_candidates"] = add_edit_candidates
                samples[i]["spacy_gold_sent"] = spacy_gold_sent
                samples[i]["spacy_question"] = spacy_question
        return samples

    def evaluate_importance(self, samples):
        model_evaluator = self.model_evaluator
        masked_contexts, indices, original_contexts = [], [], []
        masked_questions, original_questions = [], []
        masked_answer_starts, masked_answer_ends, answer_starts, answer_ends = [], [], [], []
        for i, sample in enumerate(samples):
            answer_start_char_in_gold = sample["best_answer"]["answer_start"] - sample["gold_sent_front_offset"]
            context_sents = sample["context_sents"]
            original_contexts.append(" ".join(context_sents))
            original_questions.append(sample["question"])
            answer_starts.append(sample["best_answer"]["answer_start"])
            answer_ends.append(sample["best_answer"]["answer_end"])
            for j, edit_candidate in enumerate(sample["edit_candidates"]):
                new_answer_start, new_answer_end = sample["best_answer"]["answer_start"], sample["best_answer"]["answer_end"]
                if edit_candidate[0].idx < answer_start_char_in_gold:
                    new_answer_start += (len(MASK_SYMBOL) - len(edit_candidate[0]))
                    new_answer_end += (len(MASK_SYMBOL) - len(edit_candidate[0]))
                masked_answer_starts.append(new_answer_start)
                masked_answer_ends.append(new_answer_end)
                masked_gold_sent = sample["gold_sent"][:edit_candidate[0].idx] + MASK_SYMBOL + \
                                   sample["gold_sent"][edit_candidate[0].idx + len(edit_candidate[0]):]
                masked_contexts.append(" ".join(context_sents[:sample["gold_sent_index"]] + [masked_gold_sent] +
                                                context_sents[sample["gold_sent_index"] + 1:]))
                masked_questions.append(sample["question"])
                indices.append([i, j])
        original_start_logits, original_end_logits = model_evaluator.evaluate_sample_logits(original_contexts,
                original_questions, answer_starts, answer_ends, single_question=False)
        masked_start_logits, masked_end_logits = model_evaluator.evaluate_sample_logits(masked_contexts, masked_questions,
                masked_answer_starts, masked_answer_ends, single_question=False)
        for i in range(len(indices)):
            original_index = indices[i][0]
            if not samples[original_index].__contains__("original_logits"):
                samples[original_index]["original_logits"] = [original_start_logits[original_index].item(),
                                                              original_end_logits[original_index].item()]
            if not samples[original_index].__contains__("masked_logits"):
                samples[original_index]["masked_logits"] = \
                    [[0 for _ in range(len(samples[original_index]["edit_candidates"]))],
                     [0 for _ in range(len(samples[original_index]["edit_candidates"]))]]
            samples[original_index]["masked_logits"][0][indices[i][1]] = masked_start_logits[indices[i][1]].item()
            samples[original_index]["masked_logits"][1][indices[i][1]] = masked_end_logits[indices[i][1]].item()
        return samples

    def get_synonyms(self, samples):
        ppdb_synonym_dict = self.ppdb_synonym_dict
        for i, sample in enumerate(tqdm(samples, desc="get edit synonyms")):
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

    def get_edited_sentences(self, samples):
        model_evaluator = self.model_evaluator
        spacy_nlp = self.spacy_nlp
        new_samples = []
        for i, sample in enumerate(tqdm(samples, desc="get edited samples")):
            if len(sample["edit_synonyms"]) > 0:
                gold_sents = [sample["gold_sent"]]
                answer_start, answer_end = sample["best_answer"]["answer_start"], sample["best_answer"]["answer_end"]
                answer_starts = [answer_start]
                answer_ends = [answer_end]
                all_answer_starts = [[a["answer_start"] for a in sample["answers"]]]
                sorted_indices = np.argsort([
                    sample["masked_logits"][0][m] + sample["masked_logits"][1][m] -
                    sample["original_logits"][0] - sample["original_logits"][1]
                    for m in range(len(sample["edit_candidates"]))]).tolist()
                for edit_index in sorted_indices:
                    edit_target, synonyms = sample["edit_candidates"][edit_index], sample["edit_synonyms"][edit_index]
                    synonyms = [s.replace("_", " ") for s in synonyms]
                    aligned_synonyms = []
                    for synonym in spacy_nlp.pipe(synonyms):
                        success_cnt = 0
                        aligned_result = []
                        for token in synonym:
                            aligned_token = token._.inflect(edit_target[0].tag_)
                            if aligned_token is not None:
                                success_cnt += 1
                                aligned_result.append(aligned_token)
                            else:
                                aligned_result.append(token.text)
                        if success_cnt > 0:
                            aligned_synonyms.append(" ".join(aligned_result))
                    # spacy_synonyms = [s[0]._.inflect(edit_target[0].tag_) for s in spacy_nlp.pipe(synonyms)]
                    if len(aligned_synonyms) > 0:
                        answer_offsets = len(" ".join(sample["context_sents"][:sample["gold_sent_index"]])) + 1
                        replaced_gold_sents, replaced_answer_starts, replaced_answer_ends, replaced_all_answer_starts = \
                            [], [], [], []
                        for s_i, gold_sent in enumerate(gold_sents):
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
                        answer_start_logits, answer_end_logits = model_evaluator.evaluate_sample_logits(new_contexts,
                                    sample["question"], replaced_answer_starts, replaced_answer_ends)
                        if answer_start_logits.size(0) < self.beam_size:
                            best_indices = list(range(answer_start_logits.size(0)))
                        else:
                            best_indices = torch.topk(answer_start_logits + answer_end_logits, k=self.beam_size,
                                                      largest=False)[1].squeeze()
                        gold_sents = [replaced_gold_sents[index] for index in best_indices]
                        answer_starts = [replaced_answer_starts[index] for index in best_indices]
                        answer_ends = [replaced_answer_ends[index] for index in best_indices]
                        all_answer_starts = [replaced_all_answer_starts[index] for index in best_indices]
                sample["edited_gold_sents"] = gold_sents
                sample["edited_answer_starts"] = answer_starts
                new_samples.append(sample)
                # if len(gold_sents) > 1:
                #     context_list = copy.deepcopy(sample["context_sents"])
                #     for i, gold_sent in enumerate(gold_sents):
                #         context_list[sample["gold_sent_index"]] = gold_sent
                #         new_context = " ".join(context_list)
                #         new_answers = []
                #         for j, answer in enumerate(sample["answers"]):
                #             new_answers.append({"answer_start": all_answer_starts[i][j], "text": answer["text"]})
                #         new_samples.append({"context": new_context,
                #                             "qas": [
                #                                 {"answers": new_answers,
                #                                  "question": sample["question"],
                #                                 "id": "-".join([sample["id"], str(i)])}
                #                             ]})
        return new_samples

    def get_answer_spacy_token_index(self, spacy_gold_sent, answer_start, answer_end):
        token_start_index, token_end_index = 0, len(spacy_gold_sent) - 1
        while token_start_index < len(spacy_gold_sent) and spacy_gold_sent[token_start_index].idx <= answer_start:
            token_start_index += 1
        while spacy_gold_sent[token_end_index].idx + len(spacy_gold_sent[token_end_index]) >= answer_end and \
                token_end_index > 0:
            token_end_index -= 1
        return token_start_index - 1, token_end_index + 1

    def replace_texts_with_candidates(self, texts, target, candidates, answer_starts_in_text=[], answer_ends_in_text=[]):
        replaced_texts, replace_answer_starts_in_gold, replace_answer_ends_in_gold = [], [], []
        for i in range(len(texts)):
            tmp_texts = []
            try:
                matched_positions = [m.span(0) for m in re.finditer(target, texts[i])]
            except:
                print("Error when using re to find matched positions")
                pos = texts[i].index(target)
                matched_positions = [(pos, pos + len(target))]
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

    def edit_ents_in_distractors(self, sample, ent_dict, model_evaluator):
        contexts_with_distractor, distractor_starts, distractor_ends = [], [], []
        for edited_gold_sent in sample["edited_gold_sents"]:
            tmp_context_sents = deepcopy(sample["context_sents"])
            tmp_context_sents[sample["gold_sent_index"]] = edited_gold_sent
            if self.distractor_insert_mode == REAR_MODE:
                tmp_context_sents.insert(sample["gold_sent_index"] + 1, sample["gold_sent"])
                contexts_with_distractor.append(" ".join(tmp_context_sents))
                distractor_start = sum([len(c) + 1 for c in tmp_context_sents[:sample["gold_sent_index"] + 1]])
                distractor_starts.append(distractor_start)
                distractor_ends.append(distractor_start + len(sample["gold_sent"]))
        answer_starts = sample["edited_answer_starts"]
        answer_ends = [s + len(sample["best_answer"]["text"]) for s in answer_starts]
        answer_starts_in_distractor, answer_ends_in_distractor = \
                [sample["best_answer"]["answer_start"] - sample["gold_sent_front_offset"]] * len(contexts_with_distractor),\
                [sample["best_answer"]["answer_end"] - sample["gold_sent_front_offset"]] * len(contexts_with_distractor)
        distractors = [sample["gold_sent"]] * len(contexts_with_distractor)
        for i, target_ent in enumerate(sample["add_edit_candidates"]):
            sampled_ents = random.sample(ent_dict[target_ent.label_], self.ent_sample_num)
            replaced_distractors, answer_starts_in_distractor, answer_ends_in_distractor = self.replace_texts_with_candidates(
                distractors, target_ent.text, sampled_ents, answer_starts_in_distractor, answer_ends_in_distractor)
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
                model_evaluator.evaluate_sample_logits(new_contexts, sample["question"], new_answer_starts,
                                                      new_answer_ends)
            if answer_start_logits.size(0) < self.beam_size:
                best_indices = list(range(answer_start_logits.size(0)))
            else:
                best_indices = \
                    torch.topk(answer_start_logits + answer_end_logits, k=self.beam_size, largest=False)[1].squeeze()
            contexts_with_distractor = [new_contexts[index] for index in best_indices]
            answer_starts = [new_answer_starts[index] for index in best_indices]
            answer_ends = [new_answer_ends[index] for index in best_indices]
            distractor_starts = [new_distractor_starts[index] for index in best_indices]
            distractor_ends = [new_distractor_ends[index] for index in best_indices]
            answer_starts_in_distractor = [answer_starts_in_distractor[index] for index in best_indices]
            answer_ends_in_distractor = [answer_ends_in_distractor[index] for index in best_indices]
            distractors = [contexts_with_distractor[i][distractor_starts[i]: distractor_ends[i]]
                           for i in range(len(contexts_with_distractor))]
        return contexts_with_distractor, answer_starts, answer_ends, distractor_starts, distractor_ends, \
               answer_starts_in_distractor, answer_ends_in_distractor


    def replace_answer_with_new_ents_or_tokens(self, contexts_with_distractors, sample, answers, target_text, candidate_texts,
            answer_starts, answer_ends, distractor_starts, distractor_ends, answer_starts_in_distractor,
            answer_ends_in_distractor, model_evaluator):
        replaced_answers, _, _ = self.replace_texts_with_candidates(answers, target_text, candidate_texts)
        new_contexts, new_distractor_starts, new_distractor_ends, new_answer_starts_in_distractor, \
                new_answer_ends_in_distractor = [], [], [], [], []
        new_answer_starts, new_answer_ends = [], []
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
            model_evaluator.evaluate_sample_logits(new_contexts, sample["question"], new_answer_starts, new_answer_ends)
        if answer_start_logits.size(0) < self.beam_size:
            best_indices = list(range(answer_start_logits.size(0)))
        else:
            best_indices = \
                torch.topk(answer_start_logits + answer_end_logits, k=self.beam_size, largest=False)[1].squeeze()
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
                                   distractor_ends, model_evaluator):
        answers = [spacy_answer.text] * len(contexts_with_distractors)
        if len(spacy_answer.ents) > 0:
            for target_ent in spacy_answer.ents:
                sampled_ents = random.sample(ent_dict[target_ent.label_], self.ent_sample_num)
                contexts_with_distractors, answers, answer_starts, answer_ends, distractor_starts, distractor_ends, \
                    answer_starts_in_distractor, answer_ends_in_distractor = \
                    self.replace_answer_with_new_ents_or_tokens(
                        contexts_with_distractors, sample, answers, target_ent.text, sampled_ents,
                        answer_starts, answer_ends, distractor_starts, distractor_ends, answer_starts_in_distractor,
                        answer_ends_in_distractor, model_evaluator)
        else:
            for token in spacy_answer:
                if token.pos_ in ANSWER_POS:
                    sampled_tokens = random.sample(pos_vocab_dict[token.pos_], self.ent_sample_num)
                    contexts_with_distractors, answers, answer_starts, answer_ends, distractor_starts, distractor_ends, \
                    answer_starts_in_distractor, answer_ends_in_distractor = \
                        self.replace_answer_with_new_ents_or_tokens(
                            contexts_with_distractors, sample, answers, token.text, sampled_tokens,
                            answer_starts, answer_ends, distractor_starts, distractor_ends, answer_starts_in_distractor,
                            answer_ends_in_distractor, model_evaluator)
        new_distractors = []
        for i in range(len(contexts_with_distractors)):
            new_distractors.append(contexts_with_distractors[i][distractor_starts[i]: distractor_ends[i]])
        return contexts_with_distractors, new_distractors, answers

    def add_distractor_sentences(self, samples):
        model_evaluator = self.model_evaluator
        ent_dict = self.ent_dict
        pos_vocab_dict = self.pos_vocab_dict
        new_samples = []
        for i, sample in enumerate(tqdm(samples, desc="add distractors")):
            if len(sample["add_edit_candidates"]) > 0:
                answer_start, answer_end = sample["best_answer"]["answer_start"], sample["best_answer"]["answer_end"]
                '''First replace all possible named entities in the gold sentence as the distractor'''
                contexts_with_distractor, answer_starts, answer_ends, distractor_starts, distractor_ends, \
                answer_starts_in_distractor, answer_ends_in_distractor = self.edit_ents_in_distractors(
                    sample, ent_dict, model_evaluator)
                '''Then replace the answer with a fake one as the final distractor'''
                answer_former_char_offset = sample["gold_sent_front_offset"]
                answer_token_start, answer_token_end = \
                    self.get_answer_spacy_token_index(sample["spacy_gold_sent"], answer_start - answer_former_char_offset,
                                                 answer_end - answer_former_char_offset)
                spacy_answer = sample["spacy_gold_sent"][answer_token_start: answer_token_end + 1]
                contexts_with_distractor, distractors, answers = self.edit_answer_in_distractors(
                    spacy_answer, contexts_with_distractor, sample, ent_dict, pos_vocab_dict, answer_starts, answer_ends,
                    answer_starts_in_distractor, answer_ends_in_distractor, distractor_starts, distractor_ends,
                    model_evaluator)
                sample["distractors"] = distractors
                sample["contexts_with_distractor"] = contexts_with_distractor
                sample["edited_answers"] = answers
                sample["edited_answer_starts"] = answer_starts
                new_samples.append(sample)
        return new_samples

    def get_adversarial_samples(self, samples):
        new_samples = []
        '''Add gold sentence for each sample'''
        samples = self.add_gold_sent(samples)

        '''Add overlap tokens between gold sentence and question'''
        samples = self.add_overlap_token(samples)

        '''Add the importance of each overlapped tokens in the gold sentence'''
        samples = self.evaluate_importance(samples)

        '''Add replacement synonym candidates for each possible editable token in the gold sentence'''
        samples = self.get_synonyms(samples)

        '''Edit gold sentences to get new samples'''
        new_samples = self.get_edited_sentences(samples)

        '''Add distractor sentences to the new samples'''
        new_samples = self.add_distractor_sentences(new_samples)
        return new_samples

def main():
    args = get_args()

    '''Get the model evaluator of the target model'''
    model_evaluator = QAEvaluator(args.target_model, "../metrics/squad", args.target_model_type, args.eval_batch_size)
    model_evaluator.evaluate_whether_has_answer(["The Broncos took an early lead in Super Bowl 50 and never trailed. Newton was limited by Denver's defense, which sacked him seven times and forced him into three turnovers, including a fumble which they recovered for a touchdown. Denver linebacker Von Miller was named Super Bowl MVP, recording five solo tackles, 2\u00bd sacks, and two forced fumbles.", "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24\u201310 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50."], "Which NFL team represented the AFC at Super Bowl 50?")

    '''Get the PPL evaluator'''
    qa_attacker = QAAttacker(args, model_evaluator)
    '''Flatten the raw data into each sample'''
    with open(args.target_dataset_file, "r", encoding="utf-8") as f:
        data = json.load(f)["data"]
    samples = []
    for doc in data[5:6]:
        for p_i, paragraph in enumerate(doc["paragraphs"][:10]):
            for qas in paragraph["qas"]:
                samples.append({"context": paragraph["context"],
                                "context_sents": nltk.sent_tokenize(paragraph["context"]),
                                "question": qas["question"],
                                "answers": qas["answers"],
                                "id": qas["id"],
                                "title": doc["title"]})

    '''Generate adversarial samples for the given dataset'''
    adversarial_samples = qa_attacker.get_adversarial_samples(samples)

    '''Construct the adversarial dataset based on the edited context and distractors'''
    new_data = []
    cnt = 0
    cur_doc = {"title": adversarial_samples[0]["title"], "paragraphs": []}
    for sample in adversarial_samples:
        if sample["title"] != cur_doc["title"]:
            new_data.append(cur_doc)
            cur_doc = {"title": sample["title"], "paragraphs": []}
        adv_sample_index = 1
        for i, context in enumerate(sample["contexts_with_distractor"]):
            new_answers = \
                [{"answer_start": sample["edited_answer_starts"][i], "text": sample["best_answer"]["text"]}] * 3
            cur_doc["paragraphs"].append({"context": context,
                                          "qas": [{"question": sample["question"],
                                                   "answers": new_answers,
                                                   "id": sample["id"] + "-" + str(adv_sample_index)}]})
            adv_sample_index += 1
            cnt += 1
    new_data.append(cur_doc)
        # for i, edited_gold_sent in enumerate(sample["edited_gold_sents"]):
        #     for distractor in sample["distractors"]:
        #         if DISTRACTOR_INSERT_MODE  == "after":
        #             context_sents = sample["context_sents"]
        #             context_sents[sample["gold_sent_index"]] = edited_gold_sent
        #             context_sents.insert(sample["gold_sent_index"] + 1, distractor)
        #             new_context = " ". join(context_sents)
        #         new_answers = \
        #             [{"answer_start": sample["edited_answer_starts"][i], "text": sample["best_answer"]["text"]}] * 3
        #         cur_doc["paragraphs"].append({"context": new_context,
        #                                       "qas": [{"question": sample["question"],
        #                                                "answers": new_answers,
        #                                                "id": sample["id"] + "-" + adv_sample_index}]})
        #         adv_sample_index += 1
        #         cnt += 1

    with open(args.output_dataset_file, "w", encoding="utf-8") as f:
        json.dump({"data": new_data, "len": cnt, "version": "1.1"}, f)

    for i in range(len(adversarial_samples)):
        sample = adversarial_samples[i]
        ec = []
        for c in sample["edit_candidates"]:
            ec.append({"context_index": c[0].i, "question_index": [cc.i for cc in c[1]]})
        sample["edit_candidates"] = ec
        adversarial_samples[i] = sample
        aec = []
        for c in sample["add_edit_candidates"]:
            aec.append([c.start, c.end])
        sample["add_edit_candidates"] = aec
    with open(args.output_track_file, "wb") as f:
        pickle.dump(adversarial_samples, f)

if __name__ == "__main__":
    main()