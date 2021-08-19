# -*- coding: utf-8 -*-
import argparse
import json
import pickle
import random
import traceback
from copy import deepcopy
from functools import partial
from itertools import chain

import spacy
import spacy_alignments
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.corpus import wordnet
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from PWWS.utils_PWWS import ReplaceCandidate, ReplaceOp

SUPPORT_POS_TAG = [
    'CC',  # coordinating conjunction, like "and but neither versus whether yet so"
    'JJ',  # Adjective, like "second ill-mannered"
    'JJR',  # Adjective, comparative, like "colder"
    'JJS',  # Adjective, superlative, like "cheapest"
    'NN',  # Noun, singular or mass
    'NNS',  # Noun, plural
    'NNP',  # Proper noun, singular
    'NNPS',  # Proper noun, plural
    'RB',  # Adverb
    'RBR',  # Adverb, comparative, like "lower heavier"
    'RBS',  # Adverb, superlative, like "best biggest"
    'VB',  # Verb, base form
    'VBD',  # Verb, past tense
    'VBG',  # Verb, gerund or present participle
    'VBN',  # Verb, past participle
    'VBP',  # Verb, non-3rd person singular present
    'VBZ',  # Verb, 3rd person singular present
]


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Question Answering task")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="The path for the trained model to be loaded",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=384,
    )
    parser.add_argument(
        "--doc_stride",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--replace_mode",
        type=str,
        default="context",
        help="The mode to replace words in input, both: replace on both context and question, context: only context, "
             "question: only question"
    )
    parser.add_argument(
        "--max_candidate_num",
        type=int,
        default=5,
        help="The maximum number for candidates in synonym or named entity"
    )
    parser.add_argument(
        "--use_NE",
        type=int,
        default=1,
        help="Whether use named entity replacement, 1: use, 0:do not use"
    )
    parser.add_argument(
        "--output_file_name",
        type=str,
        default="PWWS_raw.json",
        help="The name for the output adversarial sample file"
    )
    parser.add_argument(
        "--determine_method",
        type=str,
        default="raw",
        help="The method to determine whether an attack is successful, raw: whether the argmax start and end indices is "
             "different, text: whether the predict text is the same as the provided texts, "
             "nbest: consider the nbest results for start and end indices which is the same as inference"
    )
    parser.add_argument(
        "--forbidden_span",
        type=int,
        default=5,
        help="The span width that tokens within this span around the ground truth answer will not be replaced"
    )
    args = parser.parse_args()
    return args


class SaliencyModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, model_inputs, answer_starts, answer_ends):
        outputs = self.model(**model_inputs)
        start_logits, end_logits = F.softmax(outputs.start_logits.detach(), dim=-1), F.softmax(
            outputs.end_logits.detach(), dim=-1)
        start_prob = torch.mean(torch.gather(start_logits, 1, answer_starts), dim=-1)
        end_prob = torch.mean(torch.gather(end_logits, 1, answer_ends), dim=-1)
        return start_prob, end_prob


class DetermineModelWrapper(nn.Module):
    def __init__(self, model, tokenizer, determine_method="raw"):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.determine_method = determine_method
        self.nbest = 20
        self.max_answer_length = 30

    def forward(self, model_inputs, context_length, answer_starts, answer_ends, answer_text):
        for k, v in model_inputs.items():
            if len(v.size()) == 1:
                model_inputs[k] = v.unsqueeze(0)
        outputs = self.model(**model_inputs)
        for k, v in model_inputs.items():
            if len(v.size()) == 2:
                model_inputs[k] = v.squeeze()
        # context_length = torch.sum(model_inputs['token_type_ids'] == 0)
        start_logits, end_logits = outputs.start_logits.detach().squeeze(), outputs.end_logits.detach().squeeze()
        start_indices = [x + 1 for x in torch.argsort(start_logits[1: context_length - 1], descending=True).tolist()]
        end_indices = [x + 1 for x in torch.argsort(end_logits[1: context_length - 1], descending=True).tolist()]
        pred_answer_start = start_indices[0]
        pred_answer_end = end_indices[0]
        pred_answer_text = self.tokenizer.decode(model_inputs['input_ids'][pred_answer_start: pred_answer_end + 1])
        if self.determine_method == "raw":
            if pred_answer_end in answer_ends or pred_answer_start in answer_starts:
                return False
        elif self.determine_method == "text":
            for text in answer_text:
                if pred_answer_text in text.lower() or text.lower() in pred_answer_text:
                    return False
        elif self.determine_method == "nbest":
            start_indices = start_indices[: self.nbest]
            end_indices = end_indices[: self.nbest]
            indices_list = []
            for start_index in start_indices:
                for end_index in end_indices:
                    if end_index >= start_index and (end_index - start_index + 1) <= self.max_answer_length:
                        indices_list.append([start_index, end_index, start_logits[start_index] + end_logits[end_index]])
            best_res = sorted(indices_list, key=lambda x: x[2], reverse=True)[0]
            pred_answer_text = self.tokenizer.decode(model_inputs['input_ids'][best_res[0]: best_res[1] + 1])
            for text in answer_text:
                if pred_answer_text in text.lower() or text.lower() in pred_answer_text:
                    return None, None
            print(pred_answer_text)
            print(answer_text)
        return pred_answer_text, (start_logits.cpu(), end_logits.cpu())


class ModelFooler():
    def __init__(self, args):
        self.sample_list = self.load_data(args.dataset_path)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
        self.model = AutoModelForQuestionAnswering.from_pretrained(args.model_path)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.data_loader = DataLoader(self.sample_list, batch_size=args.batch_size, collate_fn=self.collate_func)
        self.saliency_model = SaliencyModelWrapper(self.model)
        self.saliency_model.to(self.device)
        self.saliency_model.eval()
        self.determine_model = DetermineModelWrapper(self.model, self.tokenizer, args.determine_method)
        self.determine_model.to(self.device)
        self.determine_model.eval()
        self.spacy_nlp = spacy.load("en_core_web_md")
        self.replace_mode = args.replace_mode
        self.use_NE = args.use_NE
        self.batch_size = args.batch_size
        self.max_candidate_num = args.max_candidate_num
        self.output_file_name = args.output_file_name
        self.forbidden_span = args.forbidden_span
        with open("ent_dict.json", "r", encoding="utf-8") as f:
            self.ent_dict = json.load(f)

    def collate_func(self, data):
        sample_batch = {"context": [], "question": [], "answer_start": [], "answer_text": [], "id": []}
        for d in data:
            sample_batch["context"].append(d["context"])
            sample_batch["question"].append(d["question"])
            sample_batch["answer_start"].append([x["answer_start"] for x in d["answers"]])
            sample_batch["answer_text"].append([x["text"] for x in d["answers"]])
            sample_batch["id"].append(d["id"])
        tokenized_samples = self.tokenizer(
            sample_batch["question"], sample_batch["context"], max_length=args.max_length,
            stride=args.doc_stride, padding=True, truncation=True, return_overflowing_tokens=True,
            return_tensors="pt")
        input_lengths, context_lengths, question_lengths = self.get_lengths(tokenized_samples)
        answer_starts, answer_ends = self.get_start_end_token_pos(sample_batch,
                                                                  [x.offsets for x in tokenized_samples.encodings],
                                                                  question_lengths)
        return {"sample_batch": sample_batch, "tokenized_samples": tokenized_samples,
                "lengths": [input_lengths, context_lengths, question_lengths],
                "answer_starts": answer_starts, "answer_ends": answer_ends}

    def replace_collate_func(self, data):
        batch = {
            "input_ids": self.pad_list_tensors([d["input_ids"] for d in data], self.tokenizer.pad_token_id),
            "attention_mask": self.pad_list_tensors([d["attention_mask"] for d in data], self.tokenizer.pad_token_id),
            "answer_starts": torch.stack([d["answer_starts"] for d in data]),
            "answer_ends": torch.stack([d["answer_ends"] for d in data]),
            "sample_ids": torch.LongTensor([d["sample_id"] for d in data])
        }
        if data[0].__contains__("token_type_ids"):
            batch["token_type_ids"] = self.pad_list_tensors([d["token_type_ids"] for d in data],
                                                            self.tokenizer.pad_token_id)
        return batch

    def pad_list_tensors(self, tensor_list, val):
        max_len = max([x.size(0) for x in tensor_list])
        data = [F.pad(x, pad=(0, max_len - x.size(0)), mode='constant', value=val) for x in tensor_list]
        return torch.stack(data)

    def load_data(self, data_path):
        with open(data_path, 'r', encoding="utf-8") as f:
            data = json.load(f)
        sample_list = []
        for doc in data["data"]:
            for paragraph in doc["paragraphs"]:
                for qa in paragraph["qas"]:
                    sample_list.append({"context": paragraph["context"], "question": qa["question"],
                                        "answers": qa["answers"], "id": qa["id"]})
        # sample_list = sample_list[2656:]
        return sample_list

    def get_lengths(self, tokenized_samples):
        input_lengths = torch.sum(tokenized_samples.data["attention_mask"], dim=1)
        ## question_lengths contains the initial [CLS] token but do not contain the final [SEP] token
        ## context_lengths contain do not contain the [SEP] tokens
        context_lengths, question_lengths = [], []
        for i in range(len(tokenized_samples.encodings)):
            end_index = input_lengths[i].item() - 2
            while tokenized_samples.encodings[i].type_ids[end_index] == 1 and end_index > 0:
                end_index -= 1
            question_lengths.append(end_index)
        question_lengths = torch.tensor(question_lengths)
        context_lengths = input_lengths - 1 - question_lengths
        return input_lengths, context_lengths, question_lengths

    def get_start_end_token_pos(self, sample_batch, offsets, question_lengths):
        def fix_answer_length(answer):
            if len(answer) > 3:
                answer = answer[:3]
            elif len(answer) < 3:
                answer = answer + [answer[0]] * (3 - len(answer))
            return answer

        answer_starts, answer_ends = [], []
        for i in range(len(sample_batch["answer_start"])):
            cur_answer_start_char = sample_batch["answer_start"][i]
            cur_answer_end_char = [x + y for x, y in
                                   zip(cur_answer_start_char, [len(x) for x in sample_batch["answer_text"][i]])]
            cur_offsets = offsets[i]
            cur_answer_start_token, cur_answer_end_token = [], []
            for j in range(len(cur_answer_start_char)):
                start_char, end_char = cur_answer_start_char[j], cur_answer_end_char[j]
                token_start_index, token_end_index = question_lengths[i].item(), len(cur_offsets) - 2
                while token_start_index < len(cur_offsets) and cur_offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                cur_answer_start_token.append(token_start_index - 1)
                while cur_offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                cur_answer_end_token.append(token_end_index + 1)
            cur_answer_start_token = fix_answer_length(cur_answer_start_token)
            cur_answer_end_token = fix_answer_length(cur_answer_end_token)
            answer_starts.append(cur_answer_start_token)
            answer_ends.append(cur_answer_end_token)
        return torch.tensor(answer_starts), torch.tensor(answer_ends)

    def evaluate_token_saliency(self, model_input, answer_starts, answer_ends):
        for k, v in model_input.items():
            model_input[k] = v.to(self.device)
        answer_starts, answer_ends = answer_starts.to(self.device), answer_ends.to(self.device)
        # start_index = 1 if self.replace_mode != "question" else torch.min(context_lengths).item() + 1
        # end_index = model_input["input_ids"].size()[1] - 1 if self.replace_mode != "context" else torch.max(context_lengths).item()
        start_index, end_index = 1, model_input["input_ids"].size()[1] - 1
        with torch.no_grad():
            original_start_prob, original_end_prob = self.saliency_model(model_input, answer_starts, answer_ends)
            token_saliency = None

            for i in range(start_index, end_index):
                new_model_input = deepcopy(model_input)
                new_model_input["input_ids"][:, i] = self.tokenizer.unk_token_id
                new_model_input["attention_mask"][:, i] = 0
                new_start_prob, new_end_prob = self.saliency_model(new_model_input, answer_starts, answer_ends)
                saliency = (original_start_prob - new_start_prob) + (original_end_prob - new_end_prob)
                if token_saliency is None:
                    token_saliency = saliency.cpu().unsqueeze(1)
                else:
                    token_saliency = torch.cat((token_saliency, saliency.cpu().unsqueeze(1)), dim=1)

        return token_saliency, start_index, end_index, original_start_prob, original_end_prob

    def reorder_alignment_with_NE(self, alignment, ents):
        offset = 0
        for ent in ents:
            start, end = ent.start - offset, ent.end - offset
            alignment = alignment[:start] + [[list(chain(*alignment[start: end])), "N"]] + alignment[end:]
            offset += (end - start - 1)
        ''' there can be no matched result from spacy in BERT tokenizer such as "", we set the index as a empty list
        '''
        alignment = [a if len(a) > 0 and a[-1] == "N" else [a, "W"] for a in alignment]
        return alignment

    def get_spacy_bert_token_alignment(self, context_spacy_text, question_spacy_text, tokenized_samples,
                                       context_lengths, question_lengths):
        context_spacy_to_bert, context_spacy_token_list, question_spacy_to_bert = [], [], []
        tokenized_questions = [tokenized_samples.encodings[i].tokens[1: question_lengths[i]] for i in
                               range(question_lengths.size(0))]
        tokenized_contexts = [
            tokenized_samples.encodings[i].tokens[question_lengths[i] + 1: context_lengths[i] + question_lengths[i]]
            for i in range(context_lengths.size(0))]
        for x, y in zip(context_spacy_text, tokenized_contexts):
            tokenized_spacy_text = [t.text.strip() for t in x]
            spacy_to_bert, _ = spacy_alignments.get_alignments(tokenized_spacy_text, y)
            if self.use_NE:
                spacy_to_bert = self.reorder_alignment_with_NE(spacy_to_bert, x.ents)
            else:
                spacy_to_bert = [[a, "W"] for a in spacy_to_bert]

            context_spacy_to_bert.append(spacy_to_bert)
        for x, y in zip(question_spacy_text, tokenized_questions):
            tokenized_spacy_text = [t.text for t in x]
            spacy_to_bert, _ = spacy_alignments.get_alignments(tokenized_spacy_text, y)
            if self.use_NE:
                spacy_to_bert = self.reorder_alignment_with_NE(spacy_to_bert, x.ents)
            else:
                spacy_to_bert = [[a, "W"] for a in spacy_to_bert]
            question_spacy_to_bert.append(spacy_to_bert)
        return context_spacy_to_bert, question_spacy_to_bert

    def get_synonym_candidates(self, token):
        def get_wordnet_pos(token):
            pos = token.tag_[0].lower()
            if pos in ["r", "n", "v"]:  # adv, noun, verb
                return pos
            elif pos == "j":
                return "a"  # adj

        def _synonym_prefilter_fn(token, synonym):
            if ((synonym.lemma == token.lemma) or (  # token and synonym are the same
                    synonym.tag != token.tag) or (  # the pos of the token synonyms are different
                    token.text.lower() == 'be')):  # token is be
                return False
            else:
                return True

        candidates = set()
        if token.tag_ in SUPPORT_POS_TAG:
            wordnet_pos = get_wordnet_pos(token)
            wordnet_synonyms = []

            synsets = wordnet.synsets(token.text, pos=wordnet_pos)
            for synset in synsets:
                wordnet_synonyms.extend(synset.lemmas())
            synonyms = list(self.spacy_nlp.pipe([w.name().replace('_', ' ') for w in wordnet_synonyms], batch_size=100))
            synonyms = [s[0] for s in synonyms]
            synonyms = filter(partial(_synonym_prefilter_fn, token), synonyms)
            for synonym in synonyms:
                candidates.add(synonym.text)
            if len(candidates) > self.max_candidate_num:
                candidates = set(random.sample(candidates, self.max_candidate_num))
        return list(candidates)

    def get_named_entity_candidates(self, ent):
        ent_candidates = []
        if self.ent_dict.__contains__(ent.label_):
            ent_candidates = [e for e in random.sample(self.ent_dict[ent.label_], self.max_candidate_num)]
            while ent.text in ent_candidates:
                ent_candidates = [e for e in random.sample(self.ent_dict[ent.label_], self.max_candidate_num)]
        return ent_candidates

    def convert_token_saliency_to_word_saliency(self, token_saliency, context_spacy_to_bert_alignment,
                                                question_spacy_to_bert_alignment, context_lengths, question_lengths):
        context_word_saliency, question_word_saliency = [], []
        for i in range(len(context_spacy_to_bert_alignment)):
            cur_word_saliency = None
            cur_token_saliency = F.softmax(
                token_saliency[i, question_lengths[i] + 1: question_lengths[i] + context_lengths[i]], dim=0)
            for j in range(len(context_spacy_to_bert_alignment[i])):
                ''' there is no matched result in BERT tokenizer for spacy token, we set a zero value
                '''
                if len(context_spacy_to_bert_alignment[i][j][0]) == 0:
                    saliency_value = torch.tensor(0)
                else:
                    saliency_value = torch.mean(
                        cur_token_saliency[
                        context_spacy_to_bert_alignment[i][j][0][0]: context_spacy_to_bert_alignment[i][j][0][-1] + 1])
                if cur_word_saliency is None:
                    cur_word_saliency = saliency_value.unsqueeze(0)
                else:
                    cur_word_saliency = torch.cat((cur_word_saliency, saliency_value.unsqueeze(0)), dim=0)
            context_word_saliency.append(cur_word_saliency)
        for i in range(len(question_spacy_to_bert_alignment)):
            cur_word_saliency = None
            cur_token_saliency = F.softmax(token_saliency[i, 1: question_lengths[i]], dim=0)
            for j in range(len(question_spacy_to_bert_alignment[i])):
                if len(question_spacy_to_bert_alignment[i][j][0]) == 0:
                    saliency_value = torch.tensor(0)
                else:
                    saliency_value = torch.mean(
                        cur_token_saliency[
                        question_spacy_to_bert_alignment[i][j][0][0]: question_spacy_to_bert_alignment[i][j][0][
                                                                          -1] + 1])
                if cur_word_saliency is None:
                    cur_word_saliency = saliency_value.unsqueeze(0)
                else:
                    cur_word_saliency = torch.cat((cur_word_saliency, saliency_value.unsqueeze(0)), dim=0)
            question_word_saliency.append(cur_word_saliency)
        return context_word_saliency, question_word_saliency

    ''' some smaples in a batch may share the the context so zip them can promote the efficiency
    '''

    def zip_context_alignment(self, context_spacy_to_bert_alignment, context_spacy_text):
        zipped_context_spacy_to_bert_alignment, original_to_zip_index, zipped_context_spacy_text = [], [], []
        cur_zip_index = -1
        former_alignment = None
        for i, alignment in enumerate(context_spacy_to_bert_alignment):
            if former_alignment is None or former_alignment != alignment:
                zipped_context_spacy_to_bert_alignment.append(alignment)
                cur_zip_index += 1
                former_alignment = alignment
                zipped_context_spacy_text.append(context_spacy_text[i])
            original_to_zip_index.append(cur_zip_index)
        return zipped_context_spacy_to_bert_alignment, original_to_zip_index, zipped_context_spacy_text

    def get_replaced_inputs(self, model_inputs, context_length, candidate_ids, target_pos, target_start, target_end,
                            offset, q_length):
        target_start += offset
        target_end += offset
        model_inputs["input_ids"] = torch.cat((model_inputs["input_ids"][: target_start + q_length],
                                               candidate_ids, model_inputs["input_ids"][q_length + target_end + 1:]))
        model_inputs["attention_mask"] = torch.cat(
            (model_inputs["attention_mask"][: target_start + q_length],
             model_inputs["attention_mask"][q_length + target_start: q_length + target_start + 1].repeat(
                 candidate_ids.size(0)),
             model_inputs["attention_mask"][q_length + target_end + 1:])
        )
        if model_inputs.__contains__("token_type_ids"):
            model_inputs["token_type_ids"] = torch.cat(
                (model_inputs["token_type_ids"][: q_length + target_start],
                 model_inputs["token_type_ids"][q_length + target_start: q_length + target_start + 1].repeat(
                     candidate_ids.size(0)),
                 model_inputs["token_type_ids"][q_length + target_end + 1:])
            )
        if torch.min(model_inputs["answer_starts"]) > target_end + q_length:
            model_inputs["answer_starts"] = model_inputs["answer_starts"] + (
                    candidate_ids.size(0) - len(target_pos))
            model_inputs["answer_ends"] = model_inputs["answer_ends"] + (
                    candidate_ids.size(0) - len(target_pos))
        # offset = (candidate_ids.size(0) - len(target_pos))
        context_length += (candidate_ids.size(0) - len(target_pos))
        return model_inputs, context_length

    def get_replaced_data_and_ops(self, tokenized_samples, answer_starts, answer_ends, context_candidates,
                                  original_to_zipped_index, question_lengths):
        replaced_data = []
        replace_ops = []
        for i in range(len(original_to_zipped_index)):
            original_input_ids, original_attention_mask = \
                tokenized_samples.data["input_ids"][i], tokenized_samples.data["attention_mask"][i]
            original_token_type_ids = None
            if tokenized_samples.data.__contains__("token_type_ids"):
                original_token_type_ids = tokenized_samples.data["token_type_ids"][i]
            cur_candidates = context_candidates[original_to_zipped_index[i]]
            cur_answer_starts, cur_answer_ends = answer_starts[i], answer_ends[i]
            question_length = question_lengths[i]
            for j, candidate in enumerate(cur_candidates):
                if candidate.size() != 0:
                    target_start, target_end = min(candidate.target_pos) + 1, max(candidate.target_pos) + 1
                    if torch.min(cur_answer_starts) - self.forbidden_span > (target_end + question_length) or \
                            torch.max(cur_answer_ends) + self.forbidden_span < (target_start + question_length):
                        tokenized_candidates = self.tokenizer(candidate.candidate_texts, add_special_tokens=False)
                        for candidate_i in range(candidate.size()):
                            candidate_ids = torch.tensor(tokenized_candidates.data["input_ids"][candidate_i])
                            cur_inputs = {"input_ids": original_input_ids, "attention_mask": original_attention_mask,
                                          "answer_starts": cur_answer_starts, "answer_ends": cur_answer_ends,
                                          "sample_id": i}
                            if original_token_type_ids is not None:
                                cur_inputs["token_type_ids"] = original_token_type_ids
                            cur_inputs, _ = self.get_replaced_inputs(cur_inputs, 0, candidate_ids, candidate.target_pos,
                                                                     target_start, target_end, 0, question_length)
                            replaced_data.append(cur_inputs)
                            replace_ops.append(ReplaceOp(i, j,
                                                         candidate=candidate.candidate_texts[candidate_i],
                                                         original_token_pos=[c + 1 for c in candidate.target_pos],
                                                         new_token_ids=candidate_ids,
                                                         start_char=candidate.start_char,
                                                         end_char=candidate.end_char,
                                                         original_text=candidate.original_text))
        return replaced_data, replace_ops

    def evaluate_candidate_weight(self, tokenized_samples, context_spacy_to_bert_alignment,
                                  context_spacy_text, original_to_zipped_index, answer_starts, answer_ends,
                                  original_start_prob, original_end_prob, question_lengths):
        context_candidates = []
        for i in range(len(context_spacy_to_bert_alignment)):
            cur_context_alignment = context_spacy_to_bert_alignment[i]
            cur_spacy_text = context_spacy_text[i]
            ner_index = 0
            cur_context_candidates = []
            spacy_token_index = 0
            for j in range(len(cur_context_alignment)):
                pos_index, word_type = context_spacy_to_bert_alignment[i][j][0], context_spacy_to_bert_alignment[i][j][
                    1]
                if len(pos_index) == 0:
                    cur_context_candidates.append(
                        ReplaceCandidate([], pos_index, j, word_type, -1, -1, None))
                else:
                    if word_type == "N":
                        cur_root = cur_spacy_text.ents[ner_index]
                        candidates = self.get_named_entity_candidates(cur_root)
                        ner_index += 1
                        spacy_token_index += (cur_root.end - cur_root.start)
                        start_char, end_char = cur_root.start_char, cur_root.end_char
                    else:
                        cur_root = cur_spacy_text[spacy_token_index]
                        candidates = self.get_synonym_candidates(cur_root)
                        spacy_token_index += 1
                        start_char, end_char = cur_root.idx, cur_root.idx + len(cur_root)
                    cur_context_candidates.append(
                        ReplaceCandidate(candidates, pos_index, j, word_type, start_char, end_char, cur_root.text))
            context_candidates.append(cur_context_candidates)
        replaced_data, replace_ops = self.get_replaced_data_and_ops(
            tokenized_samples, answer_starts, answer_ends, context_candidates, original_to_zipped_index,
            question_lengths)

        replaced_dataloader = DataLoader(replaced_data, batch_size=self.batch_size,
                                         collate_fn=self.replace_collate_func)
        all_candidate_weights = None
        for batch in replaced_dataloader:
            for k, v in batch.items():
                batch[k] = v.to(self.device)
            answer_starts = batch.pop("answer_starts")
            answer_ends = batch.pop("answer_ends")
            sample_ids = batch.pop("sample_ids")
            new_start_prob, new_end_prob = self.saliency_model(batch, answer_starts, answer_ends)
            selected_ori_start_prob = torch.index_select(original_start_prob, 0, sample_ids)
            selected_ori_end_prob = torch.index_select(original_end_prob, 0, sample_ids)
            weights = (selected_ori_start_prob - new_start_prob) + (selected_ori_end_prob - new_end_prob)
            if all_candidate_weights is None:
                all_candidate_weights = weights
            else:
                all_candidate_weights = torch.cat((all_candidate_weights, weights), dim=0)
        return all_candidate_weights, replace_ops

    def rank_word_weight(self, all_candidate_weights, replace_ops, context_spacy_to_bert_alignment):
        ops_ranked_in_word = [[ReplaceOp(i, j) for j in range(len(context_spacy_to_bert_alignment[i]))]
                              for i in range(len(context_spacy_to_bert_alignment))]
        if len(replace_ops) > 0:
            replace_ops[0].word_weight = all_candidate_weights[0].item()
            former_list = [replace_ops[0]]
            for i in range(len(replace_ops)):
                replace_ops[i].word_weight = all_candidate_weights[i].item()
                if len(former_list) == 0:
                    former_list = [replace_ops[i]]
                else:
                    if replace_ops[i].sample_id != former_list[-1].sample_id or \
                            replace_ops[i].word_id != former_list[-1].word_id:
                        former_list = sorted(former_list, key=lambda o: o.word_weight, reverse=True)
                        most_import_op = former_list[0]
                        ops_ranked_in_word[most_import_op.sample_id][most_import_op.word_id] = most_import_op
                        former_list = [replace_ops[i]]
                    else:
                        former_list.append(replace_ops[i])
        return ops_ranked_in_word

    def rank_position(self, context_word_saliency, ops_ranked_in_word, original_to_zipped_index):
        for i in range(len(ops_ranked_in_word)):
            for j in range(len(ops_ranked_in_word[i])):
                ops_ranked_in_word[i][j].word_weight = ops_ranked_in_word[i][j].word_weight * \
                                                       context_word_saliency[original_to_zipped_index[i]][j].item()
            ops_ranked_in_word[i] = sorted(ops_ranked_in_word[i], key=lambda o: o.word_weight, reverse=True)
        return ops_ranked_in_word

    def evaluate_each_operation(self, ranked_ops, model_input, answer_starts, answer_ends, answer_texts,
                                context_lengths, question_lengths):
        sample_replace_ops = []
        replaced_answer_starts, replaced_answer_ends = [], []
        all_logits, all_replaced_texts = [], []
        all_model_pred_texts = []
        with torch.no_grad():
            for i in range(len(ranked_ops)):
                cur_sample_replace_ops = []
                cur_model_input = {
                    "input_ids": deepcopy(model_input["input_ids"][i]),
                    "attention_mask": deepcopy(model_input["attention_mask"][i]),
                    "answer_starts": deepcopy(answer_starts[i]),
                    "answer_ends": deepcopy(answer_ends[i])
                }
                if model_input.__contains__("token_type_ids"):
                    cur_model_input["token_type_ids"] = deepcopy(model_input["token_type_ids"][i])
                for k, v in cur_model_input.items():
                    cur_model_input[k] = v.to(self.device)
                pos_offset = []
                answer_text = answer_texts[i]
                context_length = context_lengths[i].item()
                question_length = question_lengths[i].item()
                logits, model_pred_text = None, None
                for j in range(len(ranked_ops[i])):
                    cur_op = ranked_ops[i][j]
                    if cur_op.candidate is None:
                        cur_sample_replace_ops = []
                        break
                    target_start, target_end = min(cur_op.original_token_pos), max(cur_op.original_token_pos)
                    if len(pos_offset) == 0 or pos_offset[0][0] >= target_end:
                        offset = 0
                        pos_offset.insert(0,
                                          (target_start, cur_op.new_token_ids.size(0) - len(cur_op.original_token_pos)))
                    else:
                        for offset_i in range(len(pos_offset)):
                            if pos_offset[offset_i][0] >= target_start:
                                break
                        offset = sum(x[1] for x in pos_offset[:offset_i])
                        pos_offset.insert(offset_i + 1,
                                          (target_start, cur_op.new_token_ids.size(0) - len(cur_op.original_token_pos)))
                    cur_model_input, context_length = self.get_replaced_inputs(
                        cur_model_input, context_length, cur_op.new_token_ids.to(self.device),
                        cur_op.original_token_pos,
                        target_start, target_end, offset, question_length)
                    cur_answer_starts, cur_answer_ends = cur_model_input.pop("answer_starts"), \
                                                         cur_model_input.pop("answer_ends")
                    model_pred_text, logits = self.determine_model(cur_model_input, context_length, cur_answer_starts,
                                                                   cur_answer_ends, answer_text)
                    cur_sample_replace_ops.append(cur_op)
                    cur_model_input["answer_starts"] = cur_answer_starts
                    cur_model_input["answer_ends"] = cur_answer_ends
                    if model_pred_text is not None:
                        break
                all_logits.append(logits)
                all_model_pred_texts.append(model_pred_text)
                replaced_text = self.tokenizer.decode(cur_model_input["input_ids"])
                if "[SEP]" in replaced_text:
                    replaced_text = replaced_text.split("[SEP]")[1]
                else:
                    replaced_text = replaced_text.split("</s>")[2]
                all_replaced_texts.append(replaced_text)
                sample_replace_ops.append(cur_sample_replace_ops)
                replaced_answer_starts.append(cur_model_input["answer_starts"].cpu())
                replaced_answer_ends.append(cur_model_input["answer_ends"].cpu())
        return sample_replace_ops, replaced_answer_starts, replaced_answer_ends, all_logits, all_replaced_texts, \
               all_model_pred_texts

    def adversarial_paraphrase(self, batch):
        sample_batch, tokenized_samples = batch["sample_batch"], batch["tokenized_samples"]

        input_lengths, context_lengths, question_lengths = batch["lengths"][0], batch["lengths"][1], batch["lengths"][2]
        answer_starts, answer_ends = batch["answer_starts"], batch["answer_ends"]
        model_input = {"input_ids": tokenized_samples.data["input_ids"],
                       "attention_mask": tokenized_samples.data["attention_mask"]}
        if tokenized_samples.data.__contains__("token_type_ids"):
            model_input["token_type_ids"] = tokenized_samples.data["token_type_ids"]
        if tokenized_samples.data["input_ids"].size(0) > answer_starts.size(0):
            return [], [], []
        '''Calculate the token saliency for each token using the model
        '''
        token_saliency, start_index, end_index, original_start_prob, original_end_prob = \
            self.evaluate_token_saliency(model_input, answer_starts, answer_ends)

        '''Using spacy to process each context and question, and get the alignment between spacy words and model tokens
        '''
        context_spacy_text = list(self.spacy_nlp.pipe(sample_batch["context"], batch_size=100))
        question_spacy_text = list(self.spacy_nlp.pipe(sample_batch["question"], batch_size=100))
        context_spacy_to_bert_alignment, question_spacy_to_bert_alignment = self.get_spacy_bert_token_alignment(
            context_spacy_text, question_spacy_text, tokenized_samples, context_lengths, question_lengths)
        zipped_context_spacy_to_bert_alignment, original_to_zipped_index, zipped_context_spacy_text = \
            self.zip_context_alignment(context_spacy_to_bert_alignment, context_spacy_text)

        ''' convert token saliency to spacy token saliency
        '''
        context_word_saliency, question_word_saliency = \
            self.convert_token_saliency_to_word_saliency(token_saliency, zipped_context_spacy_to_bert_alignment,
                                                         question_spacy_to_bert_alignment, context_lengths,
                                                         question_lengths)

        ''' evaluate the spacy word weight using synonym and NE replacement
        '''
        all_candidate_weights, replace_ops = self.evaluate_candidate_weight(
            tokenized_samples, zipped_context_spacy_to_bert_alignment, zipped_context_spacy_text,
            original_to_zipped_index, answer_starts, answer_ends, original_start_prob, original_end_prob,
            question_lengths)
        ops_ranked_in_word = self.rank_word_weight(all_candidate_weights, replace_ops, context_spacy_to_bert_alignment)

        ''' rerank the candidate in each position according to both word weight and saliency
        '''
        ranked_ops = self.rank_position(context_word_saliency, ops_ranked_in_word, original_to_zipped_index)

        ''' evaluate the replacement in each position to obtain the final replacement 
        '''
        sample_replace_ops, replaced_answer_starts, replaced_answer_ends, logits, all_texts, all_model_pred_texts = \
            self.evaluate_each_operation(ranked_ops, model_input, answer_starts, answer_ends,
                                         sample_batch["answer_text"], context_lengths, question_lengths)
        return sample_replace_ops, replaced_answer_starts, replaced_answer_ends, logits, all_texts, all_model_pred_texts

    def generate_adversarial_sample(self, sample_batch, sample_replace_ops):
        adversarial_samples = []
        fail_cnt = 0
        for i in range(len(sample_replace_ops)):
            cur_ops = sample_replace_ops[i]
            if len(cur_ops) > 0:
                cur_ops = sorted(cur_ops, key=lambda o: o.start_char)
                context = sample_batch["context"][i]
                answer_start = sample_batch["answer_start"][i]
                answer_texts = sample_batch["answer_text"][i]
                char_offset = 0
                for op in cur_ops:
                    context = context[: op.start_char + char_offset] + op.candidate + context[
                                                                                      op.end_char + char_offset:]
                    for j in range(len(answer_start)):
                        if answer_start[j] > op.end_char + char_offset:
                            answer_start[j] += len(op.candidate) - (op.end_char - op.start_char)
                    char_offset += len(op.candidate) - (op.end_char - op.start_char)
                answers = [{"answer_start": answer_start[i], "text": answer_texts[i]} for i in range(len(answer_start))]
                qas = {"question": sample_batch["question"][i], "id": sample_batch["id"][i] + "-adv",
                       "answers": answers}
                adversarial_samples.append({"context": context, "qas": [qas]})
            else:
                fail_cnt += 1
        return adversarial_samples, fail_cnt

    def fool_qa_model(self):
        all_sample_replace_ops, all_replaced_answer_starts, all_replaced_answer_ends = [], [], []
        all_adversarial_samples = []
        all_logits, all_texts, all_pred_texts = [], [], []
        fail_cnt = 0
        tqdm_data = tqdm(self.data_loader, desc="generating")
        tqdm_data.set_postfix({"success": 0})
        success_cnt = 0
        for i, batch in enumerate(tqdm_data):
            # with open('AttackResults/PWWS_raw_ops.pickle', 'rb') as f:
            #     ops = [pickle.load(f)[0][3445]]
            # adversarial_samples = self.generate_adversarial_sample(batch["sample_batch"], ops)
            try:
                sample_replace_ops, replaced_answer_starts, replaced_answer_ends, logits, texts, pred_texts = \
                    self.adversarial_paraphrase(batch)
                all_logits.extend([logits[i] for i in range(len(logits)) if len(sample_replace_ops[i]) != 0])
                all_texts.extend([texts[i] for i in range(len(texts)) if len(sample_replace_ops[i]) != 0])
                all_pred_texts.extend(
                    [pred_texts[i] for i in range(len(pred_texts)) if len(sample_replace_ops[i]) != 0])
                all_sample_replace_ops.extend(sample_replace_ops)
                all_replaced_answer_starts.extend(replaced_answer_starts)
                all_replaced_answer_ends.extend(replaced_answer_ends)
                adversarial_samples, tmp_fail_cnt = \
                    self.generate_adversarial_sample(batch["sample_batch"], sample_replace_ops)
                all_adversarial_samples.extend(adversarial_samples)
                success_cnt += len(adversarial_samples)
                tqdm_data.set_postfix({"success": success_cnt})
                fail_cnt += tmp_fail_cnt
            except Exception:
                print(traceback.format_exc())
            if i >= 100:
                break
        for i in range(len(all_texts)):
            adversarial_tokens = self.tokenizer.tokenize(all_adversarial_samples[i]["context"].lower())
            orig_tokens = self.tokenizer.tokenize(all_texts[i])
            if len(adversarial_tokens) != len(orig_tokens):
                print("error " + str(i))
            else:
                for x, y in zip(adversarial_tokens, orig_tokens):
                    if x != y:
                        print("error " + str(i) + " " + x + "-" + y)
        all_adversarial_samples = [{"title": "adv", "paragraphs": all_adversarial_samples}]
        with open(self.output_file_name, "w", encoding="utf-8") as f:
            json.dump({"data": all_adversarial_samples, "version": "1.1", "length": len(all_adversarial_samples)}, f)
        ops_file_name = self.output_file_name[:self.output_file_name.rfind(".")] + "_ops.pickle"
        with open(ops_file_name, "wb") as f:
            pickle.dump([all_sample_replace_ops, all_replaced_answer_starts, all_replaced_answer_ends], f)
        all_texts_file_name = self.output_file_name[:self.output_file_name.rfind(".")] + "_all_texts.json"
        with open(all_texts_file_name, "w", encoding="utf-8") as f:
            json.dump(all_texts, f)
        with open("logits.pickle", "wb") as f:
            pickle.dump(all_logits, f)


if __name__ == "__main__":
    args = parse_args()
    model_fooler = ModelFooler(args)
    model_fooler.fool_qa_model()
