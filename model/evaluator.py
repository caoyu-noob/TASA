import torch
import torch.nn.functional as F
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from utils.common_utils import pad_sequence
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import LambdaLR
from model.dataset import EvaluateDataset
from allennlp.predictors import Predictor
from bidaf import squad_qa

import transformers
from accelerate import Accelerator
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    # SchedulerType,
    default_data_collator,
    # get_scheduler,
    set_seed,
)

SPANBERT_MAX_LENGTH = 512

class QAEvaluator():
    def __init__(self, model_name_or_path, metric_path, model_type, batch_size=8, max_seq_len=384, doc_stride=128):
        self.model_type = model_type
        if self.model_type == "bert" or self.model_type == "spanbert":
            self.accelerator = Accelerator()
            self.metric = load_metric(metric_path)
            self.config = AutoConfig.from_pretrained(model_name_or_path)
            if model_type == "bert":
                self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
                self.max_seq_len = min(max_seq_len, self.tokenizer.model_max_length)
            elif model_type == "spanbert":
                self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, do_lower_case=False)
                self.max_seq_len = SPANBERT_MAX_LENGTH
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name_or_path)
        elif self.model_type == "bidaf":
            self.device_id = -1
            if torch.cuda.device_count() > 0:
                self.device_id = 0
            self.predictor = Predictor.from_path(model_name_or_path, cuda_device=self.device_id)
            self.dataset_reader = self.predictor._dataset_reader
        self.doc_stride = doc_stride
        self.batch_size = batch_size

    def prepare_features(self, beam_contexts, question, answer_starts=None, answer_ends=None, single_question=True):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        pad_on_right = self.tokenizer.padding_side == "right"
        if single_question:
            questions = [question for _ in range(len(beam_contexts))]
        else:
            questions = question
        tokenized_examples = self.tokenizer(
            questions,
            beam_contexts,
            truncation="only_second" if pad_on_right else "only_first",
            max_length=self.max_seq_len,
            stride=self.doc_stride,
            return_overflowing_tokens=False,
            return_offsets_mapping=True,
            padding=False,
        )

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        # tokenized_examples["example_id"] = []
        #
        # for i in range(len(tokenized_examples["input_ids"])):
        #     tokenized_examples["example_id"].append(i)
        offset_mapping = tokenized_examples.pop("offset_mapping")
        start_positions, end_positions = [], []
        if answer_starts is not None or answer_ends is not None:
            for i, offsets in enumerate(offset_mapping):
                input_ids = tokenized_examples["input_ids"][i]
                start_char, end_char = answer_starts[i], answer_ends[i]
                cls_index = input_ids.index(self.tokenizer.cls_token_id)
                # Grab the sequence corresponding to that example (to know what is the context and what is the question).
                sequence_ids = tokenized_examples.sequence_ids(i)

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    start_positions.append(cls_index)
                    end_positions.append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    start_positions.append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    end_positions.append(token_end_index + 1)
        return tokenized_examples, start_positions, end_positions

    def validation(self, model, eval_dataloader, softmax=False, desc=None):
        # Validation
        all_start_logits, all_end_logits = None, None
        data_loader = eval_dataloader
        if desc is not None:
            data_loader = tqdm(eval_dataloader, desc=desc)
        for step, batch in enumerate(data_loader):
            with torch.no_grad():
                outputs = model(**batch)
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits
                logits_mask = -1e5 * (torch.ones_like(batch["attention_mask"]) - batch["attention_mask"])

                start_logits = self.accelerator.gather(start_logits).detach().cpu()
                end_logits = self.accelerator.gather(end_logits).detach().cpu()
                if softmax:
                    start_logits = torch.softmax(start_logits + logits_mask, dim=-1)
                    end_logits = torch.softmax(end_logits + logits_mask, dim=-1)
                if all_start_logits is None:
                    all_start_logits = start_logits
                    all_end_logits = end_logits
                else:
                    if all_start_logits.size(1) > start_logits.size(1):
                        start_logits = F.pad(start_logits, pad=(0, all_start_logits.size(1) - start_logits.size(1)),
                                             mode='constant', value=0)
                        end_logits = F.pad(end_logits, pad=(0, all_end_logits.size(1) - end_logits.size(1)),
                                             mode='constant', value=0)
                    elif all_start_logits.size(1) < start_logits.size(1):
                        all_start_logits = F.pad(all_start_logits, pad=(0, start_logits.size(1) - all_start_logits.size(1)),
                                             mode='constant', value=0)
                        all_end_logits = F.pad(all_end_logits, pad=(0, end_logits.size(1) - all_end_logits.size(1)),
                                           mode='constant', value=0)
                    all_start_logits = torch.cat((all_start_logits, start_logits))
                    all_end_logits = torch.cat((all_end_logits, end_logits))

        return all_start_logits, all_end_logits

    def collate_fn(self, data):
        data_dict = {}
        for k in data[0].keys():
            data_dict[k] = pad_sequence([d[k] for d in data], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        return data_dict

    def _bert_evaluate_sample_logits(self,  beam_contexts, question, answer_start_char, answer_end_char,
                             single_question, desc):
        tokenized_examples, start_positions, end_positions = self.prepare_features(beam_contexts, question,
                                                                                   answer_start_char, answer_end_char,
                                                                                   single_question)
        dataset = EvaluateDataset(tokenized_examples)
        data_loader = DataLoader(dataset, num_workers=0, batch_size=self.batch_size, collate_fn=self.collate_fn)
        self.model, data_loader = self.accelerator.prepare(self.model, data_loader)
        all_start_logits, all_end_logits = self.validation(self.model, data_loader, desc=desc)
        answer_start_logits = all_start_logits.gather(1, torch.tensor([start_positions]).t())
        answer_end_logits = all_end_logits.gather(1, torch.tensor([end_positions]).t())
        return all_start_logits, all_end_logits, answer_start_logits, answer_end_logits, start_positions, end_positions

    def _bidaf_evaluate_sample_logits(self,  beam_contexts, question, answer_start_char, answer_end_char,
                             single_question, desc):
        instance_list = []
        if single_question:
            questions = [question for _ in range(len(beam_contexts))]
        else:
            questions = question
        for context, question, answer_start, answer_end in \
                zip(beam_contexts, questions, answer_start_char, answer_end_char):
            instance_list.append(
                self.dataset_reader.text_to_instance(question, context, char_spans=[[answer_start, answer_end]],
                                                     answer_texts=[""]))
        answer_start_logits, answer_end_logits = [], []
        start_positions, end_positions = [], []
        best_answer_starts, best_answer_ends = [], []

        index_dataloader = range(0, len(instance_list), self.batch_size)
        if desc is not None:
            index_dataloader = tqdm(index_dataloader, desc=desc)
        for index in index_dataloader:
            batch = instance_list[index: index + self.batch_size]
            outputs = self.predictor.predict_batch_instance(batch)
            start_logits = torch.tensor([x["span_start_logits"] for x in outputs])
            end_logits = torch.tensor([x["span_end_logits"] for x in outputs])
            tmp_start_positions = [x.fields["span_start"].sequence_index for x in batch]
            tmp_end_positions = [x.fields["span_end"].sequence_index for x in batch]
            start_positions.extend(tmp_start_positions)
            end_positions.extend(tmp_end_positions)
            best_answer_starts.append(torch.argmax(start_logits, dim=1))
            best_answer_ends.append(torch.argmax(end_logits, dim=1))
            answer_starts, answer_ends = torch.tensor(tmp_start_positions), torch.tensor(tmp_end_positions)
            answer_start_logits.append(start_logits.gather(1, answer_starts.unsqueeze(0)).squeeze(0))
            answer_end_logits.append(end_logits.gather(1, answer_ends.unsqueeze(0)).squeeze(0))
            index += self.batch_size
        answer_start_logits = torch.cat(answer_start_logits, dim=0)
        answer_end_logits = torch.cat(answer_end_logits, dim=0)
        best_answer_starts = torch.cat(best_answer_starts, dim=0)
        best_answer_ends = torch.cat(best_answer_ends, dim=0)
        return answer_start_logits, answer_end_logits, start_positions, end_positions, best_answer_starts,\
               best_answer_ends

    def evaluate_sample_logits(self, beam_contexts, question, answer_start_char, answer_end_char, single_question=True,
                               return_best_position=False, desc=None):
        '''

        :param beam_contexts (obj: list): the list of a series of context
        :param question (obj: string): the string of question
        :param answer_start_pos (obj list): the list of the answer start char positions in the context
        :param answer_end_pos (obj list): the list of the answer end char positions in the context
        :return: the predicted logits on the answer start and end positions
        '''
        if self.model_type == "bert" or self.model_type == "spanbert":
            all_start_logits, all_end_logits, answer_start_logits, answer_end_logits, start_positions, end_positions = \
                self._bert_evaluate_sample_logits(beam_contexts, question, answer_start_char, answer_end_char,
                                                  single_question, desc)
            if return_best_position:
                best_start_positions = torch.argmax(all_start_logits, dim=1)
                best_end_positions = torch.argmax(all_end_logits, dim=1)
                return answer_start_logits.squeeze(-1), answer_end_logits.squeeze(-1), start_positions, end_positions, \
                       best_start_positions, best_end_positions
            return answer_start_logits.squeeze(-1), answer_end_logits.squeeze(-1)
        elif self.model_type == "bidaf":
            answer_start_logits, answer_end_logits, start_positions, end_positions, best_answer_starts, \
                    best_answer_ends = \
                self._bidaf_evaluate_sample_logits(beam_contexts, question, answer_start_char, answer_end_char,
                                                  single_question, desc)
            if return_best_position:
                return answer_start_logits, answer_end_logits, start_positions, end_positions, best_answer_starts, \
                    best_answer_ends
            return answer_start_logits, answer_end_logits

    def evaluate_whether_has_answer(self, beam_contexts, question, mode="argmax", th=0.01):
        '''
        :param beam_contexts (obj: list): the list of a series of context
        :param question (obj: string): the string of question
        :return: the predictions about whether the given samples contains a valid answer or not
        '''
        tokenized_examples, start_positions, end_positions = self.prepare_features(beam_contexts, question)
        dataset = EvaluateDataset(tokenized_examples)
        data_loader = DataLoader(dataset, num_workers=0, batch_size=self.batch_size, collate_fn=self.collate_fn)
        self.model, data_loader = self.accelerator.prepare(self.model, data_loader)
        if mode == "argmax":
            all_start_logits, all_end_logits = self.validation(self.model, data_loader)
            best_start_position = torch.argmax(all_start_logits, dim=1)
            best_end_position = torch.argmax(all_end_logits, dim=1)
            zeros = torch.zeros_like(best_start_position)
            start_has = (zeros != best_start_position)
            end_has = (zeros != best_end_position)
            return torch.logical_and(start_has, end_has)
        elif mode == "threshold":
            all_start_logits, all_end_logits = self.validation(self.model, data_loader, softmax=False)
            batch_size = all_start_logits.size(0)
            answer_start_probs = all_start_logits.gather(1, torch.zeros(batch_size, 1).t())
            answer_end_probs = all_end_logits.gather(1, torch.zeros(batch_size, 1).t())
            null_answer_probs = answer_start_probs + answer_end_probs
            return (null_answer_probs >= th)
