import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from utils.common_utils import pad_sequence
import numpy as np

MAX_LEN = 512
SQUAD1_AVG_PPL = 105.9295
SQUAD2_AVG_PPL = 80.008232
NEWSQA_AVG_PPL = 92.715476
NQ_AVG_PPL = 157.16346

DATASET_PPL = {"squad": SQUAD1_AVG_PPL, "squad2": SQUAD2_AVG_PPL, "newsqa": NEWSQA_AVG_PPL, "nq": NQ_AVG_PPL}

class PPLEvaluator:
    def __init__(self, model_path, batch_size):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
        self.batch_size = batch_size
        self.model.to(self.device)

    def collate_fn(self, data):
        data_dict = {}
        data_dict["input_ids"] = pad_sequence([d["input_ids"] for d in data], batch_first=True, padding_value=0)
        data_dict["labels"] = pad_sequence([d["input_ids"] for d in data], batch_first=True, padding_value=-100)
        data_dict["attention_mask"] = pad_sequence([d["attention_mask"] for d in data], batch_first=True, padding_value=0)
        return data_dict

    def calculate_ppl(self, texts, dataset_type=None):
        all_ppls = []
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(text, return_tensors="pt", max_length=512)
                inputs["labels"] = inputs["input_ids"].clone()
                for k, v in inputs.items():
                    inputs[k] = v.to(self.device)
                output = self.model(**inputs)
                if dataset_type is None:
                    all_ppls.append(torch.exp(output[0]).item())
                else:
                    all_ppls.append(torch.exp(output[0]).item() / DATASET_PPL[dataset_type])
        return np.array(all_ppls)

    def calculate_norm_ppl(self, ref, texts, dataset_type=None):
        ref_ppl = self.calculate_ppl([ref], dataset_type)
        texts_ppls = self.calculate_ppl(texts, dataset_type)
        return texts_ppls / ref_ppl

if __name__ == "__main__":
    import json
    import nltk
    ppl_cal = PPLEvaluator("../../model_hubs/gpt2_small", 8)
    ppl_cal.calculate_norm_ppl("i like this thing", ["i like it", "i do not like it"])
    with open("../data/nq/nq-dev.json", "r", encoding="utf-8") as f:
        data = json.load(f)["data"]
    all_sents = []
    for doc in data:
        for para in doc["paragraphs"]:
            all_sents.extend(nltk.sent_tokenize(para["context"]))
    all_ppls = ppl_cal.calculate_ppl(all_sents)
    new_all_ppls = []
    for ppl in all_ppls:
        if not torch.isnan(ppl):
            new_all_ppls.append(ppl)
    print(torch.mean(torch.tensor(new_all_ppls)).item())
    print("1")