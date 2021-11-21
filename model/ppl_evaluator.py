import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from utils.common_utils import pad_sequence
from torch.utils.data.dataloader import DataLoader
from model.dataset import EvaluateDataset

MAX_LEN = 512

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

    def calculate_ppl(self, texts):
        all_ppls = []
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(text, return_tensors="pt")
                inputs["labels"] = inputs["input_ids"].clone()
                for k, v in inputs.items():
                    inputs[k] = v.to(self.device)
                output = self.model(**inputs)
                all_ppls.append(torch.exp(output[0]))
        return all_ppls