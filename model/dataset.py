import torch
from torch.utils.data import Dataset

class EvaluateDataset(Dataset):
    def __init__(self, samples):
        self.data = samples.data
        if samples.encodings is not None:
            self.length = len(samples.encodings)
        else:
            self.length = len(self.data["input_ids"])

    def __getitem__(self, index):
        item = {}
        for k, v in self.data.items():
            item[k] = torch.tensor(v[index])
        return item

    def __len__(self):
        return self.length