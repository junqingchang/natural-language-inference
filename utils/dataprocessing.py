import torch
from torch.utils.data import Dataset
import json
import random

class MNLI(Dataset):
    def __init__(self, path, genre=False, snli=True):
        self.LABEL_MAP = {
                        "entailment": 0,
                        "neutral": 1,
                        "contradiction": 2,
                        "hidden": 0
                    }
        self.genre = genre
        self.snli = snli
        self.dataset = self.load_nli_data_genre(path, genre, snli)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return item['sentence1'], item['sentence2'], item['label']

    def load_nli_data_genre(self, path, genre, snli=True):
        """
        Taken from official MNLI github

        Load a specific genre's examples from MultiNLI, or load SNLI data and assign a "snli" genre to the examples.
        If the "snli" parameter is set to True, a genre label of snli will be assigned to the data. If set to true, it will overwrite the genre label for MultiNLI data.
        """
        data = []
        with open(path) as f:
            for line in f:
                loaded_example = json.loads(line)
                if loaded_example["gold_label"] not in self.LABEL_MAP:
                    continue
                loaded_example["label"] = self.LABEL_MAP[loaded_example["gold_label"]]
                if snli:
                    loaded_example["genre"] = "snli"
                if genre:
                    if loaded_example["genre"] == genre:
                        data.append(loaded_example)
                else:
                    data.append(loaded_example)
            random.shuffle(data)
        return data