import torch
from torch.utils.data import Dataset
import json
import random
from transformers import BertTokenizer
import math
from torch.nn.utils.rnn import pad_sequence


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


class BERTMNLI(Dataset):
    def __init__(self, path, genre=False, snli=True, batch_size=8, bert_type='bert-large-cased'):
        self.LABEL_MAP = {
            "entailment": 0,
            "neutral": 1,
            "contradiction": 2,
            "hidden": 0
        }
        self.genre = genre
        self.snli = snli
        self.dataset = self.load_nli_data_genre(path, genre, snli)
        self.tokenizer = BertTokenizer.from_pretrained(bert_type)
        self.batch_size = batch_size

    def __getitem__(self, idx):
        if idx < self.__len__():
            sentence = [[x['sentence1'], x['sentence2']]
                        for x in self.dataset[self.batch_size*idx:self.batch_size*idx+self.batch_size]]
            label = [x['label'] for x in self.dataset[self.batch_size *
                                                      idx:self.batch_size*idx+self.batch_size]]
        else:
            sentence = [[x['sentence1'], x['sentence2']]
                        for x in self.dataset[self.batch_size*idx:]]
            label = [x['label'] for x in self.dataset[self.batch_size*idx:]]
        sentence = self.tokenizer.batch_encode_plus(
            sentence, pad_to_max_length=True, return_token_type_ids=True, return_attention_masks=True)
        for key in sentence:
            sentence[key] = torch.tensor(sentence[key])
        return sentence, torch.tensor(label)

    def __len__(self):
        return math.ceil(len(self.dataset)/self.batch_size)

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


class LossEditedBERTMNLI(Dataset):
    def __init__(self, path, genre=False, snli=True, batch_size=8, bert_type='bert-base-cased'):
        self.LABEL_MAP = {
            "entailment": 0,
            "neutral": 1,
            "contradiction": 2,
            "hidden": 0
        }
        self.genre = genre
        self.snli = snli
        self.dataset = self.load_nli_data_genre(path, genre, snli)
        self.tokenizer = BertTokenizer.from_pretrained(bert_type)
        self.batch_size = batch_size

    def __getitem__(self, idx):
        if idx < self.__len__():
            sentence1 = [[x['sentence1'], x['sentence2']]
                        for x in self.dataset[self.batch_size*idx:self.batch_size*idx+self.batch_size]]
            label1 = [x['label'] for x in self.dataset[self.batch_size *
                                                      idx:self.batch_size*idx+self.batch_size]]
            sentence2 = [[x['sentence1'], x['sentence1']]
                        for x in self.dataset[self.batch_size*idx:self.batch_size*idx+self.batch_size]]
            label2 = [0 for x in self.dataset[self.batch_size *
                                                      idx:self.batch_size*idx+self.batch_size]]
            sentence3 = [[x['sentence2'], x['sentence2']]
                        for x in self.dataset[self.batch_size*idx:self.batch_size*idx+self.batch_size]]
            label3 = [0 for x in self.dataset[self.batch_size *
                                                      idx:self.batch_size*idx+self.batch_size]]
            sentence4 = [[x['sentence2'], x['sentence1']]
                        for x in self.dataset[self.batch_size*idx:self.batch_size*idx+self.batch_size]]
            label4 = [x['label'] for x in self.dataset[self.batch_size *
                                                      idx:self.batch_size*idx+self.batch_size]]
        else:
            sentence1 = [[x['sentence1'], x['sentence2']]
                        for x in self.dataset[self.batch_size*idx:]]
            label1 = [x['label'] for x in self.dataset[self.batch_size*idx:]]
            sentence2 = [[x['sentence1'], x['sentence1']]
                        for x in self.dataset[self.batch_size*idx:]]
            label2 = [0 for x in self.dataset[self.batch_size*idx:]]
            sentence3 = [[x['sentence2'], x['sentence2']]
                        for x in self.dataset[self.batch_size*idx:]]
            label3 = [0 for x in self.dataset[self.batch_size*idx:]]
            sentence4 = [[x['sentence2'], x['sentence1']]
                        for x in self.dataset[self.batch_size*idx:]]
            label4 = [x['label'] for x in self.dataset[self.batch_size*idx:]]

        sentences = [sentence1, sentence2, sentence3, sentence4]
        labels = [label1, label2, label3, label4]
        sentences = [self.tokenizer.batch_encode_plus(
            sentence, pad_to_max_length=True, return_token_type_ids=True, return_attention_masks=True) for sentence in sentences]
        for sentence in sentences:
            for key in sentence:
                sentence[key] = torch.tensor(sentence[key])
        for i in range(len(labels)):
            labels[i] = torch.tensor(labels[i])
        return sentences, labels

    def __len__(self):
        return math.ceil(len(self.dataset)/self.batch_size)

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
                        if len(loaded_example['sentence1'])*2+3 <= 512 or len(loaded_example['sentence2'])*2+3 <= 512: # Hard cap for BERT model
                            data.append(loaded_example)
                else:
                    if len(loaded_example['sentence1'])*2+3 <= 512 or len(loaded_example['sentence2'])*2+3 <= 512: # Hard cap for BERT model
                        data.append(loaded_example)
            random.shuffle(data)
        return data
