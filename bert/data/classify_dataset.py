import math
import random
import pkuseg
import numpy as np

from tqdm import tqdm
from bert.common.tokenizers import Tokenizer
from pretrain_config import *
from torch.utils.data import Dataset


class BertDataSet(Dataset):
    def __init__(self, corpus_path):
        self.tokenizer = Tokenizer(VocabPath)
        self.corpus_path = corpus_path
        self.descriptions = []
        self.labels = []
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line:
                    line = line.strip().replace(' ', '')
                    line = line.split(',')
                    if line[0] and line[1]:
                        self.descriptions.append(line[0][:510])
                        self.labels.append(int(line[1]))

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, item):
        output = {}
        descriptions_text = self.descriptions[item]
        label_text = self.labels[item]
        token_ids = self.__gen_input_token(descriptions_text)
        segment_ids = [1 if x else 0 for x in token_ids]
        output['input_token_ids'] = token_ids
        output['segment_ids'] = segment_ids
        output['token_ids_labels'] = label_text
        instance = {k: torch.tensor(v, dtype=torch.long) for k, v in output.items()}
        return instance

    def __gen_input_token(self, texts):
        input_token_ids = []
        input_token_ids.append(101)
        for token in texts:
            input_token_ids.append(self.tokenizer.token_to_id(token))
        input_token_ids.append(102)
        if len(input_token_ids) < SentenceLength:
            for i in range(SentenceLength - len(input_token_ids)):
                input_token_ids.append(0)
        return input_token_ids


class RobertaTestSet(Dataset):
    def __init__(self, test_path, onehot_type=False):
        self.tokenizer = Tokenizer(VocabPath)
        self.corpus_path = test_path
        self.onehot_type = onehot_type
        self.descriptions = []
        self.labels = []
        with open(test_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line:
                    line = line.strip()
                    line = line.split(',')
                    if line[0] and line[1]:
                        self.descriptions.append(line[0][:510])
                        self.labels.append(int(line[1]))

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, item):
        output = {}
        descriptions_text = self.descriptions[item]
        label_text = self.labels[item]
        token_ids = self.__gen_input_token(descriptions_text)
        segment_ids = [1 if x else 0 for x in token_ids]
        output['input_token_ids'] = token_ids
        output['segment_ids'] = segment_ids
        output['token_ids_labels'] = label_text
        instance = {k: torch.tensor(v, dtype=torch.long) for k, v in output.items()}
        return instance

    def __gen_input_token(self, texts):
        input_token_ids = []
        input_token_ids.append(101)
        for token in texts:
            input_token_ids.append(self.tokenizer.token_to_id(token))
        input_token_ids.append(102)
        if len(input_token_ids) < SentenceLength:
            for i in range(SentenceLength - len(input_token_ids)):
                input_token_ids.append(0)
        return input_token_ids
