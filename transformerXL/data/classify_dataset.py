import math
import random
import pkuseg
import numpy as np

from tqdm import tqdm
from transformerXL.common.tokenizers import Tokenizer
from pretrain_config import *
from torch.utils.data import Dataset


class TransformerXLDataSet(Dataset):
    def __init__(self, corpus_path):
        # self.tokenizer = Tokenizer(VocabPath)
        self.corpus_path = corpus_path
        self.descriptions = []
        self.labels = []
        self.segment_length = SentenceLength - 1

        with open(Assistant, 'r', encoding='utf-8') as f:
            line = f.read().strip().split(',')
            self.cls = int(line[0])
            self.padding = self.cls + 1

        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line:
                    line = line.strip()
                    label, desc = line.split(',')
                    desc = [int(x) for x in desc.split(' ')]
                    label = int(label)
                    self.descriptions.append(desc)
                    self.labels.append(label)

    # def __gen_input_token(self, texts):
    #     input_token_ids = []
    #     input_token_ids.append(101)
    #     for token in texts:
    #         input_token_ids.append(self.tokenizer.token_to_id(token))
    #     input_token_ids.append(102)
    #     if len(input_token_ids) < SentenceLength:
    #         for i in range(SentenceLength - len(input_token_ids)):
    #             input_token_ids.append(0)
    #     return input_token_ids

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, item):
        output = {}
        descriptions_ids = self.descriptions[item]
        label_id = self.labels[item]
        # descriptions_ids = self.__gen_input_token(descriptions_ids)
        desc_segments = []
        type_segments = []
        segments_count = 0
        while True:
            if segments_count * self.segment_length < len(descriptions_ids):
                current_type = []
                current_segment = descriptions_ids[segments_count*self.segment_length:
                                                    min((segments_count+1) * self.segment_length,
                                                        len(descriptions_ids))]
                # 补全padding
                if len(current_segment) < self.segment_length:
                    for i in range(0, self.segment_length - len(current_segment)):
                        current_segment.append(self.padding)
                current_segment = [self.cls] + current_segment
                desc_segments.append(current_segment)
                segments_count += 1
                # 生成对应的type
                for i in current_segment:
                    if i == self.padding:
                        current_type.append(0)
                    else:
                        current_type.append(1)
                type_segments.append(current_type)
            else:
                break
        output['desc_segments'] = desc_segments
        output['type_segments'] = type_segments
        output['token_ids_labels'] = label_id
        instance = {k: torch.tensor(v, dtype=torch.long) for k, v in output.items()}
        return instance


class TransformerXLTestSet(Dataset):
    def __init__(self, eval_path):
        # self.tokenizer = Tokenizer(VocabPath)
        self.eval_path = eval_path
        self.descriptions = []
        self.labels = []
        self.segment_length = SentenceLength - 1

        with open(Assistant, 'r', encoding='utf-8') as f:
            line = f.read().strip().split(',')
            self.cls = int(line[0])
            self.padding = self.cls + 1

        with open(eval_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line:
                    line = line.strip()
                    label, desc = line.split(',')
                    desc = [int(x) for x in desc.split(' ')]
                    label = int(label)
                    self.descriptions.append(desc)
                    self.labels.append(label)

    # def __gen_input_token(self, texts):
    #     input_token_ids = []
    #     input_token_ids.append(101)
    #     for token in texts:
    #         input_token_ids.append(self.tokenizer.token_to_id(token))
    #     input_token_ids.append(102)
    #     if len(input_token_ids) < SentenceLength:
    #         for i in range(SentenceLength - len(input_token_ids)):
    #             input_token_ids.append(0)
    #     return input_token_ids

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, item):
        output = {}
        descriptions_ids = self.descriptions[item]
        label_id = self.labels[item]
        # descriptions_ids = self.__gen_input_token(descriptions_ids)
        desc_segments = []
        type_segments = []
        segments_count = 0
        while True:
            if segments_count * self.segment_length < len(descriptions_ids):
                current_type = []
                current_segment = descriptions_ids[segments_count*self.segment_length:
                                                    min((segments_count+1) * self.segment_length,
                                                        len(descriptions_ids))]
                # 补全padding
                if len(current_segment) < self.segment_length:
                    for i in range(0, self.segment_length - len(current_segment)):
                        current_segment.append(self.padding)
                current_segment = [self.cls] + current_segment
                desc_segments.append(current_segment)
                segments_count += 1
                # 生成对应的type
                for i in current_segment:
                    if i == self.padding:
                        current_type.append(0)
                    else:
                        current_type.append(1)
                type_segments.append(current_type)
            else:
                break
        output['desc_segments'] = desc_segments
        output['type_segments'] = type_segments
        output['token_ids_labels'] = label_id
        instance = {k: torch.tensor(v, dtype=torch.long) for k, v in output.items()}
        return instance


if __name__ == '__main__':
    # dataloader = TransformerXLDataSet(CorpusPath)
    dataloader = TransformerXLTestSet(EvalPath)
    for data in dataloader:
        x = 1
        break
    print('加载完成')
