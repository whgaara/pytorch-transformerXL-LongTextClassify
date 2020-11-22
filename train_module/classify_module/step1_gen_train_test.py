import random

from pretrain_config import *


class PretrainProcess(object):
    def __init__(self):
        self.src_lines = open(SourcePath, 'r', encoding='utf-8').readlines()
        self.train_data_set = {}
        self.classes_count_list = []
        self.max_char_num = 0
        self.max_label_num = 0
        self.max_sentence_length = 0

    def traverse(self):
        f = open(Assistant, 'w', encoding='utf-8')
        for i, line in enumerate(self.src_lines):
            if i == 0:
                continue
            if line:
                label, desc = line.split('\t')
                label = int(label)
                desc_list = desc.split(' ')
                desc_list = [int(x) for x in desc_list]

                if self.max_char_num < max(desc_list):
                    self.max_char_num = max(desc_list)
                if self.max_label_num < label:
                    self.max_label_num = label
                if self.max_sentence_length < len(desc_list):
                    self.max_sentence_length = len(desc_list)

                if label in self.train_data_set:
                    self.train_data_set[label].append(desc_list)
                else:
                    self.train_data_set[label] = []
                    self.train_data_set[label].append(desc_list)

        # 补充了cls和padding两个字符
        f.write(str(self.max_char_num + 2) + ',' +
                str(self.max_label_num + 1) + ',' +
                str(self.max_sentence_length) + '\n')
        f.close()

    def balance(self):
        for label in self.train_data_set:
            self.classes_count_list.append(len(self.train_data_set[label]))
        self.classes_count_list.sort()
        mean_pos = len(self.classes_count_list) // 2
        mean_value = self.classes_count_list[mean_pos]

        f1 = open(CorpusPath, 'w', encoding='utf-8')
        f2 = open(EvalPath, 'w', encoding='utf-8')
        for label, descs in self.train_data_set.items():
            if len(descs) < mean_value:
                random.shuffle(descs)
                cut_point = int(len(descs) * 0.9)
                train = descs[:cut_point]
                eval = descs[cut_point:]
                for desc in train:
                    desc = [str(x) for x in desc]
                    f1.write(str(label) + ',' + ' '.join(desc) + '\n')
                for desc in eval:
                    desc = [str(x) for x in desc]
                    f2.write(str(label) + ',' + ' '.join(desc) + '\n')
            else:
                random.shuffle(descs)
                train = descs[:mean_value]
                eval = descs[mean_value:]
                for desc in train:
                    desc = [str(x) for x in desc]
                    f1.write(str(label) + ',' + ' '.join(desc) + '\n')
                for desc in eval:
                    rnd = random.random()
                    if rnd < 0.1:
                        desc = [str(x) for x in desc]
                        f2.write(str(label) + ',' + ' '.join(desc) + '\n')


if __name__ == '__main__':
    pp = PretrainProcess()
    print(get_time())
    pp.traverse()
    print(get_time())
    pp.balance()
    print(get_time())
