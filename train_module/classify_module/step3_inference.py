# coding: utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch

from tqdm import tqdm
from pretrain_config import PretrainPath, device, SentenceLength, VocabPath
from transformerXL.common.tokenizers import Tokenizer


class Inference(object):
    def __init__(self):
        self.sen_count = 0
        self.sen_acc = 0
        self.tokenizer = Tokenizer(VocabPath)
        self.model = torch.load(PretrainPath).to(device).eval()
        print('加载模型完成！')

    def get_id_from_text(self, text):
        assert isinstance(text, str)
        ids = []
        inputs = []
        segments = []
        for token in text:
            ids.append(self.tokenizer.token_to_id(token))
        inputs.append(101)
        segments.append(1)

        for id in ids:
            if len(inputs) < SentenceLength - 1:
                if isinstance(id, list):
                    for x in id:
                        inputs.append(x)
                        segments.append(1)
                else:
                    inputs.append(id)
                    segments.append(1)
            else:
                inputs.append(102)
                segments.append(1)
                break

        if len(inputs) != len(segments):
            print('len error!')
            return None

        if len(inputs) < SentenceLength - 1:
            inputs.append(102)
            segments.append(1)
            for i in range(SentenceLength - len(inputs)):
                inputs.append(0)
                segments.append(0)

        inputs = torch.tensor(inputs).unsqueeze(0).to(device)
        segments = torch.tensor(segments).unsqueeze(0).to(device)
        return inputs, segments

    def inference_single(self, text):
        text = text[:510]
        text2id, segments = self.get_id_from_text(text)
        with torch.no_grad():
            output_tensor = self.model(text2id, segments)
            output_tensor = torch.nn.Softmax(dim=-1)(output_tensor)
            output_topk_prob = torch.topk(output_tensor, 1).values.squeeze(0).tolist()
            output_topk_indice = torch.topk(output_tensor, 1).indices.squeeze(0).tolist()
        return output_topk_indice[0], output_topk_prob[0]

    def inference_batch(self, file_path):
        f = open(file_path, 'r', encoding='utf-8')
        for line in tqdm(f):
            if line:
                line = line.strip().replace(' ', '')
                self.sen_count += 1
                line = line.split(',')
                src = line[0]
                label = int(line[1])
                output, probs = self.inference_single(src)
                if label == output:
                    self.sen_acc += 1
        print('判断正确个数：%s，句子总共个数：%s，判断正确率：%s' %
              (self.sen_acc, self.sen_count, round(float(self.sen_acc) / float(self.sen_count), 2)))


if __name__ == '__main__':
    bert_infer = Inference()
    bert_infer.inference_batch('../../data/test_data/test.txt')
