import time
import torch

cuda_condition = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda_condition else 'cpu')

# ## 模型文件路径 ## #
SourcePath = '../../data/src_data/src_set.csv'
CorpusPath = '../../data/train_data/train_set.csv'
# CorpusPath = '../../data/train_data/train_demo.csv'
EvalPath = '../../data/test_data/eval_set.csv'
TestPath = '../../data/test_data/test_a.csv'

# 保存最大句长，字符数，类别数
Assistant = '../../data/train_data/assistant.txt'

# ## 训练调试参数开始 ## #
Epochs = 16
BatchSize = 1
LearningRate = 1e-5
MemoryLength = 128
AttentionMask = False
HiddenLayerNum = 6
SentenceLength = 128
PretrainPath = '../../checkpoint/finetune/transformerXL_classify_%s.model' % SentenceLength
# ## 训练调试参数结束 ## #

# ## 通用参数 ## #
DropOut = 0.1
VocabSize = int(open(Assistant, 'r', encoding='utf-8').readline().split(',')[0])
HiddenSize = 768
IntermediateSize = 3072
AttentionHeadNum = 12


def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
