import torch.nn as nn

from pretrain_config import *
from bert.layers.TransformerXLEmbeddings import TokenEmbedding, RelPositionalEmbedding
from bert.layers.TransformerXLBlock import TransformerXLBlock
from bert.layers.Classify import Classify


class TransformerXL(nn.Module):
    def __init__(self,
                 kinds_num,
                 vocab_size=VocabSize,
                 hidden=HiddenSize,
                 sen_length=SentenceLength,
                 mem_length=MemoryLength,
                 num_hidden_layers=HiddenLayerNum,
                 attention_heads=AttentionHeadNum,
                 dropout_prob=DropOut,
                 intermediate_size=IntermediateSize
                 ):
        super(TransformerXL, self).__init__()
        self.vocab_size = vocab_size
        self.kinds_num = kinds_num
        self.hidden_size = hidden
        self.sen_length = sen_length
        self.mem_length = mem_length
        self.num_hidden_layers = num_hidden_layers
        self.attention_head_num = attention_heads
        self.dropout_prob = dropout_prob
        self.attention_head_size = hidden // attention_heads
        self.intermediate_size = intermediate_size

        # 申明网络
        self.bert_emd = TokenEmbedding(vocab_size=self.vocab_size, hidden_size=self.hidden_size)
        self.transformer_blocks = nn.ModuleList(
            TransformerXLBlock(
                hidden_size=self.hidden_size,
                attention_head_num=self.attention_head_num,
                attention_head_size=self.attention_head_size,
                intermediate_size=self.intermediate_size).to(device)
            for _ in range(self.num_hidden_layers)
        )
        self.classify = Classify(self.hidden_size, self.kinds_num)

    @staticmethod
    def gen_attention_masks(segment_ids):
        def gen_attention_mask(segment_id):
            dim = segment_id.size()[-1]
            attention_mask = torch.zeros([dim, dim], dtype=torch.int64)
            end_point = 0
            for i, segment in enumerate(segment_id.tolist()):
                if segment:
                    end_point = i
                else:
                    break
            for i in range(end_point + 1):
                for j in range(end_point + 1):
                    attention_mask[i][j] = 1
            return attention_mask
        attention_masks = []
        segment_ids = segment_ids.tolist()
        for segment_id in segment_ids:
            attention_mask = gen_attention_mask(torch.tensor(segment_id))
            attention_masks.append(attention_mask.tolist())
        return torch.tensor(attention_masks)

    # def forward(self, input_token, segment_ids):
    #     # embedding
    #     embedding_x = self.bert_emd(input_token, segment_ids)
    #     attention_mask = self.gen_attention_masks(segment_ids).to(device)
    #     feedforward_x = None
    #     # transformer
    #     for i in range(self.num_hidden_layers):
    #         feedforward_x = self.transformer_blocks[i](embedding_x, attention_mask)
    #     # mlm
    #     output = self.classify(feedforward_x)
    #     return output
