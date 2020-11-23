import torch.nn as nn

from pretrain_config import *
from transformerXL.layers.TransformerXLEmbeddings import TokenEmbedding, RelPositionEmbedding
from transformerXL.layers.TransformerXLBlock import TransformerXLBlock
from transformerXL.layers.Classify import Classify


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
        self.kinds_num = kinds_num
        self.vocab_size = vocab_size
        self.hidden_size = hidden
        self.sen_length = sen_length
        self.mem_length = mem_length
        self.num_hidden_layers = num_hidden_layers
        self.attention_head_num = attention_heads
        self.dropout_prob = dropout_prob
        self.attention_head_size = hidden // attention_heads
        self.intermediate_size = intermediate_size

        # 申明网络
        self.init_memories = nn.Parameter(torch.randn(MemoryLength, self.hidden_size)).expand(
            [HiddenLayerNum, BatchSize, MemoryLength, self.hidden_size])
        self.token_emd = TokenEmbedding(vocab_size=self.vocab_size, hidden_size=self.hidden_size)
        self.rel_post_emb = RelPositionEmbedding(self.hidden_size)
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

    def forward(self, desc_segments, type_segments):
        _, segments_count, _ = desc_segments.size()
        # 初始化memories
        memories = torch.cat((self.init_memories,
                              torch.zeros([HiddenLayerNum,
                                           BatchSize,
                                           segments_count*SentenceLength,
                                           self.hidden_size])), dim=2)

        # 这里需要遍历的是第二维度
        for segments_num in range(segments_count):
            # 抽取当前batch的当前segment
            input_token = desc_segments[:, segments_num, :]
            segment_ids = type_segments[:, segments_num, :]

            # embedding
            embedding_x = self.token_emd(input_token)
            rel_pos_emb = self.rel_post_emb(SentenceLength, MemoryLength)

            # transformer block
            transformerxl_block_x = None
            if AttentionMask:
                attention_mask = self.gen_attention_masks(segment_ids).to(device)
            else:
                attention_mask = None

            for layers_num in range(self.num_hidden_layers):
                if layers_num == 0:
                    transformerxl_block_x, new_memories = \
                        self.transformer_blocks[layers_num](embedding_x,
                                                            rel_pos_emb,
                                                            attention_mask,
                                                            memories,
                                                            layers_num,
                                                            segments_num)
                    memories = new_memories
                else:
                    transformerxl_block_x, new_memories = \
                        self.transformer_blocks[layers_num](transformerxl_block_x,
                                                            rel_pos_emb,
                                                            attention_mask,
                                                            memories,
                                                            layers_num,
                                                            segments_num)
                    memories = new_memories

        output = self.classify(transformerxl_block_x)
        return output
