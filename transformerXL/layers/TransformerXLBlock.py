import torch.nn as nn

from transformerXL.layers.FeedForward import FeedForward
from transformerXL.layers.RelPosMultiHeadSelfAttention import RelPosMultiHeadSelfAttention


class TransformerXLBlock(nn.Module):
    def __init__(self,
                 hidden_size,
                 attention_head_num,
                 attention_head_size,
                 intermediate_size,
                 dropout_prob=0.1
                 ):
        super(TransformerXLBlock, self).__init__()
        self.rel_pos_multi_attention = RelPosMultiHeadSelfAttention(
            attention_head_num=attention_head_num,
            attention_head_size=attention_head_size)
        self.attention_layernorm = nn.LayerNorm(hidden_size)
        self.feedforward = FeedForward(
                hidden_size,
                intermediate_size,
                dropout_prob)
        self.feedforward_layernorm = nn.LayerNorm(hidden_size)

    def forward(self, x, rel_pos_emb, attention_mask, memories, layer_num):
        attention_x = self.rel_pos_multi_attention(x, rel_pos_emb, attention_mask, memories, layer_num)
        attention_x = x + attention_x
        attention_x = self.attention_layernorm(attention_x)

        feedforward_x = self.feedforward(attention_x)
        feedforward_x = attention_x + feedforward_x
        feedforward_x = self.feedforward_layernorm(feedforward_x)

        return feedforward_x
