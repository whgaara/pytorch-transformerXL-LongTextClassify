import math
import numpy
import torch.nn as nn

from pretrain_config import *


# 相对位置多头自注意力
class RelPosMultiHeadSelfAttention(nn.Module):
    def __init__(self,
                 attention_head_num,
                 attention_head_size,
                 dropout_prob=0.1
                 ):
        super(RelPosMultiHeadSelfAttention, self).__init__()
        self.attention_head_num = attention_head_num
        self.attention_head_size = attention_head_size
        self.out_dim = attention_head_num * attention_head_size

        # 申明网络
        self.W_q = nn.Linear(self.out_dim, self.out_dim, bias=False)
        self.W_ke = nn.Linear(self.out_dim, self.out_dim, bias=False)
        self.W_v = nn.Linear(self.out_dim, self.out_dim, bias=False)
        self.W_kr = nn.Linear(self.out_dim, self.out_dim, bias=False)
        self.u = nn.Parameter(torch.randn(self.attention_head_num, self.attention_head_size))
        self.v = nn.Parameter(torch.randn(self.attention_head_num, self.attention_head_size))

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout_prob)
        self.o_dense = nn.Linear(self.out_dim, self.out_dim)

    def update_memories_by_layer(self, memories, last_hidden, layer_num, segment_num):
        with torch.no_grad():
            for bz in range(len(memories[layer_num])):
                memories[layer_num][bz][segment_num*SentenceLength+MemoryLength:(segment_num+1)*SentenceLength+MemoryLength] =\
                    last_hidden[bz][0:SentenceLength]
            return memories

    def rel_shift(self, bd):
        qlen, klen, bsz, n_head = bd.size()
        for qi in range(qlen):
            bd[qi] = torch.cat((bd[qi][qlen-1-qi:], torch.zeros([qlen-1-qi, bsz, n_head])), dim=0)
        return bd

    def forward(self, x, rel_pos_emb, attention_mask, memories, layer_num, segment_num):
        # 初始化qx，kx，vx
        layer_count, b, total_len, m_hidden_size = memories.size()
        assert total_len >= MemoryLength
        qx = x
        sg_m = memories[layer_num][:, segment_num*MemoryLength:(segment_num+1)*MemoryLength, :].to(device)
        kx = torch.cat((sg_m, x), dim=1).to(device)
        vx = torch.cat((sg_m, x), dim=1).to(device)

        # 计算q，k，v
        qx = self.W_q(qx)
        kx = self.W_ke(kx)
        vx = self.W_v(vx)
        kr = self.W_kr(rel_pos_emb)
        k_length = kx.size()[1]
        r_length = kr.size()[0]

        # 接下来对qx，kx，vx进行reshape，此时qx的维度是（qlen，768），kx是（qlen+mlen，768），vx也是（qlen+mlen，768）
        qx = qx.view(SentenceLength, BatchSize, self.attention_head_num, self.attention_head_size)
        kx = kx.view(k_length, BatchSize, self.attention_head_num, self.attention_head_size)
        vx = vx.view(k_length, BatchSize, self.attention_head_num, self.attention_head_size)
        kr = kr.view(r_length, self.attention_head_num, self.attention_head_size)

        # 接下来进行多头a、c矩阵的计算，结果的shape：qlen x klen x bsz x n_head
        qx_u = qx + self.u
        ac = torch.einsum('ibnd,jbnd->ijbn', [qx_u, kx])

        # 接下来进行多头b、d矩阵的计算，结果的shape：qlen x klen x bsz x n_head
        qx_v = qx + self.v
        bd = torch.einsum('ibnd,jnd->ijbn', [qx_v, kr])

        # 接下来，因为BD部分是qx和v乘以相对位置，原理同bert的qi*kj
        # 但是这里因为是相对位置，所以例如q0和k0相乘时，其相对位置是mlen，且q0最长的相对位置只有mlen
        # 而q511的最长相对位置有51bd1+mlen，最终效果相当于所有的q与所有的相对位置（m+511）相乘，但q要按照位置进行平移。
        bd = self.rel_shift(bd)

        # shape：qlen x klen x bsz x n_head
        rel_pos_attention = ac + bd
        # 因为q、k相乘，结果变大，因此对结果除以根号
        rel_pos_attention = rel_pos_attention / math.sqrt(float(self.attention_head_size))

        # 防止padding补全的0经过softmax后影响结果，对每个0值都加一个很大的负数，这样softmax后也会约等于0
        # attention_mask的shape为：[batch_size, qlen, klen]
        # rel_pos_attention = rel_pos_attention.view(BatchSize,
        #                                            self.attention_head_num,
        #                                            SentenceLength,
        #                                            k_length)
        # m_supplement = torch.ones([SentenceLength, MemoryLength], dtype=torch.long).to(device)
        # m_supplement = m_supplement.expand([BatchSize, SentenceLength, MemoryLength]).to(device)
        # attention_mask = torch.cat((m_supplement, attention_mask), dim=-1).to(device)
        # add_mask = (1.0 - attention_mask.float()) * 1e9
        # rel_pos_attention -= add_mask

        rel_pos_attention = self.softmax(rel_pos_attention)
        rel_pos_attention = torch.einsum('ijbn,jbnd->ibnd', (rel_pos_attention, vx))
        # 因为view()需要Tensor中的元素地址是连续的，但可能出现Tensor不连续的情况，所以先用 .contiguous() 将其在内存中变成连续分布：
        rel_pos_attention = rel_pos_attention.contiguous().view(BatchSize, SentenceLength, self.out_dim)
        rel_pos_attention = self.dropout(rel_pos_attention)
        rel_pos_attention = self.o_dense(rel_pos_attention)

        memories = self.update_memories_by_layer(memories, rel_pos_attention, layer_num, segment_num)

        return rel_pos_attention, memories
