import torch
import torch.nn as nn

from pretrain_config import device


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()
        self.demb = demb
        self.inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))

    def forward(self, seq_len):
        pos_seq = torch.arange(0.0, seq_len).to(device)
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq).to(device)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1).to(device)
        return pos_emb


class TokenEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout_prob=0.1):
        super(TokenEmbeddings, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.emb_normalization = nn.LayerNorm(hidden_size)
        self.emb_dropout = nn.Dropout(p=dropout_prob)

    def forward(self, input_token):
        token_embeddings = self.token_embeddings(input_token)
        embedding_x = self.emb_normalization(token_embeddings)
        embedding_x = self.emb_dropout(embedding_x)
        return embedding_x
