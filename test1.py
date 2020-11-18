import torch

all_ones = torch.ones([3, 4])

dec_attn_mask = (torch.triu(all_ones, 12) + torch.tril(all_ones, -5)).byte()[:, :, None]
print(dec_attn_mask)