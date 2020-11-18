import torch

inv_freq = 1 / (10000 ** (torch.arange(0.0, 768, 2.0) / 768))
