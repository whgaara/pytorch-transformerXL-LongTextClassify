import torch

a = torch.ones([2, 2, 3, 3], dtype=torch.float)
print(a.size())


b = torch.tensor([[[2, 3, 4], [1, 1, 1], [4, 5, 6]]], dtype=torch.float)
# b = b.unsqueeze(1)
print(b.size())

print(a - b)

import numpy
a = numpy.zeros()