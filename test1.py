import torch

a = torch.ones([128, 1, 12, 64], dtype=torch.float)
print(a.size())
a = a.view([1, 128, 768])
print(a.size())


# b = torch.tensor([[[2, 3, 4], [1, 1, 1], [4, 5, 6]]], dtype=torch.float)
# # b = b.unsqueeze(1)
# print(b.size())
#
# print(a - b)
#
# import numpy
# a = numpy.zeros()