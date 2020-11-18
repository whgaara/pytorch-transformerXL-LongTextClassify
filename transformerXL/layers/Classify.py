import torch.nn as nn


class Classify(nn.Module):
    def __init__(self, hidden_size, classify):
        super(Classify, self).__init__()
        self.mlm_dense = nn.Linear(hidden_size, classify)

    def forward(self, feedforward_x):
        classify_info = self.mlm_dense(feedforward_x[:, 0:1, :].squeeze(1))
        return classify_info
