import torch.nn as nn
import torch
import math

class MLP(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        num_hidden = cfg.hidden
        num_output = cfg.label_num
        num_input = cfg.num_features

        self.fc_input = nn.Linear(num_input, num_hidden)

        self.norm = nn.LayerNorm(num_hidden)
        self.dropout = nn.Dropout(cfg.dropout)

        self.predict = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(num_hidden, num_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(num_hidden // 2, num_output)
        )

    def forward(self, x):
        x = self.fc_input(x)
        x = self.norm(x)
        x = self.dropout(x)

        output = self.predict(x)

        return output

