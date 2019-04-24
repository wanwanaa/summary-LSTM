import torch
import torch.nn as nn
from models import *


class Encoder_cnn(nn.Module):
    def __init__(self, embeds, config):
        super().__init__()
        self.embeds = embeds
        self.embedding_dim = config.embedding_dim
        self.hidden_size = config.hidden_size
        self.n_layer = config.n_layer
        self.t_len = config.t_len

        # nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # convolution path1
        self.conv1 = nn.Sequential(
            nn.Conv1d(self.hidden_size, self.hidden_size, 1, 1, 0),
            nn.BatchNorm1d(config.hidden_size),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(self.hidden_size, self.hidden_size, 3, 1, 1),
            nn.BatchNorm1d(config.hidden_size),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(self.hidden_size, self.hidden_size, 3, 1, 1),
            nn.BatchNorm1d(config.hidden_size),
            nn.ReLU()
        )

        # GLU
        self.input = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size*2),
            nn.GLU()
        )
        # Linear
        self.linear_out = nn.Sequential(
            nn.Linear(self.hidden_size*self.t_len, self.hidden_size*4),
            nn.Linear(self.hidden_size*4, self.hidden_size)
        )

    def forward(self, x):
        # e(batch, t_len, hidden_size)
        e = self.embeds(x)

        e = self.input(e).transpose(1, 2)

        # (batch, hidden_size, t_len)
        out = self.conv1(e)
        out = self.conv2(out)
        out = self.conv3(out)

        out = out.view(x.size(0), -1)
        out = self.linear_out(out).view(1, -1, self.hidden_size)
        out = out.repeat(self.n_layer, 1, 1)

        return out