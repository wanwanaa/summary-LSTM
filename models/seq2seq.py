import torch
import torch.nn as nn
import numpy as np
from models import *


class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder, config):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.bos = config.bos
        self.s_len = config.s_len
        self.loss_func = nn.CrossEntropyLoss()
        self.linear_out = nn.Linear(config.hidden_size, config.vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    # add <bos> to sentence
    def convert(self, x):
        """
        :param x:(batch, s_len) (word_1, word_2, ... , word_n)
        :return:(batch, s_len) (<bos>, word_1, ... , word_n-1)
        """
        if torch.cuda.is_available():
            start = (torch.ones(x.size(0), 1) * self.bos).type(torch.cuda.LongTensor)
        else:
            start = (torch.ones(x.size(0), 1) * self.bos).type(torch.LongTensor)
        x = torch.cat((start, x), dim=1)
        return x[:, :-1]

    def output_layer(self, x):
        """
        :param x: (batch, hidden_size) decoder output
        :return: (batch, vocab_size)
        """
        return self.linear_out(x)

    def compute_loss(self, result, y):
        result = result.contiguous().view(-1, 4000)
        y = y.contiguous().view(-1)
        loss = self.loss_func(result, y)
        return loss

    def forward(self, x, y):
        """
        :param x: (batch, t_len) encoder input
        :param y: (batch, s_len) decoder input
        :return:
        """
        h, encoder_out = self.encoder(x)

        # add <bos>
        y_c = self.convert(y)

        # decoder
        result = []
        for i in range(self.s_len):
            _, out, h = self.decoder(y_c[:, i], h, encoder_out)
            gen = self.output_layer(out).squeeze()
            result.append(gen)

        outputs = torch.stack(result).transpose(0, 1)
        loss = self.compute_loss(outputs, y)
        return loss, outputs

    def sample(self, x, y):
        h, encoder_out = self.encoder(x)
        out = torch.ones(x.size(0)) * self.bos
        result = []
        idx = []
        for i in range(self.s_len):
            if torch.cuda.is_available():
                out = out.type(torch.cuda.LongTensor)
            else:
                out = out.type(torch.LongTensor)
            _, out, h = self.decoder(out, h, encoder_out)
            gen = self.linear_out(out.squeeze(1))
            result.append(gen)
            gen = self.softmax(gen)
            out = torch.argmax(gen, dim=1)
            idx.append(out.cpu().numpy())
        result = torch.stack(result).transpose(0, 1)
        idx = np.transpose(np.array(idx))
        loss = self.compute_loss(result, y)
        return loss, idx

    def beam_search(self, x):
        pass
