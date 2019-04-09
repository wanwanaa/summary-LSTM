import torch
import torch.nn as nn
import numpy as np
from models.beam import *


class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder, config):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.bos = config.bos
        self.s_len = config.s_len
        self.beam_size = config.beam_size
        self.config = config

        self.loss_func = nn.CrossEntropyLoss()

        self.linear_out = nn.Linear(config.hidden_size, config.tgt_vocab_size)
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

    # def compute_loss(self, result, y):
    #     result = result.contiguous().view(-1, self.config.tgt_vocab_size)
    #     y = y.contiguous().view(-1)
    #     loss = self.loss_func(result, y)
    #     return loss

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
        if self.config.intra_decoder:
            if torch.cuda.is_available():
                outs = torch.zeros(x.size(0), 1, self.config.hidden_size).type(torch.cuda.FloatTensor)
            else:
                outs = torch.zeros(x.size(0), 1, self.config.hidden_size)
        else:
            outs = None
        for i in range(self.s_len):
            _, out, h = self.decoder(y_c[:, i], h, encoder_out, outs)
            if self.config.intra_decoder:
                if i == 0:
                    outs = h[0].transpose(0, 1)[:, 1, :].unsqueeze(1)
                else:
                    outs = torch.cat((outs, h[0].transpose(0, 1)[:, 1, :].unsqueeze(1)), dim=1)
            gen = self.output_layer(out).squeeze()
            result.append(gen)

        outputs = torch.stack(result).transpose(0, 1)
        # print('result:', outputs.size())
        # print('result:', y.size())
        # loss = self.compute_loss(outputs, y)
        return outputs

    def sample(self, x, y):
        h, encoder_out = self.encoder(x)
        out = torch.ones(x.size(0)) * self.bos
        result = []
        idx = []
        if self.config.intra_decoder:
            if torch.cuda.is_available():
                outs = torch.zeros(x.size(0), 1, self.config.hidden_size).type(torch.cuda.FloatTensor)
            else:
                outs = torch.zeros(x.size(0), 1, self.config.hidden_size)
        else:
            outs = None
        for i in range(self.s_len):
            if torch.cuda.is_available():
                out = out.type(torch.cuda.LongTensor)
            else:
                out = out.type(torch.LongTensor)
            _, out, h = self.decoder(out, h, encoder_out, outs)
            if self.config.intra_decoder:
                if i == 0:
                    outs = h[0].transpose(0, 1)[:, 1, :].unsqueeze(1)
                else:
                    outs = torch.cat((outs, h[0].transpose(0, 1)[:, 1, :].unsqueeze(1)), dim=1)
            gen = self.linear_out(out.squeeze(1))
            result.append(gen)
            gen = self.softmax(gen)
            out = torch.argmax(gen, dim=1)
            idx.append(out.cpu().numpy())
        result = torch.stack(result).transpose(0, 1)
        idx = np.transpose(np.array(idx))
        # loss = self.compute_loss(result, y)
        return result, idx

    def beam_search(self, x):
        h, encoder_out = self.encoder(x)
        encoder_out = encoder_out.repeat(1, self.beam_size, 1).view(-1, self.config.t_len, self.config.hidden_size)
        # initial beam
        beam = []
        for i in range(x.size(0)):
            if self.config.cell == 'lstm':
                beam.append(Beam(self.config, (h[0][:, i].squeeze(), h[1][:, i].squeeze())))
            else:
                beam.append(Beam(self.config, h[:, i].squeeze()))

        if self.config.intra_decoder:
            outs = torch.zeros(x.size(0)*self.beam_size, 1, self.config.hidden_size)
        else:
            outs = None

        for i in range(self.s_len):
            out = []
            h = []
            for i in range(x.size(0)):
                out.append(beam[i].get_node())
                h.append(beam[i].get_h())
            out = torch.stack(out).view(-1) # (batch_size, beam_size, 1) -> (batch_size*beam_size)
            # (batch_size, beam_size, n_layer, hidden_size) -> (batch_size*beam_size, n_layer, hidden_size)
            # ->(n_layer, batch_size, hidden_size)
            if self.config.cell == 'lstm':
                h0 = []
                h1 = []
                for i in range(len(h)):
                    h0.append(h[i][0])
                    h1.append(h[i][1])
                h0 = torch.stack(h0).view(-1, self.config.n_layer, self.config.hidden_size).transpose(0, 1)
                h1 = torch.stack(h1).view(-1, self.config.n_layer, self.config.hidden_size).transpose(0, 1)
                h = (h0, h1)
            else:
                h = torch.stack(h).view(-1, self.config.n_layer, self.hidden_size).transpose(0, 1)
            if torch.cuda.is_available():
                out = out.type(torch.cuda.LongTensor)
            else:
                out = out.type(torch.LongTensor)

            # out (batch_size*beam_size, 1, vocab_size)
            # h (n_layer, batch_size*beam_size, hidden_size)
            _, out, h = self.decoder(out, h, encoder_out, outs)

            if self.config.intra_decoder:
                if i == 0:
                    outs = h[0].transpose(0, 1)[:, 1, :].unsqueeze(1)
                else:
                    outs = torch.cat((outs, h[0].transpose(0, 1)[:, 1, :].unsqueeze(1)), dim=1)

            out = self.linear_out(out)
            out = self.softmax(out)

            # out (batch_size, beam_size, vocab_size)
            # h (n_layer, batch_size, beam_size, hidden_size)
            out = out.view(-1, self.beam_size, self.config.vocab_size)
            if self.config.cell == 'lstm':
                h0 = h[0].view(self.config.n_layer, -1, self.beam_size, self.config.hidden_size)
                h1 = h[1].view(self.config.n_layer, -1, self.beam_size, self.config.hidden_size)
                h = (h0, h1)
                for i in range(x.size(0)):
                    beam[i].advance((h[0][:, i], h[1][:, i]), out[i])
            else:
                h = h.view(self.config.n_layer, -1, self.beam_size, self.config.hidden_size)
                for i in range(x.size(0)):
                    beam[i].advance(h[:, i], out[i])
        idx = []
        for i in range(x.size(0)):
            # print(beam[i].path)
            idx.append(np.array(beam[i].path[0][0]))
        return idx