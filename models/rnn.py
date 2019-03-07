import torch
import torch.nn as nn
from models import *


class Embeds(nn.Module):
    def __init__(self, config, embedding=None):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim
        if embedding:
            self.embeds = nn.Embedding.from_pretrained(embedding)
        else:
            self.embeds = nn.Embedding(self.vocab_size, self.embedding_dim)

    def forward(self, x):
        """
        :param x: (batch, t_len)
        :return: (batch, t_len, embedding_dim
        """
        return self.embeds(x)


class Encoder(nn.Module):
    def __init__(self, embeds, config):
        super().__init__()
        self.embeds = embeds
        self.n_layer = config.n_layer
        self.cell = config.cell
        self.bidirectional = config.bidirectional
        self.hidden_size = config.hidden_size

        if config.cell == 'lstm':
            self.rnn = nn.LSTM(
                input_size=config.embedding_dim,
                hidden_size=config.hidden_size,
                num_layers=config.n_layer,
                batch_first=True,
                dropout=config.dropout,
                bidirectional=config.bidirectional
            )

        else:
            self.rnn = nn.GRU(
                input_size=config.embedding_dim,
                hidden_size=config.hidden_size,
                num_layers=config.n_layer,
                batch_first=True,
                dropout=config.dropout,
                bidirectional=config.bidirectional
            )

    def forward(self, x):
        """
        :param x:(batch, t_len)
        :return: gru_h(n_layer, batch, hidden_size) lstm_h(h, c)
                  out(batch, t_len, hidden_size)
        """
        e = self.embeds(x)
        # out (batch, time_step, hidden_size*bidirection)
        # h (batch, n_layers*bidirection, hidden_size)
        encoder_out, h = self.rnn(e)

        if self.bidirectional:
            encoder_out = encoder_out[:, :, :self.hidden_size] + encoder_out[:, :, self.hidden_size:]
        if self.cell == 'lstm':
            h = (h[0][::2].contiguous(), h[1][::2].contiguous())
        else:
            h = h[:self.n_layer]
        return h, encoder_out


class Decoder(nn.Module):
    def __init__(self, embeds, config):
        super().__init__()
        self.embeds = embeds
        self.attn_flag = config.attn_flag
        self.cell = config.cell

        if config.cell == 'lstm':
            self.rnn = nn.LSTM(
                input_size=config.embedding_dim,
                hidden_size=config.hidden_size,
                num_layers=config.n_layer,
                batch_first=True,
                dropout=config.dropout,
            )
        else:
            self.rnn = nn.GRU(
                input_size=config.embedding_dim,
                hidden_size=config.hidden_size,
                num_layers=config.n_layer,
                batch_first=True,
                dropout=config.dropout,
            )

        if config.attn_flag == 'bahdanau':
            self.attention = Bahdanau_Attention(config)
        elif config.attn_flag == 'luong':
            self.attention = Luong_Attention(config)
        else:
            self.attention = None

    def forward(self, x, h, encoder_output):
        """
        :param x: (batch, 1) decoder input
        :param h: (batch, n_layer, hidden_size)
        :param encoder_output: (batch, t_len, hidden_size) encoder hidden state
        :return: attn_weight (batch, 1, time_step)
                  out (batch, 1, hidden_size) decoder output
                  h (batch, n_layer, hidden_size) decoder hidden state
        """
        attn_weights = None
        e = self.embeds(x).unsqueeze(1) # (batch, 1, embedding_dim)
        if self.attn_flag == 'bahdanau':
            if self.cell == 'lstm':
                attn_weights, e = self.attention(e, h[0], encoder_output)
            else:
                attn_weights, e = self.attention(e, h, encoder_output)
        out, h = self.rnn(e, h)

        if self.attn_flag == 'luong':
            attn_weights, out = self.attention(out, encoder_output)

        return attn_weights, out, h