import torch
import torch.nn as nn


class Luong_Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_layer = config.n_layer

        self.linear_in = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.SELU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        self.linear_out = nn.Sequential(
            nn.Linear(config.hidden_size*2, config.hidden_size),
            nn.SELU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, output, encoder_out):
        """
        :param output: (batch, 1, hidden_size) decoder output
        :param encoder_out: (batch, t_len, hidden_size) encoder hidden state
        :return: attn_weight (batch, 1, time_step)
                  output (batch, 1, hidden_size) attention vector
        """
        out = self.linear_in(output) # (batch, 1, hidden_size)
        out = out.transpose(1, 2) # (batch, hidden_size, 1)
        attn_weights = torch.bmm(encoder_out, out) # (batch, t_len, 1)
        attn_weights = self.softmax(attn_weights.transpose(1, 2)) # (batch, 1, t_len)

        context = torch.bmm(attn_weights, encoder_out) # (batch, 1, hidden_size)
        output = self.linear_out(torch.cat((output, context), dim=2))

        return attn_weights, output


class Mulit_head(nn.Module):
    def __init__(self, config, attention):
        super().__init__()
        self.n_layer = config.n_layer
        self.hidden_size = config.hidden_size
        self.attention = attention

        self.linear_out = nn.Linear(self.hidden_size*self.n_layer, self.hidden_size)

    def forward(self, output, encoder_out):
        """
        :param output: (batch, n_layer, hidden_size) decoder output
        :param encoder_out: (n_layer, batch, t_len, hidden_size) encoder hidden state
        :return: output (batch, 1, hidden_size) attention vector
        """
        # context
        context = None
        for i in range(self.n_layer):
            _, c = self.attention(output[:, i, :].unsqueeze(1), encoder_out[i])
            if i == 0:
                context = c
            else:
                context = torch.cat((context, c), dim=-1)
        context = self.linear_out(context)
        return None, context


# class Mulit_head(nn.Module):
#     def __init__(self, config, attention):
#         super().__init__()
#         self.hidden_size = config.hidden_size
#         self.attention = attention
#
#         self.hidden_1 = nn.Linear(self.hidden_size, self.hidden_size)
#         self.hidden_2 = nn.Linear(self.hidden_size, self.hidden_size)
#         self.encoder_1 = nn.Linear(self.hidden_size, self.hidden_size)
#         self.encoder_2 = nn.Linear(self.hidden_size, self.hidden_size)
#         self.linear_out = nn.Linear(self.hidden_size*2, self.hidden_size)
#
#     def forward(self, output, encoder_out):
#         """
#         :param output: (batch, 1, hidden_size) decoder output
#         :param encoder_out: (batch, t_len, hidden_size) encoder hidden state
#         :return: output (batch, 1, hidden_size) attention vector
#         """
#         # context 1
#         output1 = self.hidden_1(output)
#         encoder_out1 = self.encoder_1(encoder_out)
#         attn_weights, context1 = self.attention(output1, encoder_out1)
#
#         # context 2
#         output2 = self.hidden_1(output)
#         encoder_out2 = self.hidden_2(output)
#         attn_weights, context2 = self.attention(output2, encoder_out2)
#
#         # concat
#         context = self.linear_out(torch.cat((context1, context2), dim=-1))
#         return attn_weights, context


class Bahdanau_Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.t_len = config.t_len
        self.linear_add = nn.Sequential(
            nn.Linear(config.hidden_size*2, config.hidden_size),
            nn.ReLU()
        )
        self.attn = nn.Linear(config.hidden_size, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.linear_out = nn.Linear(config.hidden_size+config.embedding_dim, config.hidden_size)

    def forward(self, x, output, encoder_out):
        """
        :param x:(batch, 1, embedding_dim)
        :param output:(n_layer, batch, hidden_size) decoder hidden state
        :param encoder_out:(batch, time_step, hidden_size) encoder hidden state
        :return: attn_weight (batch, 1, time_step)
                  context (batch, 1, hidden_size) attention vector
        """
        h = output[-1].view(-1, 1, self.hidden_size).repeat(1, self.t_len, 1) # (batch, t_len, hidden_size)
        vector = torch.cat((h, encoder_out), dim=2) # (batch, t_len, hidden_size*2)
        vector = self.linear_add(vector) # (batch, t_len, hidden_size)

        attn_weights = self.attn(vector).squeeze(2) # (batch, t_len)
        attn_weights = self.softmax(attn_weights).unsqueeze(1) # (batch, 1, t_len)

        context = torch.bmm(attn_weights, encoder_out) # (batch, 1, hidden_size)
        context = self.linear_out(torch.cat((context, x), dim=2))

        return attn_weights, context
