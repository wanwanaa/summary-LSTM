import torch
import torch.nn as nn
import math


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


# # n_layer
# class Multi_head(nn.Module):
#     def __init__(self, config, attention):
#         super().__init__()
#         self.n_layer = config.n_layer
#         self.hidden_size = config.hidden_size
#         self.attention = attention
#
#         self.linear_out1 = nn.Sequential(
#             nn.Linear(self.hidden_size, self.hidden_size),
#             nn.SELU(),
#             nn.Linear(self.hidden_size, self.hidden_size)
#         )
#         self.linear_out2 = nn.Sequential(
#             nn.Linear(self.hidden_size, self.hidden_size),
#             nn.SELU(),
#             nn.Linear(self.hidden_size, self.hidden_size)
#         )
#
#         self.linear_enc1 = nn.Sequential(
#             nn.Linear(self.hidden_size, self.hidden_size),
#             nn.SELU(),
#             nn.Linear(self.hidden_size, self.hidden_size)
#         )
#         self.linear_enc2 = nn.Sequential(
#             nn.Linear(self.hidden_size, self.hidden_size),
#             nn.SELU(),
#             nn.Linear(self.hidden_size, self.hidden_size)
#         )
#
#         self.linear_out = nn.Sequential(
#             nn.Linear(self.hidden_size*self.n_layer, self.hidden_size),
#             nn.SELU(),
#             nn.Linear(self.hidden_size, self.hidden_size))
#
#     def forward(self, output, encoder_out):
#         """
#         :param output: (batch, n_layer, hidden_size) decoder output
#         :param encoder_out: (batch, t_len, n_layer, hidden_size) encoder hidden state
#         :return: output (batch, 1, hidden_size) attention vector
#         """
#         output1 = self.linear_out1(output[:, 0, :])
#         encoder_out1 = self.linear_enc1(encoder_out[:, :, 0, :])
#         _, context1 = self.attention(output1, encoder_out1)
#
#         output2 = self.linear_out1(output[:, 1, :])
#         encoder_out2 = self.linear_enc1(encoder_out[:, :, 1, :])
#         _, context2 = self.attention(output2, encoder_out2)
#
#         context = torch.cat((context1, context2), -1)
#         context = self.linear_out(context)
#         return None, context


# class Multi_head(nn.Module):
#     def __init__(self, config, attention):
#         super().__init__()
#         self.hidden_size = config.hidden_size
#         self.attention = attention
#
#         self.hidden_1 = nn.Linear(self.hidden_size, self.hidden_size)
#         self.hidden_2 = nn.Linear(self.hidden_size, self.hidden_size)
#         self.hidden_3 = nn.Linear(self.hidden_size, self.hidden_size)
#         self.hidden_4 = nn.Linear(self.hidden_size, self.hidden_size)
#
#         self.encoder_1 = nn.Linear(self.hidden_size, self.hidden_size)
#         self.encoder_2 = nn.Linear(self.hidden_size, self.hidden_size)
#         self.encoder_3 = nn.Linear(self.hidden_size, self.hidden_size)
#         self.encoder_4 = nn.Linear(self.hidden_size, self.hidden_size)
#
#         self.linear_out = nn.Sequential(
#             nn.Linear(self.hidden_size*4, self.hidden_size),
#             nn.SELU(),
#             nn.Linear(self.hidden_size, self.hidden_size)
#         )
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
#         output2 = self.hidden_2(output)
#         encoder_out2 = self.encoder_2(output)
#         attn_weights, context2 = self.attention(output2, encoder_out2)
#
#         # context 3
#         output3 = self.hidden_3(output)
#         encoder_out3 = self.encoder_3(output)
#         attn_weights, context3 = self.attention(output3, encoder_out3)
#
#         # context 4
#         output4 = self.hidden_4(output)
#         encoder_out4 = self.encoder_4(output)
#         attn_weights, context4 = self.attention(output4, encoder_out4)
#
#         # concat
#         context = self.linear_out(torch.cat((context1, context2, context3, context4), dim=-1))
#
#         return attn_weights, context

# transformer Multi-head attention
# Scaled Dot Product Attention
class Multi_head(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.h = config.h_head
        self.d_k = config.hidden_size // self.h

        self.linear_enc = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_out = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_v = nn.Linear(config.hidden_size, config.hidden_size)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, output, encoder_outputs):
        """
        :param output: (batch, 1, hidden_size) decoder output
        :param encoder_out: (batch, t_len, hidden_size) encoder hidden state
        """
        len_out = output.size(1)
        len_enc = encoder_outputs.size(1)
        # (batch, len, h, d_k)
        output = self.linear_out(output).contiguous().view(-1, len_out, self.h, self.d_k)
        value = self.linear_v(encoder_outputs).contiguous().view(-1, len_enc, self.h, self.d_k)
        encoder_outputs = self.linear_enc(encoder_outputs).contiguous().view(-1, len_enc, self.h, self.d_k)

        # (batch*h, len, d_k)
        output = output.transpose(1, 2).contiguous().view(-1, len_out, self.d_k)
        encoder_outputs = encoder_outputs.transpose(1, 2).contiguous().view(-1, len_enc, self.d_k)
        value = value.contiguous().view(-1, len_enc, self.d_k)

        attn = torch.bmm(encoder_outputs, output.transpose(1, 2)) # (batch, enc_len, out_len 1)
        attn = attn.transpose(1, 2) # (batch, 1, enc_len)
        attn = attn / math.sqrt(self.d_k)
        weights = self.softmax(attn)
        out = torch.bmm(attn, value)
        out = out.view(-1, self.h, 1, self.d_k)
        out = out.transpose(1, 2).contiguous().view(-1, 1, self.hidden_size)

        return weights, out


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


# class Self_attention(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.hidden_size = config.hidden_size
#         self.linear_enc = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
#                                         nn.SELU(),
#                                         nn.Linear(self.hidden_size, self.hidden_size))
#         self.linear_out = nn.Sequential(nn.Linear(self.hidden_size*2, self.hidden_size),
#                                         nn.SELU(),
#                                         nn.Linear(self.hidden_size, self.hidden_size))
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, encoder_out):
#         """
#         :param encoder_out: (batch, time_step, hidden_size) encoder hidden state
#         :return:context: batch, time_step, hidden_size)
#         """
#         enc = self.linear_enc(encoder_out) # (batch, time_step, hidden_size)
#         h = enc.transpose(1, 2) # (batch, hidden_size, time_step)
#         weights = torch.bmm(enc, h) # (batch, time_step, hidden_size)
#         weights = self.softmax(weights/math.sqrt(self.hidden_size))
#         context = torch.bmm(weights, enc) # (batch, time_step, hidden_size)
#         context = self.linear_out(torch.cat((enc, context), 2)) + encoder_out
#
#         return context


class Self_attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.linear_enc = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                        nn.SELU(),
                                        nn.Linear(self.hidden_size, self.hidden_size))
        self.linear_value = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                          nn.SELU(),
                                          nn.Linear(self.hidden_size, self.hidden_size))
        self.linear_out = nn.Sequential(nn.Linear(self.hidden_size*2, self.hidden_size),
                                        nn.SELU(),
                                        nn.Linear(self.hidden_size, self.hidden_size))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, encoder_out):
        """
        :param encoder_out: (batch, time_step, hidden_size) encoder hidden state
        :return:context: batch, time_step, hidden_size)
        """
        enc = self.linear_enc(encoder_out) # (batch, time_step, hidden_size)
        value = self.linear_value(encoder_out)
        h = enc.transpose(1, 2) # (batch, hidden_size, time_step)
        weights = torch.bmm(enc, h) # (batch, time_step, hidden_size)
        weights = self.softmax(weights/math.sqrt(self.hidden_size))
        context = torch.bmm(weights, value) # (batch, time_step, hidden_size)
        context = self.linear_out(torch.cat((context, encoder_out), dim=-1)) + encoder_out

        return context