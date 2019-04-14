import torch
import torch.nn as nn


class Embeds(nn.Module):
    def __init__(self, config, vocab_size, embedding=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = config.embedding_dim
        if embedding:
            self.embeds = nn.Embedding.from_pretrained(embedding)
        else:
            self.embeds = nn.Embedding(self.vocab_size, self.embedding_dim)

    def forward(self, x):
        """
        :param x: (batch, t_len)
        :return: (batch, t_len, embedding_dim)
        """
        return self.embeds(x)