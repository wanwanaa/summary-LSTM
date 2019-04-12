import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertConfig
import os


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


class Bert_Embeds(nn.Module):
    def __init__(self, config):
        super().__init__()
        # if config.train:
        #     if torch.cuda.is_available():
        #         self.model = BertModel.from_pretrained('bert-base-chinese',
        #                                                cache_dir=os.path.join(config.filename_cache,
        #                                                                       'distributed_{}'.format(config.local_rank)))
        #     else:
        #         self.model = BertModel.from_pretrained('bert-base-chinese',
        #                                                cache_dir=os.path.join(config.filename_cache, 'distributed'))
        #     # self.model = BertModel(bert_config)
        # else:
        #     # self.model = BertModel.from_pretrained('bert-base-chinese')
        #     bert_config = BertConfig()
        #     self.model = BertModel(bert_config)
        # # self.linear_out = nn.Linear(12*786, config.embedding_dim)
        self.model = BertModel.from_pretrained('bert-base-chinese')

    def forward(self, ids):
        #  ids(batch, len)
        # print(ids.size())
        if len(ids.size()) == 1:
            ids = ids.unsqueeze(1)
        if torch.cuda.is_available():
            segment_ids = torch.ones(ids.size()).type(torch.cuda.LongTensor)
        else:
            segment_ids = torch.ones(ids.size()).type(torch.LongTensor)
        encoded_layers, _ = self.model(ids, segment_ids)
        # # print(encoded_layers[0].size())
        # h = torch.cat((encoded_layers[0], encoded_layers[1]), dim=2)
        # for i in range(2, 12):
        #     torch.cat((h, encoded_layers[i]), dim=2)
        # h = h.view(-1, )
        # h = self.linear_out(h)

        return encoded_layers[-1].squeeze()