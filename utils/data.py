import torch
import torch.utils.data as data_util
import numpy as np
from scipy.stats import truncnorm


class Datasets():
    def __init__(self, config):
        self.train_text, self.train_summary = self._get_datasets_train(config.filename_train)
        self.valid_text, self.valid_summary = self._get_datasets(config.filename_valid)
        self.test_text, self.test_summary = self._get_datasets(config.filename_test)

    def _get_datasets_train(self, filename):
        text = []
        summary = []
        group = []
        i = 0
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                group.append(line.strip())
                i += 1
                if i % 8 == 0:
                    summary.append(group[2])
                    text.append(group[5])
                    group = []
                    i = 0
        return text, summary

    # vaild, test(human label)
    def _get_datasets(self, filename):
        text = []
        summary = []
        group = []
        i = 0
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                group.append(line.strip())
                i += 1
                if i % 9 == 0:
                    label = int(list(group[1].split('<')[1])[-1])
                    if label >= 3:
                        summary.append(group[3])
                        text.append(group[6])
                    group = []
                    i = 0
        return text, summary


# save pt
def get_trimmed_datasets(datasets, word2idx, max_length):
    data = np.zeros([len(datasets), max_length])
    k = 0
    for line in datasets:
        line = list(line)
        sen = np.zeros(max_length, dtype=np.int32)
        for i in range(max_length):
            if i == len(line):
                sen[i] = word2idx['<eos>']
                break
            else:
                flag = word2idx.get(line[i])
                if flag is None:
                    sen[i] = word2idx['<unk>']
                else:
                    sen[i] = word2idx[line[i]]
        data[k] = sen
        k += 1
    data = torch.from_numpy(data).type(torch.LongTensor)
    return data


def save_data(text, summary, word2idx, t_len, s_len, filename):
    text = get_trimmed_datasets(text, word2idx, t_len)
    summary = get_trimmed_datasets(summary, word2idx, s_len)
    data = data_util.TensorDataset(text, summary)
    print('data save at ', filename)
    torch.save(data, filename)


def get_embeddings(config, vocab):
    embeddings = np.zeros((config.vocab_size, config.embedding_dim))
    flag = list(np.arange(0, 4000))
    with open(config.filename_embedding, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            if word in vocab.word2idx.keys():
                flag.remove(vocab.word2idx[word])
                embedding = [float(x) for x in line[1:]]
                embeddings[vocab.word2idx[word]] = embedding
    for i in flag:
        np.random.seed(i)
        embedding = truncnorm.rvs(-2, 2, size=config.embedding_dim)
        embeddings[i] = embedding
    embeddings = torch.from_numpy(embeddings)
    torch.save(embeddings, config.filename_trimmed_embedding)
    print('embeddings save at:', config.filename_trimmed_embedding)