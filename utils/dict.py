import pickle
import os


class Vocab():
    def __init__(self, config, datasets=None):
        self.filename_idx2word = config.filename_idx2word
        self.filename_word2idx = config.filename_word2idx
        self.vocab_size = config.vocab_size
        self.word2idx = {}
        self.idx2word = {}

        if datasets is not None:
            self.vocab = self._get_vocab(datasets)
            self.idx2word = self.index2word()
            self.word2idx = self.word2index()
            self.writeFile(self.idx2word, self.filename_idx2word)
            self.writeFile(self.word2idx, self.filename_word2idx)
        else:
            self.idx2word = self.load_vocab(self.filename_idx2word)
            self.word2idx = self.load_vocab(self.filename_word2idx)

    # check whether the given 'filename' exists
    # raise a FileNotFoundError when file not found
    def file_check(self, filename):
        if os.path.isfile(filename) is False:
            raise FileNotFoundError('No such file or directory: {}'.format(filename))

    # get the vocabulary and sort it by frequency
    def _get_vocab(self, datasets):
        vocab = {}
        for line in datasets:
            line = list(line)
            for c in line:
                flag = vocab.get(c)
                if flag:
                    vocab[c] += 1
                else:
                    vocab[c] = 0
        vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
        return vocab

    # vocab word2idx
    def word2index(self):
        self.word2idx['<pad>'] = 0
        self.word2idx['<unk>'] = 1
        self.word2idx['<bos>'] = 2
        self.word2idx['<eos>'] = 3
        for i in range(self.vocab_size-4):
            self.word2idx[self.vocab[i][0]] = i + 4
        return self.word2idx

    # vocab idx2word
    def index2word(self):
        self.idx2word = ['<pad>', '<unk>', '<bos>', '<eos>']
        for i in range(self.vocab_size - 4):
            self.idx2word.append(self.vocab[i][0])
        return self.idx2word

    # save vocab in .pkl
    def writeFile(self, vocab, filename):
        with open(filename, 'wb') as f:
            pickle.dump(vocab, f)
        print('vocab saved at:', filename)

    # load vocab
    def load_vocab(self, filename):
        print('load vocab from', filename)
        self.file_check(filename)
        f = open(filename, 'rb')
        return pickle.load(f)


# convert idx to words, if idx <bos> is stop, return sentence
def index2sentence(index, idx2word):
    sen = []
    for i in range(len(index)):
        if idx2word[index[i]] == '<eos>':
            break
        if idx2word[index[i]] == '<bos>':
            continue
        else:
            sen.append(idx2word[index[i]])
    if len(sen) == 0:
        sen.append('<unk>')
    return sen