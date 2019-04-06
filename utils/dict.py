import pickle
import os
import jieba


class Vocab():
    def __init__(self, config, src_datasets=None, tgt_datasets=None):
        self.src_filename_idx2word = config.src_filename_idx2word
        self.src_filename_word2idx = config.src_filename_word2idx
        self.vocab_size = config.src_vocab_size
        self.src_word2idx = {}
        self.src_idx2word = {}

        self.tgt_filename_idx2word = config.tgt_filename_idx2word
        self.tgt_filename_word2idx = config.tgt_filename_word2idx
        self.vocab_size = config.tgt_vocab_size
        self.tgt_word2idx = {}
        self.tgt_idx2word = {}

        if src_datasets is not None:
            if config.word_seg:
                self.src_vocab = self._get_vocab_word(src_datasets)
            self.src_idx2word = self.index2word(config.src_vocab_size, self.src_vocab)
            self.src_word2idx = self.word2index(self.src_word2idx, config.src_vocab_size, self.src_vocab)
            self.writeFile(self.src_idx2word, self.src_filename_idx2word)
            self.writeFile(self.src_word2idx, self.src_filename_word2idx)
        else:
            self.src_idx2word = self.load_vocab(self.src_filename_idx2word)
            self.src_word2idx = self.load_vocab(self.src_filename_word2idx)

        if config.word_share:
            self.tgt_idx2word = self.src_idx2word
            self.tgt_word2idx = self.src_word2idx
        else:
            if tgt_datasets is not None:
                self.tgt_vocab = self._get_vocab(tgt_datasets)
                self.tgt_idx2word = self.index2word(config.tgt_vocab_size, self.tgt_vocab)
                self.tgt_word2idx = self.word2index(self.tgt_word2idx, config.tgt_vocab_size, self.tgt_vocab)
                self.writeFile(self.tgt_idx2word, self.tgt_filename_idx2word)
                self.writeFile(self.tgt_word2idx, self.tgt_filename_word2idx)
            else:
                self.tgt_idx2word = self.load_vocab(self.tgt_filename_idx2word)
                self.tgt_word2idx = self.load_vocab(self.tgt_filename_word2idx)

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

    def _get_vocab_word(self, datasets):
        vocab = {}
        for line in datasets:
            line = line.strip()
            line = jieba.cut(line)
            for word in line:
                flag = vocab.get(word)
                if flag:
                    vocab[word] += 1
                else:
                    vocab[word] = 0
        vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
        return vocab

    # vocab word2idx
    def word2index(self, word2idx, vocab_size, vocab):
        word2idx['<pad>'] = 0
        word2idx['<unk>'] = 1
        word2idx['<bos>'] = 2
        word2idx['<eos>'] = 3
        for i in range(vocab_size-4):
            word2idx[vocab[i][0]] = i + 4
        return word2idx

    # vocab idx2word
    def index2word(self, vocab_size, vocab):
        idx2word = ['<pad>', '<unk>', '<bos>', '<eos>']
        for i in range(vocab_size - 4):
            idx2word.append(vocab[i][0])
        return idx2word

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