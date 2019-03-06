from utils import *


def main():
    config = Config()

    # get datasets(train, valid, test)
    print('Loading data ... ...')
    datasets = Datasets(config)

    # get vocab(idx2word, word2idx)
    print('Building vocab ... ...')
    vocab = Vocab(config, datasets.train_text)

    # save pt(train, valid, test)
    save_data(datasets.train_text, datasets.train_summary, vocab.word2idx, config.t_len, config.s_len, config.filename_trimmed_train)
    save_data(datasets.valid_text, datasets.valid_summary, vocab.word2idx, config.t_len, config.s_len, config.filename_trimmed_valid)
    save_data(datasets.test_text, datasets.test_summary, vocab.word2idx, config.t_len, config.s_len, config.filename_trimmed_test)


def test():
    config = Config()
    vocab = Vocab(config)

    test = torch.load(config.filename_trimmed_test)
    sen = index2sentence(np.array(test[0][0]), vocab.idx2word)
    print(sen)


if __name__ == '__main__':
    main()
    # test()
