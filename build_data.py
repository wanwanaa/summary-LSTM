import argparse
from utils import *


def main():
    config = Config()

    parser = argparse.ArgumentParser()
    parser.add_argument('--t_len', '-t', metavar='NUM', type=int, help='display max_length')
    parser.add_argument('--s_len', '-s', metavar='NUM', type=int, help='display summary_length')

    args = parser.parse_args()
    if args.t_len:
        config.t_len = args.t_len
    if args.s_len:
        config.s_len = args.s_len

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


# test trimmed file result
def test():
    config = Config()
    vocab = Vocab(config)

    test = torch.load(config.filename_trimmed_test)
    sen = index2sentence(np.array(test[0][0]), vocab.idx2word)
    print(sen)


if __name__ == '__main__':
    main()
    # test()
