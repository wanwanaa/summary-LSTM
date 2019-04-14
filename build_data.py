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
    vocab = Vocab(config, datasets.train_text, datasets.train_summary)

    # save pt(train, valid, test)
    save_data(datasets.train_text, datasets.train_summary, vocab.src_word2idx, vocab.tgt_word2idx, config.t_len, config.s_len, config.filename_trimmed_train)
    save_data(datasets.valid_text, datasets.valid_summary, vocab.src_word2idx, vocab.tgt_word2idx, config.t_len, config.s_len, config.filename_trimmed_valid)
    save_data(datasets.test_text, datasets.test_summary, vocab.src_word2idx, vocab.tgt_word2idx, config.t_len, config.s_len, config.filename_trimmed_test)


# test trimmed file result
def test():
    config = Config()
    vocab = Vocab(config)

    test = torch.load('DATA/data_hybird/test.pt')
    sen = index2sentence(np.array(test[0][0]), vocab.src_idx2word)
    print(sen)


def write_file(datasets, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(datasets))


if __name__ == '__main__':
    # config = Config()
    # vocab = Vocab(config)
    # datasets = Datasets(config)
    # save_data(datasets.train_text, datasets.train_summary, vocab.src_word2idx, vocab.tgt_word2idx, config.t_len,
    #           config.s_len, config.filename_trimmed_train)
    # main()
    test()
    # config = Config()
    # vocab = Vocab(config)
    # print(vocab.src_word2idx)
    # config = Config()
    # datasets = Datasets(config)
    # train_src = datasets.train_text
    # write_file(train_src, 'src-train.txt')
    # train_tgt = datasets.train_summary
    # write_file(train_tgt, 'tgt-train.txt')
    # valid_src = datasets.valid_text
    # write_file(valid_src, 'src-valid.txt')
    # valid_tgt = datasets.valid_summary
    # write_file(valid_tgt, 'tgt-valid.txt')
    # test_src = datasets.test_text
    # write_file(test_src, 'src-test.txt')
    # test_tgt = datasets.test_summary
    # write_file(test_tgt, 'tgt-test.txt')
