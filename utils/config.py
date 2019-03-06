class Config():
    def __init__(self):
        # dataset
        self.filename_train = 'DATA/LCSTS/PART_I.txt'
        self.filename_valid = 'DATA/LCSTS/PART_II.txt'
        self.filename_test = 'DATA/LCSTS/PART_III.txt'

        # embedding
        self.dim = 512

        # trimmed data
        self.filename_trimmed_train = 'DATA/data/train.pt'
        self.filename_trimmed_valid = 'DATA/data/valid.pt'
        self.filename_trimmed_test = 'DATA/data/test.pt'

        # sequence length
        self.t_len = 150
        self.s_len = 50

        # bos
        self.bos = 2

        # vocab
        self.filename_word2idx = 'DATA/data/word2index.pkl'
        self.filename_idx2word = 'DATA/data/index2word.pkl'
        self.vocab_size = 4000

        # Hyper Parameters
        self.LR = 0.001