class Config():
    def __init__(self):
        # # dataset
        # self.filename_train_src = 'DATA/raw_data/LCSTS_clean/train.source'
        # self.filename_train_tgt = 'DATA/raw_data/LCSTS_clean/train.target'
        # self.filename_valid_src = 'DATA/raw_data/LCSTS_clean/valid.source'
        # self.filename_valid_tgt = 'DATA/raw_data/LCSTS_clean/valid.target'
        # self.filename_test_src = 'DATA/raw_data/LCSTS_clean/test.source'
        # self.filename_test_tgt = 'DATA/raw_data/LCSTS_clean/test.target'

        # self.filename_train = 'DATA/LCSTS/PART_I.txt'
        # self.filename_valid = 'DATA/LCSTS/PART_II.txt'
        # self.filename_test = 'DATA/LCSTS/PART_III.txt'

        # # # trimmed data LCSTS
        # self.filename_trimmed_train = 'DATA/data/clean/data_hybird/train.pt'
        # self.filename_trimmed_valid = 'DATA/data/clean/data_hybird/valid.pt'
        # self.filename_trimmed_test = 'DATA/data/clean/data_hybird/test.pt'
        # #
        # #
        # #
        # # # vocab
        # self.src_filename_word2idx = 'DATA/data/clean/data_hybird/src_word2index.pkl'
        # self.src_filename_idx2word = 'DATA/data/clean/data_hybird/src_index2word.pkl'
        # self.src_vocab_size = 523566
        # # self.src_vocab_size = 961195
        #
        # self.tgt_filename_word2idx = 'DATA/data/clean/data_hybird/tgt_word2index.pkl'
        # self.tgt_filename_idx2word = 'DATA/data/clean/data_hybird/tgt_index2word.pkl'
        # self.tgt_vocab_size = 8250

        # char
        self.filename_trimmed_train = 'DATA/data/LCSTS2.0/data_char/valid.pt'
        self.filename_trimmed_valid = 'DATA/data/LCSTS2.0/data_char/valid.pt'
        self.filename_trimmed_test = 'DATA/data/LCSTS2.0/data_char/valid.pt'

        self.src_filename_word2idx = 'DATA/data/LCSTS2.0/data_char/src_word2index.pkl'
        self.src_filename_idx2word = 'DATA/data/LCSTS2.0/data_char/src_index2word.pkl'

        self.src_vocab_size = 4000
        self.tgt_vocab_size = 4000

        # # gigaword
        # self.filename_trimmed_train = 'DATA/data/gigaword/train.pt'
        # self.filename_trimmed_valid = 'DATA/data/gigaword/valid.pt'
        # self.filename_trimmed_test = 'DATA/data/gigaword/test.pt'
        #
        # self.src_filename_word2idx = 'DATA/data/gigaword/word2index.pkl'
        # self.src_filename_idx2word = 'DATA/data/gigaword/index2word.pkl'
        #
        # self.src_vocab_size = 119504
        # self.tgt_vocab_size = 119504
        #
        self.word_share = True
        self.word_seg = False # hybird word-character

        # bos eos
        self.bos = 2
        self.eos = 3

        # sequence length
        self.t_len = 150
        self.s_len = 50

        # embedding
        self.filename_embedding = ''
        self.filename_trimmed_embedding = ''

        # filename
        #################################################
        self.filename_model = 'result/model/'
        self.filename_data = 'result/data/'
        self.filename_rouge = 'result/data/ROUGE.txt'
        #################################################
        self.filename_gold = 'result/gold/valid_summaries.txt'

        # Hyper Parameters
        self.LR = 0.0003
        self.batch_size = 32
        self.iters = 10000
        self.embedding_dim = 512
        self.hidden_size = 512
        self.beam_size = 5

        # transformer multi-head attention
        self.h_head = 8

        self.n_layer = 2
        self.cell = 'lstm'
        self.attn_flag = 'luong'
        self.dropout = 0
        self.bidirectional = True
        self.optimzer = 'Adam'
        self.intra_decoder = False
        self.enc_attn = False
        self.cnn = 2 # 0: None
                     # 1: hidden state
                     # 2: encoder outputs