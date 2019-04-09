class Config():
    def __init__(self):
        # dataset
        self.filename_train = 'DATA/LCSTS/PART_I.txt'
        self.filename_valid = 'DATA/LCSTS/PART_II.txt'
        self.filename_test = 'DATA/LCSTS/PART_III.txt'

        # trimmed data
        self.filename_trimmed_train = 'DATA/data/valid.pt'
        self.filename_trimmed_valid = 'DATA/data/valid.pt'
        self.filename_trimmed_test = 'DATA/data/test.pt'

        # bos eos
        self.bos = 2
        self.eos = 3

        # vocab
        self.src_filename_word2idx = 'DATA/data/src_word2index.pkl'
        self.src_filename_idx2word = 'DATA/data/src_index2word.pkl'
        self.src_vocab_size = 961195
        # self.src_vocab_size = 4000

        self.tgt_filename_word2idx = 'DATA/data/tgt_word2index.pkl'
        self.tgt_filename_idx2word = 'DATA/data/tgt_index2word.pkl'
        self.tgt_vocab_size = 8250
        # self.tgt_vocab_size = 4000

        self.word_share = False
        self.word_seg = True # hybird word-character

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
        self.filename_gold = 'result/gold/gold_summaries.txt'

        # Hyper Parameters
        self.LR = 0.0003
        self.batch_size = 2
        self.iters = 10000
        self.embedding_dim = 512
        self.hidden_size = 512
        self.beam_size = 10

        # transformer multi-head attention
        self.h_head = 8

        self.n_layer = 2
        self.cell = 'lstm'
        self.attn_flag = False
        self.dropout = 0
        self.bidirectional = True
        self.optimzer = 'Adam'
        self.intra_decoder = False
        self.enc_attn = False