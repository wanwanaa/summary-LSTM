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
        self.filename_word2idx = 'DATA/data/word2index.pkl'
        self.filename_idx2word = 'DATA/data/index2word.pkl'
        self.vocab_size = 4000

        # sequence length
        self.t_len = 150
        self.s_len = 50

        # embedding
        self.filename_embedding = ''
        self.filename_trimmed_embedding = ''

        # filename
        #################################################
        self.filename_model = 'result/model/n_layer/'
        self.filename_data = 'result/data/n_layer/'
        self.filename_rouge = 'result/data/n_layer/ROUGE.txt'
        #################################################
        self.filename_gold = 'result/gold/gold_summaries.txt'

        # Hyper Parameters
        self.LR = 0.0003
        self.batch_size = 64
        self.iters = 10000
        self.embedding_dim = 512
        self.hidden_size = 512
        self.beam_size = 2

        self.n_layer = 2
        self.cell = 'lstm'
        self.attn_flag = 'mulit'
        self.dropout = 0
        self.bidirectional = True
        self.optimzer = 'Adam'
        self.intra_decoder = False