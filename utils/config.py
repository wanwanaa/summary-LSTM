class Config():
    def __init__(self):
        # dataset
        self.filename_train = 'DATA/LCSTS/PART_I.txt'
        self.filename_valid = 'DATA/LCSTS/PART_II.txt'
        self.filename_test = 'DATA/LCSTS/PART_III.txt'

        # trimmed data
        self.filename_trimmed_train = 'DATA/data_hybird/valid.pt'
        self.filename_trimmed_valid = 'DATA/data_hybird/valid.pt'
        self.filename_trimmed_test = 'DATA/data_hybird/test.pt'

        # bos eos
        self.bos = 2
        self.eos = 3

        # vocab
        self.src_filename_word2idx = 'DATA/data_hybird/src_word2index.pkl'
        self.src_filename_idx2word = 'DATA/data_hybird/src_index2word.pkl'
        self.src_vocab_size = 961195
        # self.src_vocab_size = 4000

        self.tgt_filename_word2idx = 'DATA/data_hybird/tgt_word2index.pkl'
        self.tgt_filename_idx2word = 'DATA/data_hybird/tgt_index2word.pkl'
        self.tgt_vocab_size = 8250
        # self.tgt_vocab_size = 4000

        # hybird word-character
        self.word_share = False
        self.word_seg = True

        # sequence length
        self.t_len = 80
        self.s_len = 50

        # embedding
        self.filename_embedding = ''
        self.filename_trimmed_embedding = ''

        # filename
        #################################################
        self.filename_model = 'result/model/hybird/'
        self.filename_data = 'result/data/hybird/'
        self.filename_rouge = 'result/data/hybird/ROUGE.txt'
        #################################################
        self.filename_gold = 'result/gold/gold_summaries.txt'

        # Hyper Parameters
        self.LR = 0.0003
        self.batch_size = 2
        self.iters = 10000
        self.embedding_dim = 768
        self.hidden_size = 768
        self.beam_size = 10

        # transformer multi-head attention
        self.h_head = 8

        self.n_layer = 2
        self.cell = 'lstm'
        self.attn_flag = 'loung'
        self.dropout = 0
        self.bidirectional = True
        self.optimzer = 'Adam'
        self.intra_decoder = False
        self.enc_attn = False

        # bert (word_share=True)
        self.bert = False
        self.fine_tuning = False
        self.filename_bert = ''
        self.filename_cache = 'cache'
        self.local_rank = 2