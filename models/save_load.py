from models.attention import *
from models.rnn import *
from models.seq2seq import *
from models.embedding import *


def build_model(config):
    if config.bert:
        enc_embeds = Bert_Embeds(config)
    else:
        enc_embeds = Embeds(config, config.src_vocab_size)
    if config.word_share:
        dec_embeds = enc_embeds
    else:
        if config.bert:
            dec_embeds = Bert_Embeds(config)
        else:
            dec_embeds = Embeds(config, config.tgt_vocab_size)
    # if config.attn_flag == 'multi':
    #     encoder = Encoder_multi(embeds, config)
    # else:
    encoder = Encoder(enc_embeds, config)
    decoder = Decoder(dec_embeds, config)
    model = Seq2seq(encoder, decoder, config)
    return model


def load_model(config, filename):
    model = build_model(config)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.load_state_dict(torch.load(filename, map_location='cpu'))
    return model


def save_model(model, filename):
    torch.save(model.state_dict(), filename)
    print('model save at ', filename)