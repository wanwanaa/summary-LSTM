from models.attention import *
from models.rnn import *
from models.seq2seq import *
from models.embedding import *
from models.cnn import Encoder_cnn


def build_model(config):
    enc_embeds = Embeds(config, config.src_vocab_size)
    if config.word_share:
        dec_embeds = enc_embeds
    else:
        dec_embeds = Embeds(config, config.tgt_vocab_size)
    # if config.attn_flag == 'multi':
    #     encoder = Encoder_multi(embeds, config)
    # else:
    encoder = Encoder(enc_embeds, config)
    decoder = Decoder(dec_embeds, config)
    if config.cnn:
        cnn = Encoder_cnn(enc_embeds, config)
        model = Seq2seq(encoder, decoder, config, cnn)
    else:
        model = Seq2seq(encoder, decoder, config)
    return model


def load_model(config, filename):
    model = build_model(config)
    model.load_state_dict(torch.load(filename, map_location='cpu'))
    return model


def save_model(model, filename):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    torch.save(model.state_dict(), filename)
    print('model save at ', filename)