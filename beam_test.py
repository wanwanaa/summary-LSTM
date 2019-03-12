from models import *
from utils import *


def beam_test(model, config, idx2word, epoch):
    model.eval()
    test_loader = data_load(config.filename_trimmed_test, config.batch_size, False)
    result = []
    for step, batch in enumerate(test_loader):
        x, y = batch

        # argmax
        # _, idx = model.sample(x, y)

        # beam batch
        idx = model.beam_search(x)
        for i in range(x.size(0)):
            sen = index2sentence(list(idx[i]), idx2word)
            result.append(' '.join(sen))

        # write result
        filename_data = config.filename_data + 'summary_' + str(epoch) + '.txt'
    with open(filename_data, 'w', encoding='utf-8') as f:
        f.write('\n'.join(result))

    # rouge
    score = rouge_score(config.filename_gold, filename_data)

    # write rouge
    write_rouge(config.filename_rouge, score, epoch)

    # print rouge
    print('epoch:', epoch, '|ROUGE-1 f: %.4f' % score['rouge-1']['f'],
          ' p: %.4f' % score['rouge-1']['p'],
          ' r: %.4f' % score['rouge-1']['r'])
    print('epoch:', epoch, '|ROUGE-2 f: %.4f' % score['rouge-2']['f'],
          ' p: %.4f' % score['rouge-2']['p'],
          ' r: %.4f' % score['rouge-2']['r'])
    print('epoch:', epoch, '|ROUGE-L f: %.4f' % score['rouge-l']['f'],
          ' p: %.4f' % score['rouge-l']['p'],
          ' r: %.4f' % score['rouge-l']['r'])


if __name__ == '__main__':
    config = Config()
    vocab = Vocab(config)
    filename = config.filename_model + 'model_9.pkl'
    model = load_model(config, filename)
    beam_test(model, config, vocab.idx2word, 9)