import jieba


def writeFile(datasets, filename):
    datasets = '\n'.join(datasets)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(datasets)


def get_datasets_src(filename):
    datasets = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            line = jieba.cut(line)
            line = list(line)
            line = ' [SEP] '.join(line)
            line = '[CLS] ' + line + ' [SEP]'
            datasets.append(line)
    return datasets


def get_datasets_tgt(filename):
    datasets = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            line = line + ' [SEP]'
            datasets.append(line)
    return datasets


if __name__ == '__main__':
    filename_train_src = 'clean_data/train.source'
    filename_train_tgt = 'clean_data/train.target'
    filename_valid_src = 'clean_data/valid.source'
    filename_valid_tgt = 'clean_data/valid.target'
    filename_test_src = 'clean_data/test.source'
    filename_test_tgt = 'clean_data/test.target'
	
    filename_src_train = 'seg/train.source'
    filename_tgt_train = 'seg/train.target'
    filename_src_valid = 'seg/valid.source'
    filename_tgt_valid = 'seg/valid.target'
    filename_src_test = 'seg/test.source'
    filename_tgt_test = 'seg/test.target'
	

    datasets = get_datasets_src(filename_train_src)
    writeFile(datasets, filename_src_train)
    datasets = get_datasets_src(filename_valid_src)
    writeFile(datasets, filename_src_valid)
    datasets = get_datasets_src(filename_test_src)
    writeFile(datasets, filename_src_test)

    datasets = get_datasets_tgt(filename_train_tgt)
    writeFile(datasets, filename_tgt_train)
    datasets = get_datasets_tgt(filename_valid_tgt)
    writeFile(datasets, filename_tgt_valid)
    datasets = get_datasets_src(filename_test_tgt)
    writeFile(datasets, filename_tgt_test)