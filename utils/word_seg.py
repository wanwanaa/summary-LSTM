import jieba


def file_word_seg(datasets):
    result = []
    for line in datasets:
        line = line.strip()
        seg_list = jieba.cut(line)
        result.append(seg_list)
    result = '\n'.join(result)
    return result