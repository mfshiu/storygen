def download_data_gdown():
    # -*- coding: utf-8 -*-
    from ckiptagger import data_utils
    data_utils.download_data_gdown("./")


def gen_texts(wiki_file):
    from gensim.corpora import WikiCorpus

    wiki_corpus = WikiCorpus(wiki_file, dictionary={})
    text_num = 0

    with open('wiki_text.txt', 'w', encoding='utf-8') as f:
        for text in wiki_corpus.get_texts():
            f.write(' '.join(text)+'\n')
            text_num += 1
            if text_num % 10000 == 0:
                print('{} articles processed.'.format(text_num))

        print('{} articles processed.'.format(text_num))


def part_test():
    from ckiptagger import WS, POS, NER
    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    text = '在自然語言處理 (NLP) 的任務中，純文字的資料型態本身是相當難以進行處理的，尤其是在機器學習的任務當中。 文字型態的資料，是沒有辦法如同數值資料一樣進行 Forward Propagation 以及 Backward Propagation 的、是沒有辦法微分的，簡單來講，是沒辦法進行運算並且訓練權重網路的。 最簡單的方法，就像我在《[PyTorch] 旅館評論情感分析實戰紀錄 (0)》這篇文章中所做的一樣，將每個『相異字』(Character) 轉換成一個特定的數字。這樣一來，我們便可以將機器學習相關的技術應用在文字領域了。 不過，只是單純地轉換 Character，有時是得不到較好的結果的。對中文而言，有時使用『詞』作為句子裡的基本元件單位會更恰當；另外，只是轉成一個『數字』也很難表現出中文詞彙的多樣性，轉換成『向量』通常效果會更好一點。 Word2Vec 就是這樣的一個工具，其最早是為 Google 開發的一個工具；而今天本文的主角 Gensim 則是它的 Python 實現 (不過只有最高層是 Python、內部還是封裝了 Word2Vec 的 C 接口)。 以下就來簡單地介紹該如何使用 Gensim 這項工具來完成將『文字轉換成向量』這項工作吧！'
    # text = '傅達仁今將執行安樂死，卻突然爆出自己20年前遭緯來體育台封殺，他不懂自己哪裡得罪到電視台。'
    print("Load model WS...")
    ws = WS("./data")
    print("Load model POS...")
    pos = POS("./data")
    print("Load model NER...")
    ner = NER("./data")

    print("Run model ws...")
    ws_results = ws([text])
    print("Run model pos...")
    pos_results = pos(ws_results)
    print("Run model ner...")
    ner_results = ner(ws_results, pos_results)

    print(ws_results)
    print(pos_results)
    for name in ner_results[0]:
        print(name)


def seg_data(text_file, output_file):
    from opencc import OpenCC
    from ckiptagger import WS
    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # Initial
    print("Initial...")
    cc = OpenCC('s2t')
    train_data = open(text_file, 'r', encoding='utf-8').read()
    # print("Convert ugly to good...")
    # train_data = cc.convert(train_data)

    # train_data = jieba.lcut(train_data)
    print("Load model WS...")
    ws = WS("./data")
    print("Run model ws...")
    train_data = ws([train_data])

    print("Output...")
    train_data = [word for word in train_data[0] if word != '']
    train_data = ' '.join(train_data)
    open(output_file, 'w', encoding='utf-8').write(train_data)


if __name__ == '__main__':
    # gen_texts('zhwiki-latest-pages-articles-multistream.xml.bz2')
    # download_data_gdown()
    # part_test()
    seg_data('wiki_text_zh-TW.txt', 'seg.txt')
