import math
import jieba
import jieba.posseg as psg
from gensim import corpora, models
from jieba import analyse
import pandas as pd
import numpy as np
from pandas import DataFrame, Series


# 停用词列表
def get_stopword_list():
    stop_word_path = 'stopwords.txt'
    stopword_list = [sw.replace('\n', '')
                     for sw in open(stop_word_path).readlines()]
    return stopword_list


# 分词
def seg_to_list(sentense, pos=False):
    if not pos:
        seg_list = jieba.cut(sentense)
    else:
        seg_list = psg.cut(sentense)
    return seg_list


# 去除干扰词（根据停用词表，词性）
def word_filter(seg_list, pos=False):
    stopword_list = get_stopword_list()
    filter_list = []
    for seg in seg_list:
        if not pos:
            word = seg
            flag = 'n'
        else:
            word = seg.word
            flag = seg.flag
        if not flag.startswith('n'):
            continue
        if word not in stopword_list and len(word) > 1:
            filter_list.append(word)
    return filter_list


# 加载语料库，去干扰词生成训练文本列表
def load_data(pos=False, corpus_path='corpus.xlsx'):
    df = pd.read_excel(corpus_path)
    doc_list = [df.loc[i, '内容'] for i in df.index]
    doc_list = [seg_to_list(sentense, pos) for sentense in doc_list]
    doc_list = [word_filter(seg_list, pos) for seg_list in doc_list]
    return doc_list


# idf值统计
def train_idf(doc_list):
    idf_dic = {}
    tt_count = len(doc_list)
    for doc in doc_list:
        for word in set(doc):
            idf_dic[word] = idf_dic.get(word, 0) + 1.0
    for k, v in idf_dic.items():
        idf_dic[k] = math.log(tt_count / (1.0+v))
    default_idf = math.log(tt_count / (1.0))
    return idf_dic, default_idf


# TF-IDF类
class Tfidf:
    def __init__(self, idf_dic, default_idf, word_list, keyword_num):
        self.word_list = word_list
        self.idf_dic, self.default_idf = idf_dic, default_idf
        self.tf_dic = self.get_tf_dic()
        self.keyword_num = keyword_num

    # 统计tf值
    def get_tf_dic(self):
        tf_dic = {}
        for word in self.word_list:
            tf_dic[word] = tf_dic.get(word, 0.0) + 1.0
        tt_count = len(self.word_list)
        for k, v in tf_dic.items():
            tf_dic[k] = float(v) / tt_count
        return tf_dic

    # 计算tf-idf
    def get_tfidf(self):
        tfidf_dic = {}
        for word in self.word_list:
            idf = self.idf_dic.get(word, self.default_idf)
            tf = self.tf_dic.get(word, 0)
            tfidf = tf * idf
            tfidf_dic[word] = tfidf
        keywords = sorted(
            tfidf_dic, key=lambda x: tfidf_dic[x])[-self.keyword_num:]
        return keywords


# tf-idf算法调用接口
def tfidf_extract(text_number, pos=False, keyword_num=10):
    doc_list = load_data(pos)
    idf_dic, default_idf = train_idf(doc_list)
    df = pd.read_excel('corpus.xlsx')
    text = df.loc[text_number, '内容']
    word_seg_list = seg_to_list(text, pos)
    word_list = word_filter(word_seg_list, pos)
    tfidf_model = Tfidf(idf_dic, default_idf, word_list, keyword_num)
    keywords = tfidf_model.get_tfidf()
    return keywords


class TopicModel():
    def __init__(self, doc_list, keyword_num, model='LSI', num_topics=4):
        self.dictionary = corpora.Dictionary(doc_list)
        corpus = [self.dictionary.doc2bow(doc) for doc in doc_list]
        self.tfidf_model = models.TfidfModel(corpus)
        self.corpus_tfidf = self.tfidf_model[corpus]
        self.keyword_num = keyword_num
        self.num_topics = num_topics
        if model == 'LSI':
            self.model = self.train_lsi()
        else:
            self.model = self.train_lda()
        word_dic = self.word_dictionary(doc_list)
        self.wordtopic_dic = self.get_wordtopic(word_dic)

    def train_lsi(self):
        lsi = models.LsiModel(
            self.corpus_tfidf, id2word=self.dictionary, num_topics=self.num_topics)
        return lsi

    def train_lda(self):
        lda = models.LdaModel(
            self.corpus_tfidf, id2word=self.dictionary, num_topics=self.num_topics)
        return lda

    # 生成词空间
    def word_dictionary(self, doc_list):
        dictionary = []
        for doc in doc_list:
            dictionary.extend(doc)
        dictionary = list(set(dictionary))
        return dictionary

    # 获取每个词对应的主题
    def get_wordtopic(self, word_dic):
        wordtopic_dic = {}
        for word in word_dic:
            single_list = [word]
            wordcorpus = self.tfidf_model[self.dictionary.doc2bow(single_list)]
            wordtopic = self.model[wordcorpus]
            wordtopic_dic[word] = wordtopic
        return wordtopic_dic

    # 计算词与文档文本相似度，取keyword_num个词作为关键词
    def get_simword(self, word_list):
        sentcorpus = self.tfidf_model[self.dictionary.doc2bow(word_list)]
        senttopic = self.model[sentcorpus]

        # 计算余弦相似度
        def calsim(l1, l2):
            a, b, c = 0.0, 0.0, 0.0
            for t1, t2 in zip(l1, l2):
                x1 = t1[1]
                x2 = t2[1]
                a += x1 * x1
                b += x1 * x1
                c += x2 * x2
            sim = a / math.sqrt(b * c) if not (b * c) == 0.0 else 0.0
            return sim

        # 计算词主题与文章主题的相似度，从而提取关键词
        sim_dic = {}
        for k, v in self.wordtopic_dic.items():
            if k not in word_list:
                continue
            sim = calsim(v, senttopic)
            sim_dic[k] = sim
        keywords = sorted(
            sim_dic, key=lambda x: sim_dic[x])[-self.keyword_num:]
        return keywords


# 主题模型算法调用接口
def topic_extract(text_number, model, pos=False, keyword_num=10):
    doc_list = load_data(pos)
    topic_model = TopicModel(doc_list, keyword_num, model)
    df = pd.read_excel('corpus.xlsx')
    text = df.loc[text_number, '内容']
    word_seg_list = seg_to_list(text, pos)
    word_list = word_filter(word_seg_list, pos)
    keywords = topic_model.get_simword(word_list)
    return keywords


# textrank算法调用接口
def textrank_extract(text_number, keyword_num=10):
    df = pd.read_excel('corpus.xlsx')
    text = df.loc[text_number, '内容']
    keywords = analyse.textrank(text, keyword_num)
    return keywords


if __name__ == '__main__':
    print(tfidf_extract(0, pos=True, keyword_num=10))
    print(textrank_extract(0))
    print(topic_extract(0, model='LDA', pos=True, keyword_num=10))
