import numpy as np
import collections
import pandas as pd
import d2lzh as d2l
from mxnet import gluon, init, nd
from mxnet.contrib import text
from mxnet.gluon import data as gdata, loss as gloss, nn, rnn, utils as gutils
import os
import random
import tarfile
import csv
import jieba

def jiebaCut(words):#结巴分词，空格隔开
    return " ".join(jieba.cut(words))



def read_imdb(folder):  # 读取数据集
    data = []
    with open(folder, 'r',encoding='UTF-8') as f:
        rows=list(csv.reader(f))   #读csv   
    random.shuffle(rows)           #乱序
    for row in rows:
        data.append([jiebaCut(row[0]),row[1]])          #将评论与感情记号打入data
    return data


        
def get_tokenized_imdb(data):  # 向量化词
    def tokenizer(text):
        return [tok.lower() for tok in text.split(' ')]#按空格将每个词都切成列表项
    return [tokenizer(review) for review, _ in data]#tokenizer向量化

train_data, test_data = read_imdb('./traindata.csv'), read_imdb('./testdata.csv')#读取数据集

def get_vocab_imdb(data):  # 生成词典
    tokenized_data = get_tokenized_imdb(data)#向量化词
    counter = collections.Counter([tk for st in tokenized_data for tk in st])#每个词出现的次数
    return text.vocab.Vocabulary(counter,reserved_tokens=['<pad>'])#生成词典，截断或者补<pad>

vocab = get_vocab_imdb(train_data)# 生成词典


def preprocess_imdb(data, vocab):  # 每条评论进行分词，并通过词典转换成词索引
    max_l = 500  # 将每条评论通过截断或者补'<pad>'，使得长度变成500
    def pad(x):
        return x[:max_l] if len(x) > max_l else x + [
            vocab.token_to_idx['<pad>']] * (max_l - len(x))

    tokenized_data = get_tokenized_imdb(data)
    features = nd.array([pad(vocab.to_indices(x)) for x in tokenized_data])
    labels = nd.array([score for _, score in data])
    return features, labels
#创建数据迭代器。每次迭代将返回一个小批量的数据。
batch_size = 64#训练集中小批量的个数。
train_set = gdata.ArrayDataset(*preprocess_imdb(train_data, vocab))
test_set = gdata.ArrayDataset(*preprocess_imdb(test_data, vocab))
train_iter = gdata.DataLoader(train_set, batch_size, shuffle=True)
test_iter = gdata.DataLoader(test_set, batch_size)

class BiRNN(nn.Block):#block模块构造类
    def __init__(self, vocab, embed_size, num_hiddens, num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)#嵌入层
        self.encoder = rnn.LSTM(num_hiddens, num_layers=num_layers,bidirectional=True, input_size=embed_size)#序列编码的隐藏层，
        self.decoder = nn.Dense(2)#生成分类结果的输出层。

    def forward(self, inputs):
        # inputs的形状是(批量大小, 词数)，因为LSTM需要将序列作为第一维，所以将输入转置后
        # 再提取词特征，输出形状为(词数, 批量大小, 词向量维度)
        embeddings = self.embedding(inputs.T)
        # rnn.LSTM只传入输入embeddings，因此只返回最后一层的隐藏层在各时间步的隐藏状态。
        # outputs形状是(词数, 批量大小, 2 * 隐藏单元个数)
        outputs = self.encoder(embeddings)
        # 连结初始时间步和最终时间步的隐藏状态作为全连接层输入。它的形状为
        # (批量大小, 4 * 隐藏单元个数)。
        encoding = nd.concat(outputs[0], outputs[-1])
        outs = self.decoder(encoding)
        return outs
embed_size, num_hiddens, num_layers, ctx = 300, 100, 2, d2l.try_all_gpus()#词向量数目，隐藏层节点数，重复出现的层数。gpu
net = BiRNN(vocab, embed_size, num_hiddens, num_layers)#含两个隐藏层的双向循环神经网络。
net.initialize(init.Xavier(), ctx=ctx)
glove_embedding = text.embedding.create('fasttext', pretrained_file_name='wiki.zh.vec', vocabulary=vocab)#使用fasttext中文预训练的词向量
net.embedding.weight.set_data(glove_embedding.idx_to_vec)#设定权重
net.embedding.collect_params().setattr('grad_req', 'null')#设定梯度反馈为null
lr, num_epochs = 0.001, 100
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})#优化器Adam
loss = gloss.SoftmaxCrossEntropyLoss()#交叉熵损失
d2l.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs)


def predict_sentiment(net, vocab, sentence):

    sentence = nd.array(vocab.to_indices(sentence), ctx=d2l.try_gpu())

    label = nd.argmax(net(sentence.reshape((1, -1))), axis=1)

    return '好' if label.asscalar() == 1 else '差'
print(predict_sentiment(net, vocab, jiebaCut('这家饭店太好吃了，很喜欢，下次一定会来的').split( )))
print(predict_sentiment(net, vocab, jiebaCut('一点也不好吃，垃圾').split( )))
print(predict_sentiment(net, vocab, jiebaCut('油炸大虾量很足，已种草，到时候推荐朋友来吃').split( )))
print(predict_sentiment(net, vocab, jiebaCut('饭里有苍蝇，差评！').split( )))
print(predict_sentiment(net, vocab, jiebaCut('物美价廉').split( )))
print(predict_sentiment(net, vocab, jiebaCut('和图片上差别很大，不知道这么高分是怎么来的').split( )))
