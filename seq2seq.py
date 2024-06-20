import torch
import lzb_model as d2l
import os
import jieba
from matplotlib import pyplot as plt
from torch import nn

import matplotlib.pyplot as plt
from d2l import torch as d2l
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class Animator: #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
    ylim=None, xscale='linear', yscale='linear',
    fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
    figsize=(7.5, 5.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)

        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        self.xdata = x
        self.ydata = y
        plt.draw()
        plt.pause(0.5)





def read_txt(file):
    '''读取原始文本'''
    with open(file, 'r', encoding='gbk', errors='ignore') as f:
        r_txt = f.read()
    r_txt = r_txt.replace('本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com',
                          '')
    return r_txt


def produce_pun(path):
    '''生成标点符号列表'''
    with open(os.path.join(path, 'cn_punctuation.txt'), 'r', encoding='utf-8', errors='ignore') as f:
        punction = f.read()
    # with open(os.path.join(path, 'cn_stopwords.txt'), 'r', encoding='utf-8', errors='ignore') as f:
    #     punction = f.read()
    # 去除停用词
    punction = punction.replace('\n', '')
    return punction


def produce_word(r_txt, punction, mode):
    '''生成列表，并去除标点符号或者停用词'''
    if mode == 'word':
        txt_words = [word for word in jieba.cut(r_txt) if (word not in punction) and (not word.isspace())]
    else:
        temp_words = [word for word in jieba.cut(r_txt) if (word not in punction) and (not word.isspace())]
        txt_words = [char for word in temp_words for char in word]
    return txt_words


def produce_sentence(path, file=''):
    """字元模型"""
    punction = produce_pun(path)
    if file == '':
        sentences = [char for file in os.listdir(os.path.join(path, 'txt'))
                      for char in produce_word(read_txt(os.path.join(path, 'txt', file)), punction, mode='char')]
    else:
        r_txt = read_txt(os.path.join(path, 'txt', file))
        sentences = produce_word(r_txt, punction, mode='char')
    return sentences


def produce_sentences(path, file=''):
    """词元模型"""
    punction = produce_pun(path)
    if file == '':
        sentences = [word for file in os.listdir(os.path.join(path, 'txt'))
                      for word in produce_word(read_txt(os.path.join(path, 'txt', file)), punction, mode='word')]
    else:
        r_txt = read_txt(os.path.join(path, 'txt', file))
        sentences = produce_word(r_txt, punction, mode='word')
    return sentences


def load_corpus(max_tokens=-1):
    """返回词元索引列表和词表"""
    sentences = produce_sentences('./novel_set', '碧血剑.txt')
    # 加载指定文本
    # sentences = produce_sentence('./novel_set')
    # 加载全部文本
    vocab = d2l.Vocab(sentences, min_freq=1)
    # 因为每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    # corpus = [vocab[token] for line in sentences for token in line]
    corpus = [vocab[token] for token in sentences]
    # 生成词元的索引列表
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


class SeqDataLoader:
    """加载序列数据的迭代器"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = d2l.seq_data_iter_random
        else:
            self.data_iter_fn = d2l.seq_data_iter_sequential
            self.corpus, self.vocab = load_corpus(max_tokens)
            self.batch_size, self.num_steps = batch_size, num_steps
    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_data(batch_size, num_steps,
              use_random_iter=False, max_tokens=10000):
    """返回迭代器和词表"""
    data_iter = SeqDataLoader(
                    batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab


class Seq2SeqEncoder(d2l.Encoder):
    """以GRU为基础的自编码器结构"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)

    def forward(self, X, *args):
        # The output `X` shape: (`batch_size`, `num_steps`, `embed_size`)
        X = self.embedding(X)
        # In RNN models, the first axis corresponds to time steps
        X = X.permute(1, 0, 2)
        # When state is not mentioned, it defaults to zeros
        output, state = self.rnn(X)
        # `output` shape: (`num_steps`, `batch_size`, `num_hiddens`)
        # `state` shape: (`num_layers`, `batch_size`, `num_hiddens`)
        return output, state


class Seq2SeqDecoder(d2l.Decoder):
    """用于序列到序列学习的循环神经网络解码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
    dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                    dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)
    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]
    # decoder共用encoder的上下文变量
    def forward(self, X, state):
        # 输出'X'的形状： (batch_size,num_steps,embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        # 广播context，使其具有与X相同的num_steps
        context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        # output的形状:(batch_size,num_steps,vocab_size)
        # state的形状:(num_layers,batch_size,num_hiddens)
        return output, state


class Loss(nn.CrossEntropyLoss):
    """The softmax cross-entropy loss with masks.

    Defined in :numref:`sec_seq2seq_decoder`"""
    # `pred` shape: (`batch_size`, `num_steps`, `vocab_size`)
    # `label` shape: (`batch_size`, `num_steps`)
    # `valid_len` shape: (`batch_size`,)
    def forward(self, pred, label):
        weights = torch.ones_like(label)
        self.reduction='none'
        unweighted_loss = super(Loss, self).forward(
            pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss


def train(net, data_iter, lr, num_epochs, device):
    """训练序列到序列模型"""
    plt.rcParams['font.size'] = 30
    plt.rcParams['lines.linewidth'] = 4
    # plt.rcParams['font.family'] = 'serif'
    # plt.rcParams['font.serif'] = ['Times New Roman']

    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])
    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = Loss()
    net.train()
    animator = Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs])

    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
        for batch in data_iter:
            optimizer.zero_grad()
            X, Y = [x.to(device) for x in batch]
            dec_input = X  # Teacher forcing
            Y_hat, _ = net(X, dec_input)
            l = loss(Y_hat, Y)
            l.sum().backward()  # Make the loss scalar for `backward`
            d2l.grad_clipping(net, 1)
            num_tokens = 10
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
        print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
                            f'tokens/sec on {str(device)}')
    plt.savefig('seq2seq.png', dpi=300)


def predict_text(net, offset, pre_steps, num_steps, file):
    net.load_state_dict(torch.load(file))
    net.eval()
    corpus, _ = load_corpus()
    X = corpus[offset: num_steps+offset]
    correct = corpus[offset: num_steps+pre_steps+offset]
    correct = [vocab.idx_to_token[i] for i in correct]
    correct.insert(30, '\n')
    # correct.insert(60, '\n')
    # correct.insert(90, '\n')
    output_seq = [vocab.idx_to_token[i] for i in X]
    print('输入的文本：', ''.join(output_seq), '\n')
    print('正确的文本：', ''.join(correct), '\n')
    X = torch.tensor(X).unsqueeze(0)
    for _ in range(pre_steps):
        Y,_ = net(X, X)
        # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        X = Y.argmax(dim=2)
        temp = X.squeeze(dim=0)
        pred = [vocab.idx_to_token[i] for i in temp]
        output_seq.append(pred[-1])
    output_seq.insert(30, '\n')
    # output_seq.insert(60, '\n')
    # output_seq.insert(90, '\n')
    print('生成的文本：', ''.join(output_seq), '\n')


embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 5
lr, num_epochs, device = 0.005, 500, d2l.try_gpu()
# train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
# batch_size, num_steps = 32, 35
train_iter, vocab = load_data(batch_size, num_steps)
encoder = Seq2SeqEncoder(len(vocab), embed_size, num_hiddens, num_layers,
                        dropout)
decoder = Seq2SeqDecoder(len(vocab), embed_size, num_hiddens, num_layers,
                        dropout)
net = d2l.EncoderDecoder(encoder, decoder)


# train(net, train_iter, lr, num_epochs, device)
# torch.save(net.state_dict(), 'seq2seq_short.pth')
# 取消注释这两行代码进行训练

for i in range(20):
    predict_text(net, i+600, 50,5 , 'seq2seq_short.pth')
# 取消注释这行代码进行测试
