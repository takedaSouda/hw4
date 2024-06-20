import math
import torch
from torch import nn
import os
import jieba
import lzb_model as d2l
from de_Animator import Animator
from matplotlib import pyplot as plt


'''基于位置的前馈网络'''
PositionWiseFFN = d2l.PositionWiseFFN

'''残差连接，层规范化，暂退法'''
AddNorm = d2l.AddNorm


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


class TransformerEncoder(d2l.Encoder):
    """Transformer encoder.

    Defined in :numref:`sec_transformer`"""
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        # 独热向量编码
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                d2l.EncoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, use_bias))
            # 调整每个注意力块的参数

    def forward(self, X, valid_lens=None, *args):
        # Since positional encoding values are between -1 and 1, the embedding
        # values are multiplied by the square root of the embedding dimension
        # to rescale before they are summed up
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))

        self.attention_weights = [None] * len(self.blks)
        #记录各头注意力权重的值

        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X
        #由源进入解码器输出的X作为键和值进入解码器结构，目标经过注意力模块作为查询和键值结合


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


class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
    num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
    num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block" + str(i),
                                 d2l.DecoderBlock(key_size, query_size, value_size, num_hiddens,
                                              norm_shape, ffn_num_input, ffn_num_hiddens,
                                              num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens=None, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))

        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # 解码器自注意力权重
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            # “编码器－解码器”自注意力权重
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights


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
    plt.savefig('transformer_1.png', dpi=300)


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


num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 20
lr, num_epochs, device = 0.005, 500, d2l.try_gpu()
ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
key_size, query_size, value_size = 32, 32, 32
norm_shape = [32]


train_iter, vocab = load_data(batch_size, num_steps)

encoder = TransformerEncoder(
            len(vocab), key_size, query_size, value_size, num_hiddens,
            norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
            num_layers, dropout)
decoder = TransformerDecoder(
            len(vocab), key_size, query_size, value_size, num_hiddens,
            norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
            num_layers, dropout)


net = d2l.EncoderDecoder(encoder, decoder)


# train(net, train_iter, lr, num_epochs, device)
# torch.save(net.state_dict(), 'transformer.pth')
# 取消注释这两行代码进行训练

for i in range(20):
    predict_text(net, i+900, 40, 20, 'transformer.pth')
# 取消注释这行代码进行测试

