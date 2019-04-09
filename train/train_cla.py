# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
import sys
sys.path.append('..')
import config.config_cla as args
import pickle
from model.cla import Cla
import numpy as np
from embeddings import GloveEmbedding
from sklearn.metrics import classification_report, precision_recall_fscore_support
import spacy
import time

srcF = Field(sequential=True, batch_first=True, lower=True, include_lengths=True)
tgtF = Field(sequential=True, batch_first=True, lower=True, include_lengths=True)
tgt_beF = Field(sequential=True, batch_first=True, include_lengths=True, use_vocab=True)
disF = Field(sequential=True, batch_first=True, include_lengths=True, use_vocab=True)
labelF = Field(sequential=False, batch_first=True, use_vocab=False)


print('load data...')
all_data = TabularDataset(path=args.all_data, format='tsv', fields=[('src', srcF), ('tgt', tgtF), ('tgt_be', tgt_beF),
                                                                    ('dis', disF),
                                                                    ('label', labelF)])
train = TabularDataset(path=args.train_data, format='tsv', fields=[('src', srcF), ('tgt', tgtF), ('tgt_be', tgt_beF),
                                                                   ('dis', disF),
                                                                   ('label', labelF)])
dev = TabularDataset(path=args.dev_data, format='tsv', fields=[('src', srcF), ('tgt', tgtF), ('tgt_be', tgt_beF),
                                                               ('dis', disF),
                                                               ('label', labelF)])

tgt_beF.build_vocab(all_data, min_freq=1)
disF.build_vocab(all_data, min_freq=1)
srcF.build_vocab(all_data, min_freq=1)
vocab = srcF.vocab
tgtF.vocab = vocab
args.vocab_size = len(vocab)


g = GloveEmbedding('common_crawl_840', d_emb=300)
embedding = []
for i in range(len(vocab)):
    if not g.emb(vocab.itos[i])[0]:
        embedding.append(np.random.uniform(-0.25, 0.25, size=(1, 300))[0])
    else:
        embedding.append(np.array(g.emb(vocab.itos[i])))
embedding = np.array(embedding, dtype=np.float32)
args.pre_embedding = True
args.embedding = embedding
args.update_embedding = False


print('build batch iterator...')
train_batch_iterator = BucketIterator(
    dataset=train, batch_size=args.batch_size,
    sort=False, sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    repeat=False
)
dev_batch_iteraor = BucketIterator(
    dataset=dev, batch_size=args.batch_size,
    sort=False, sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    repeat=False
)


cla = Cla(args)
optimizer = torch.optim.Adam(cla.parameters(), lr=args.lr)
loss_func = nn.CrossEntropyLoss()
cla.cuda()
loss_func.cuda()
for name, p in cla.named_parameters():
    print(name)



def get_origin(var, var_length, vocab, is_num):
    seq_len = var.size(1)  # pad之后的长度
    batch_size = var.size(0)
    str_tensor = [['' for i in range(seq_len)] for j in range(batch_size)]
    for i in range(batch_size):
        for j in range(var_length[i]):
            if is_num:
                var[i][j] = int(vocab.itos[var[i][j]])
            else:
                str_tensor[i][j] = vocab.itos[var[i][j]]
    
    if is_num:
        return var.type(torch.LongTensor)
    else:
        return str_tensor



def run(batch_generator, mode, model):
    if mode == 'train':
        model.train()
    else:
        model.eval()
    
    # 计算分类报告的
    y_true, y_pred = [], []
    target_names = ['neutral', 'negative', 'positive']

    # 计算总体分类准确率的
    correct_pred = 0
    all_pred = 0

    # 用来衰减lstm的lr
    last_loss_data = float('inf')
    loss_drop_counter = 0

    # 用来保存中间结果的
    error = []; right = []

    for batch in batch_generator:
        # 读batch
        srcs, src_lengths = getattr(batch, 'src')
        tgts, tgt_lengths = getattr(batch, 'tgt')
        diss, dis_lengths = getattr(batch, 'dis')
        tgt_bes, tgt_be_lengths = getattr(batch, 'tgt_be')
        labels = getattr(batch, 'label')

        # 将tgt_be, pos, deprel和head还原
        tgt_bes = get_origin(tgt_bes, tgt_be_lengths, tgt_beF.vocab, True)
        diss = get_origin(diss, dis_lengths, disF.vocab, True)

        # 更新
        pred = model(srcs.cuda(), src_lengths.cuda(), tgts.cuda(), tgt_lengths.cuda(), tgt_bes.cuda(), diss.cuda() )
        if mode == 'train':
            loss = loss_func(pred.cuda(), labels.cuda() )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        # 计算分类结果
        pred = pred.max(1)[1]
        y_true.extend(labels.data.numpy().tolist())
        y_pred.extend(pred.cpu().data.numpy().tolist())

        correct_pred += pred.cpu().long().eq(labels.data.long()).sum()
        all_pred += len(srcs)
        
        # # 衰减lstm和其它一些的lr
        # if mode == 'train':
        #     loss_data = loss.item()
        #     if loss_data >= last_loss_data:
        #         loss_drop_counter += 1
        #         if loss_drop_counter >= 3:
        #             for param_group in optimizer.param_groups:
        #                 if param_group['name']!='gcn':
        #                     param_group['lr'] = param_group['lr'] * 0.5
        #                     loss_drop_counter = 0
        #     else:
        #         loss_drop_counter = 0
        #     last_loss_data = loss_data


    # 在屏幕上打印本轮分类结果
    acc = str(round(100.*correct_pred.data.numpy()/all_pred, 2))
    print(mode)
    print('acc: ', acc )
    print(classification_report(y_true, y_pred, target_names=target_names))
    print(precision_recall_fscore_support(y_true, y_pred, average='macro'))

    return acc, cla



print('开始训练 ...')

acc_on_dev = []

for epoch in range(1, args.epochs+1):

    print('第', str(epoch), '轮训练')

    batch_generator = train_batch_iterator.__iter__()
    acc, cla = run(batch_generator, 'train', cla)

    batch_generator = dev_batch_iteraor.__iter__()
    acc, cla = run(batch_generator, 'dev', cla)
    
    acc_on_dev.append(str(acc))

with open('acc_records.txt', 'a') as f:
    f.write(' '.join(acc_on_dev)+'\n')
