# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.aoa_attention import AOA_Attention
import numpy as np
from model.cnn import textCNN
from utils import constant
import time

class Cla(nn.Module):
    def __init__(self, args):
        super(Cla, self).__init__()
        self.embedding_size = args.embedding_size
        self.hidden_size = args.hidden_size
        self.bidirectional = args.bidirectional
        self.vocab_size = args.vocab_size
        self.input_dropout = nn.Dropout(args.input_dropout)
        self.output_dropout = args.output_dropout
        self.cato_num = args.cato_num
        self.layer_num = args.layer_num

        if args.rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=args.pad_id)
        if args.pre_embedding:
            self.embedding.weight = nn.Parameter(torch.from_numpy(args.embedding))
        self.embedding.requires_grad = args.update_embedding

        # cnn的参数
        self.cnn_args = {}
        self.cnn_args['filter_sizes'] = [3]
        self.cnn_args['filter_num'] = 50
        self.cnn_args['embedding_size'] = self.hidden_size * 2
        self.cnn_args['cato_num'] = self.cato_num
        self.cnn_args['dropout'] = self.output_dropout
        self.cnn = textCNN(self.cnn_args)  # 会输出最后的概率
    
        # 两个lstm计算attention
        self.rnn_src = self.rnn_cell(self.embedding_size, self.hidden_size, self.layer_num, batch_first=True,
                                 bidirectional=self.bidirectional)
        self.rnn_tgt = self.rnn_cell(self.embedding_size, self.hidden_size, self.layer_num, batch_first=True,
                                 bidirectional=self.bidirectional)
        
        # 将第二个lstm的h映射成概率
        self.s_t_w = nn.Linear(self.hidden_size*4, self.hidden_size*2)
        self.control_w = nn.Linear(self.hidden_size*2, 1)

        self.initial('lstm')
        self.initial('linear')
        self.initial('cnn')

        self.C = 40.0
        self.aoa = AOA_Attention()

    def initial(self, cato):
        if cato == 'lstm':
            torch.nn.init.uniform_(self.rnn_src.weight_ih_l0, a=-0.01, b=0.01)
            torch.nn.init.uniform_(self.rnn_src.weight_hh_l0, a=-0.01, b=0.01)
            torch.nn.init.constant_(self.rnn_src.bias_ih_l0, 0.0)
            torch.nn.init.constant_(self.rnn_src.bias_hh_l0, 0.0)

            torch.nn.init.uniform_(self.rnn_src.weight_ih_l0_reverse, a=-0.01, b=0.01)
            torch.nn.init.uniform_(self.rnn_src.weight_hh_l0_reverse, a=-0.01, b=0.01)
            torch.nn.init.constant_(self.rnn_src.bias_ih_l0_reverse, 0.0)
            torch.nn.init.constant_(self.rnn_src.bias_hh_l0_reverse, 0.0)

            torch.nn.init.uniform_(self.rnn_tgt.weight_ih_l0, a=-0.01, b=0.01)
            torch.nn.init.uniform_(self.rnn_tgt.weight_hh_l0, a=-0.01, b=0.01)
            torch.nn.init.constant_(self.rnn_tgt.bias_ih_l0, 0.0)
            torch.nn.init.constant_(self.rnn_tgt.bias_hh_l0, 0.0)

            torch.nn.init.uniform_(self.rnn_tgt.weight_ih_l0_reverse, a=-0.01, b=0.01)
            torch.nn.init.uniform_(self.rnn_tgt.weight_hh_l0_reverse, a=-0.01, b=0.01)
            torch.nn.init.constant_(self.rnn_tgt.bias_ih_l0_reverse, 0.0)
            torch.nn.init.constant_(self.rnn_tgt.bias_hh_l0_reverse, 0.0)
        elif cato == 'linear':
            torch.nn.init.uniform_(self.s_t_w.weight, a=-0.01, b=0.01)
            torch.nn.init.constant_(self.s_t_w.bias, 0.0)
            torch.nn.init.uniform_(self.control_w.weight, a=-0.01, b=0.01)
            torch.nn.init.constant_(self.control_w.bias, 0.0)
        elif cato == 'cnn':
            for i in range(len(self.cnn_args['filter_sizes'])):
                torch.nn.init.uniform_(self.cnn.convs[i].weight, a=-0.01, b=0.01)
                torch.nn.init.constant_(self.cnn.convs[i].bias, 0.0)


    def lstm_process(self, inputs, input_lengths, model, kind):
        if kind == 'tgt':
            inputs = inputs.data.cpu().numpy()
            input_lengths = input_lengths.data.cpu().numpy()

            sort_idx = np.argsort(-input_lengths)
            inputs = inputs[sort_idx]
            input_lengths = input_lengths[sort_idx]
            unsort_idx = np.argsort(sort_idx)

            inputs = torch.from_numpy(inputs)
            input_lengths = torch.from_numpy(input_lengths)

            inputs = nn.utils.rnn.pack_padded_sequence(inputs.cuda(), input_lengths.cuda(), batch_first=True)
            output, _ = model(inputs)
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

            output = output[unsort_idx]
            input_lengths = input_lengths[unsort_idx]

        elif kind == 'src':
            inputs = nn.utils.rnn.pack_padded_sequence(inputs, input_lengths, batch_first=True)
            output, _ = model(inputs)
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        return output


    def forward(self, src, src_length, tgt, tgt_length, tgt_beg_end, dis):
        batch_size = src.size(0); seq_len = src.size(1)

        # 得到距离的表示
        dis = dis.type(torch.FloatTensor).cuda()
        dis = dis / self.C
        dis = 1 - dis
        mask_ = torch.arange(1, dis.size(1)+1).type(torch.LongTensor).repeat(dis.size(0), 1).cuda()
        mask_ = mask_.le(src_length.unsqueeze(1))
        dis.data.masked_fill_(1-mask_, 0.0)
        dis = dis.view(batch_size, seq_len, 1)
        
        # 得到embedding
        src = self.embedding(src); src = self.input_dropout(src)
        tgt = self.embedding(tgt); tgt = self.input_dropout(tgt)

        # 第一层lstm
        src_output = self.lstm_process(src, src_length, self.rnn_src, 'src')
        tgt_output = self.lstm_process(tgt, tgt_length, self.rnn_tgt, 'tgt')

        # 第一层cpt
        tmps = self.aoa(src_output, src_length, tgt_output, tgt_length)
        new_src_output = F.sigmoid(self.s_t_w(torch.cat([src_output, tmps], -1)))  # 这里是不是用的tanh文章里没有说
        a = F.sigmoid(self.control_w(src_output))
        src_output = (1-a) * src_output + a * new_src_output
        # src_output = src_output + new_src_output
        src_output = src_output * dis

        # 第二层cpt
        tmps = self.aoa(src_output, src_length, tgt_output, tgt_length)
        new_src_output = F.sigmoid(self.s_t_w(torch.cat([src_output, tmps], -1)))  # 这里是不是用的tanh文章里没有说
        a = F.sigmoid(self.control_w(src_output))
        src_output = (1-a) * src_output + a * new_src_output
        # src_output = src_output + new_src_output
        src_output = src_output * dis

        # 用CNN分类
        out = self.cnn(src_output)

        return out
