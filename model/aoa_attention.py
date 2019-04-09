import sys
from os import path
import torch.nn as nn
import torch
import torch.nn.functional as F


class AOA_Attention(nn.Module):
    def __init__(self, gpu=True):
        super(AOA_Attention, self).__init__()
        self.use_gpu = gpu

    def sequence_mask(self, input_weight, seq_lengths, target_lengths):

        def get_softmax_att(tmp_input_weight, tgt_len, batch_size):
            input_weight = tmp_input_weight.contiguous()
            mask_ = torch.arange(1, tgt_len+1).type(torch.LongTensor).repeat(batch_size, 1).cuda()  # batch, seq_len
            mask_ = mask_.le(target_lengths.unsqueeze(1)).unsqueeze(1)
            input_weight.data.masked_fill_(1-mask_, -10000)
            align_weight = F.softmax((input_weight.view(-1, tgt_len)), dim=1).view(batch_size, -1, tgt_len)
            return align_weight
        
        batch_size = input_weight.size(0); tgt_len = input_weight.size(2)

        align_weight_src_tgt = get_softmax_att(input_weight, tgt_len, batch_size)  # batch, src_len, tgt_len

        return align_weight_src_tgt

    
    def forward(self, src, src_length, tgt, tgt_length):
        # 论文里面这个地方没有加tanh
        input_weight = torch.bmm(src, tgt.transpose(1, 2)).contiguous()  # batch, src_len, tgt_len

        align_weight_src_tgt = self.sequence_mask(input_weight, src_length, tgt_length)  # batch, src_len, tgt_len

        align_hidden = torch.bmm(align_weight_src_tgt, tgt)  # batch, src_len, hidden
        
        return  align_hidden
