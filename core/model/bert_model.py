import torch
import torch.nn as nn
import numpy as np
from einops import repeat
import random

from utils.registery import MODEL_REGISTRY

def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]
    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    return torch.FloatTensor(sinusoid_table)
        

class TransformerEncoder(nn.Module):
    def __init__(self, inc, nheads, feedforward_dim, nlayers, dropout):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=inc, 
            nhead=nheads, 
            dim_feedforward=feedforward_dim, 
            dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

    def forward(self, x):
        out = self.transformer_encoder(x)

        return out


def Regressor(inc_dim, out_dim, dims_list=[512, 256], dropout=0.3, act=nn.GELU(), has_tanh=True, has_sigmoid=False):
        module_list = list()
        module_list.append(nn.Linear(inc_dim, dims_list[0]))
        module_list.append(act)
        if dropout != None:
            module_list.append(nn.Dropout(dropout))
        for i in range(len(dims_list) - 1):
            module_list.append(nn.Linear(dims_list[i], dims_list[i + 1]))
            module_list.append(act)
            if dropout != None:
                module_list.append(nn.Dropout(dropout))

        module_list.append(nn.Linear(dims_list[-1], out_dim))
        if has_tanh:
            module_list.append(nn.Tanh())
        if has_sigmoid:
            module_list.append(nn.Sigmoid())
        module = nn.Sequential(*module_list)

        return module


@MODEL_REGISTRY.register()
class BERT(nn.Module):
    def __init__(self, 
                 input_dim, 
                 feedforward_dim, 
                 affine_dim, 
                 nheads, 
                 nlayers, 
                 dropout, 
                 use_pe, 
                 seq_len, 
                 head_dropout, 
                 head_dims, 
                 out_dim, 
                 task):

        super().__init__()

        self.input_dim = input_dim
        inc = input_dim
        self.feedforward_dim = feedforward_dim
        self.affine_dim = affine_dim
        self.task = task
        if self.affine_dim != None:
            self.affine = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, affine_dim),
                nn.ReLU(),
            )
            self.affine = nn.Linear(input_dim, affine_dim)
            inc = affine_dim

        self.use_pe = use_pe
        if use_pe:
            self.pe = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(seq_len, inc), freeze=True)

        self.transformer_encoder = TransformerEncoder(inc=inc, nheads=nheads, feedforward_dim=feedforward_dim, nlayers=nlayers, dropout=dropout)

        if self.task == 'va':
            self.v_head = Regressor(inc_dim=inc, out_dim=out_dim, dims_list=head_dims, dropout=head_dropout, act=nn.ReLU())
            self.a_head = Regressor(inc_dim=inc, out_dim=out_dim, dims_list=head_dims, dropout=head_dropout, act=nn.ReLU())
        elif self.task == 'eri':
            self.head1 = Regressor(inc_dim=inc, out_dim=out_dim, dims_list=head_dims, dropout=head_dropout, act=nn.ReLU())
            self.head2 = Regressor(inc_dim=inc, out_dim=out_dim, dims_list=head_dims, dropout=head_dropout, act=nn.ReLU())
            self.head3 = Regressor(inc_dim=inc, out_dim=out_dim, dims_list=head_dims, dropout=head_dropout, act=nn.ReLU())
            self.head4 = Regressor(inc_dim=inc, out_dim=out_dim, dims_list=head_dims, dropout=head_dropout, act=nn.ReLU())
            self.head5 = Regressor(inc_dim=inc, out_dim=out_dim, dims_list=head_dims, dropout=head_dropout, act=nn.ReLU())
            self.head6 = Regressor(inc_dim=inc, out_dim=out_dim, dims_list=head_dims, dropout=head_dropout, act=nn.ReLU())
            self.head7 = Regressor(inc_dim=inc, out_dim=out_dim, dims_list=head_dims, dropout=head_dropout, act=nn.ReLU())
            # self.fc1 = nn.Linear(seq_len+512, 1)
            # self.fc2 = nn.Linear(seq_len+512, 1)
            # self.fc3 = nn.Linear(seq_len+512, 1)
            # self.fc4 = nn.Linear(seq_len+512, 1)
            # self.fc5 = nn.Linear(seq_len+512, 1)
            # self.fc6 = nn.Linear(seq_len+512, 1)
            # self.fc7 = nn.Linear(seq_len+512, 1)
            self.fc1 = nn.Linear(seq_len, 7)
            # self.fc2 = nn.Linear(seq_len, 1)
            # self.fc3 = nn.Linear(seq_len, 1)
            # self.fc4 = nn.Linear(seq_len, 1)
            # self.fc5 = nn.Linear(seq_len, 1)
            # self.fc6 = nn.Linear(seq_len, 1)
            # self.fc7 = nn.Linear(seq_len, 1)
            self.sigmoid = nn.Sigmoid()
            self.new_fc = nn.Linear(seq_len, 1)
            self.new_fc_1 = nn.Linear(1024, 512)
        else:
            self.head = Regressor(inc_dim=inc, out_dim=out_dim, dims_list=head_dims, dropout=head_dropout, act=nn.ReLU(), has_tanh=False)


    def forward(self, x):
        seq_len, bs, _ = x.shape
        if self.affine_dim != None:
            x = self.affine(x)
        if self.use_pe:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(1).expand([seq_len, bs])
            position_embeddings = self.pe(position_ids)
            x = x + position_embeddings

        out = self.transformer_encoder(x)
        #out = out.permute(2, 1, 0)
        #out = self.new_fc(out)
        #out = out.squeeze(2)
        #print(out.size())
        #print(audio_x.size())
        #out = torch.cat([out, audio_x], dim=0)
        # print(out.size())
        if self.task == 'eri':
            out1 = self.head1(out)
            out2 = self.head2(out)
            out3 = self.head3(out)
            out4 = self.head4(out)
            out5 = self.head5(out)
            out6 = self.head6(out)
            out7 = self.head7(out)

            out1 = torch.squeeze(out1)
            out2 = torch.squeeze(out2)
            out3 = torch.squeeze(out3)
            out4 = torch.squeeze(out4)
            out5 = torch.squeeze(out5)
            out6 = torch.squeeze(out6)
            out7 = torch.squeeze(out7)

            # out1 = torch.cat([out1, audio_x], dim=0)
            # out2 = torch.cat([out2, audio_x], dim=0)
            # out3 = torch.cat([out3, audio_x], dim=0)
            # out4 = torch.cat([out4, audio_x], dim=0)
            # out5 = torch.cat([out5, audio_x], dim=0)
            # out6 = torch.cat([out6, audio_x], dim=0)
            # out7 = torch.cat([out7, audio_x], dim=0)

            out1 = out1.transpose(0, 1)
            out2 = out2.transpose(0, 1)
            out3 = out3.transpose(0, 1)
            out4 = out4.transpose(0, 1)
            out5 = out5.transpose(0, 1)
            out6 = out6.transpose(0, 1)
            out7 = out7.transpose(0, 1)

            out1 = self.fc1(out1)
            # out2 = self.fc2(out2)
            # out3 = self.fc3(out3)
            # out4 = self.fc4(out4)
            # out5 = self.fc5(out5)
            # out6 = self.fc6(out6)
            # out7 = self.fc7(out7)

            # out1 = self.sigmoid(out1)
            # out2 = self.sigmoid(out2)
            # out3 = self.sigmoid(out3)
            # out4 = self.sigmoid(out4)
            # out5 = self.sigmoid(out5)
            # out6 = self.sigmoid(out6)
            # out7 = self.sigmoid(out7)
            out = out1
            out_ass = out1[:, 1] + out1[:, 5]
            # out = torch.cat([out1, out2, out3, out4, out5, out6, out7], dim=-1)
        else:
            out = self.head(out)
       
        return out, out_ass
