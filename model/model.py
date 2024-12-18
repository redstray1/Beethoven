import math

import torch
from torch import nn
import torch.nn.functional as F
from typing import Type

from torch.nn import init

from dataloader import BEETDataset
from torch.utils.data import DataLoader
from dataloader import ctc_collate

class LipNet(nn.Module):

    def __init__(self, vocab_size, max_length, rnn_type : Type = nn.RNN, 
                 dropout = 0.5, hidden_size=256, rnn_layers=1):
        super(LipNet, self).__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length

        self.conv = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Dropout3d(dropout),

            nn.Conv3d(32, 64, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Dropout3d(dropout),

            nn.Conv3d(64, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Dropout3d(dropout)
        )
        # batch, seq_len, 96 * 3 * 6 -> batch, seq_len, num_layers*hidden_size
        self.rnn1 = rnn_type(input_size=96 * 3 * 6, hidden_size=hidden_size, num_layers=rnn_layers, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout)

        # batch, seq_len, num_layers*hidden_size -> batch, seq_len, num_layers*hidden_size
        self.rnn2 = rnn_type(input_size=hidden_size * 2, hidden_size=hidden_size, num_layers=rnn_layers, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(dropout)

        # batch, seq_len, num_layers*hidden_size -> batch, seq_len, vocab_size
        self.linear1 = nn.Linear(in_features=hidden_size * 2, out_features=self.vocab_size)

        for m in self.conv.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, nonlinearity='relu')
                init.constant_(m.bias, 0)
        
        init.kaiming_normal_(self.linear1.weight, nonlinearity='sigmoid')
        init.constant_(self.linear1.bias, 0)

        for m in (self.rnn1, self.rnn2):
            stdv = math.sqrt(2 / (96 * 3 * 6 + hidden_size))
            for i in range(0, hidden_size * 3, hidden_size):
                init.uniform_(m.weight_ih_l0[i: i + hidden_size],
                              -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                init.orthogonal_(m.weight_hh_l0[i : i + hidden_size])
                init.constant_(m.bias_ih_l0[i : i + hidden_size], 0)
                init.uniform_(m.weight_ih_l0_reverse[i : i + hidden_size],
                              -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                init.orthogonal_(m.weight_hh_l0_reverse[i : i + hidden_size])
                init.constant_(m.bias_ih_l0_reverse[i : i + hidden_size], 0)
        
    def forward(self, x):
        x = self.conv(x) # batch, channels, frames, height, width
        x = x.permute(0, 2, 1, 3, 4).contiguous() # batch, frames, channels, height, width
        x = x.view(x.size(0), x.size(1), -1) # batch, frames, channels * height * width

        x, _ = self.rnn1(x) # batch, frames, 2 * hidden_size
        x = self.dropout1(x) # batch, frames, 2 * hidden_size

        x,_ = self.rnn2(x) # batch, frames, 2 * hidden_size
        x = self.dropout2(x) # batch, frames, 2 * hidden_size

        x = self.linear1(x) # batch, frames, vocab_size
        return x
    
class BEETModel:
    def __init__(self, opt):
        self.trainset = BEETDataset(folders=opt['folders'], train=True, max_frames=opt['max_frames'], sp_model_prefix=opt['sp_model_prefix'])
        self.valset = BEETDataset(folders=opt['folders'], train=False, max_frames=opt['max_frames'], sp_model_prefix=opt['sp_model_prefix'])

        self.trainloader = DataLoader(self.trainset, batch_size=opt['batch_size'], 
                                      shuffle=True, num_workers=opt['num_workers'], 
                                      collate_fn=ctc_collate, pin_memory=True)
        self.valloader = DataLoader(self.valset, batch_size=opt['batch_size'],
                                    shuffle=False, num_workers=opt['num_workers'],
                                    collate_fn=ctc_collate, pin_memory=True)
        
        self.input_img_size = [3, 64, 128]
        self.chan, self.height, self.width = self.input_img_size
        self.vocab_size = self.trainset.sp_model.vocab_size()
        self.maxT = self.trainset.max_frames

        self.model = LipNet(self.vocab_size, max_length=150, rnn_type=nn.LSTM, hidden_size=opt['hidden_size'], dropout=opt['dropout'])
        self.optimfunc = torch.optim.Adam(self.model.parameters(), lr=opt['lr'])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimfunc, gamma=0.9)

    def optim(self, epoch):
        return self.optimfunc
