#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn

#convC4_3からの出力をscale=20のL2Normで正規化する層
class L2Norm(nn.Module):
    def __init__(self, input_channels=512, scale=20):
        super(L2Norm, self).__init__() #親クラスのコンストラクタを実行
        self.weight = nn.Parameter(torch.Tensor(input_channels))
        self.scale = scale      #係数weightの初期値として設定する値
        self.reset_parameters() #パラメータの初期化
        self.eps = 1e-10
        
    def reset_parameters(self):
        '''結合パラメータの大きさscaleに値する初期化を実行'''
        nn.init.constant_(self.weight, self.scale) #weightの値がすべてscaleになる
        
    def forward(self, x):
        '''38*38の特徴量に対して、512チャネルにわたって２乗和のルートを求めた
        38*38個の値を使用し、各特徴量を正規化してから係数を掛け算する層'''
        
        #各チャネルにおける38*38個の特徴量のチャネル方向の二乗和を計算し、
        #さらにルートを求め、割り算して正規化する
        #normのテンソルサイズはtorch.Size([batch_num, 1, 38, 38])になる
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        
        #係数をかける。係数はチャネルごとに一つで、512個の係数を持つ
        #self.weightのテンソルサイズはtorch.Size([512])なので
        #torch.Size([batch_num, 512, 38, 38])まで変形する
        weights = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)
        out = weights * x
        
        return out