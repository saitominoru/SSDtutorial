#!/usr/bin/env python
# coding: utf-8


import torch
import numpy as np
import itertools

#デフォルトボックスを出力するクラス
class DBox(object):
    def __init__(self, cfg):
        super(DBox, self).__init__()
        
        #初期設定
        self.image_size = cfg['input_size'] #画像サイズの300
        #[38, 19, ...]各sourceの特徴量マップのサイズ
        self.feature_maps = cfg['feature_maps']
        self.num_priors = len(cfg['feature_maps']) #sourceの個数=6
        self.steps = cfg['steps'] #[8, 16, ...] DBoxのピクセルサイズ(特徴マップの1つの大きさ)
        self.min_sizes = cfg['min_sizes'] #[30,60,...]小さい正方形のDBoxのピクセルサイズ
        self.max_sizes = cfg['max_sizes'] #[60,111,...]大きい正方形のDBoxのピクセルサイズ
        self.aspect_ratios = cfg['aspect_ratios'] #長方形のDBoxのアスペクト比
        
    def make_dbox_list(self):
        '''DBoxを作成する'''
        mean =[]
        #'feature_maps' : [38,19,10,5,3,1]
        for k, f in enumerate(self.feature_maps):
            for i, j in itertools.product(range(f), repeat=2): #fまでの数で2ペアの組み合わせを作る
                
                #特徴量の画像サイズ
                #300 / 'steps':[8,16,32, 64, 100, 300], 
                f_k = self.image_size / self.steps[k]
                
                #DBoxの中心座標x,y　ただし0~1で規格化している
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k
                
                #アスペクト比1の小さいDBox[cx,cy, width, height]
                #'min_sizes':[30,60,111,162,213,264]
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]
                
                #アスペクト比1の大きいDBox [cx, cy, width, height]
                #'max_size': [60, 111, 162, 213, 264, 315],
                s_k_prime = np.sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]
                
                #その他のアスペクト比のdefBox[cx, cy, width, height]
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*np.sqrt(ar), s_k/np.sqrt(ar)]
                    mean += [cx, cy, s_k/np.sqrt(ar), s_k*np.sqrt(ar)]
                    
        #DBoxをテンソルに変換torch.Size([8732, 4])
        output = torch.Tensor(mean).view(-1, 4)
            
        #DBoxが画像の外にはみ出るのを防ぐため、大きさを最小0　最大1にする
        output.clamp_(max=1, min=0)
            
        return output
            

