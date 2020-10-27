#!/usr/bin/env python
# coding: utf-8

# In[27]:


#SSDの推論時にconfとlocの出力から、被りを消去したBBoxを出力する。
import torch.nn as nn
import torch
from decode import decode
from nm_suppression import nm_suppression
class Detect(torch.autograd.Function):
    
    def __init__(self, conf_thresh=0.01, top_k=200, nms_thresh=0.45):
        self.softmax = nn.Softmax(dim=-1) #confをソフトマックス関数で正規化するために用意
        self.conf_thresh= conf_thresh #confが閾値0.01よりも大きいDBoxのみを扱う。
        self.top_k = top_k #nms_suppressionでconfの高いtop_k個を計算に使用する。
        self.nms_thresh = nms_thresh #nm_suppressionでIoUがnms_thresh=0.45より大きいと、
                                     #同一の物体へのBBoxとみなす。
            
    def forward(self, loc_data, conf_data, dbox_list):
        """
        順伝搬の計算を実行する。
        
        Parameters
        ----------
        loc_data: [batch_num, 8732, 4]
               オフセット情報
        conf_data: [batch_num, 8732, num_classes]
               検出の確信度
        dbox_list: [8732, 4]
               DBoxの情報
        
        Returns
        -------
        output :torch.Size([batch_num, 21, top_k, 5])
                (batch_num, クラス, confのtop_k, DBoxの情報)
        """
        
        #各サイズを取得
        num_batch = loc_data.size(0) #ミニバッチのサイズ
        num_box = loc_data.size(1)    #DBoxの数
        num_classes = conf_data.size(2) #クラス数 = 21
        
        #confはソフトマックスを適用して正規化する
        conf_data = self.softmax(conf_data)
        
        #出力の型を作成する。テンソルサイズは[minibatch数, 21, 200, 5]
        output = torch.zeros(num_batch, num_classes, self.top_k, 5)
        
        #conf_dataを[batch_num, 8732, num_classes]から
        #           [batch_num, num_classes, 732]に順番変更
        conf_preds = conf_data.transpose(2, 1)
        
        #ミニバッチごとのループ
        for i in range(num_batch):
            #1. locとDBoxから修正したBBox[xmin, ymin, xmax, ymax]を求める
            decoded_boxes = decode(loc_data[i], dbox_list)
            
            #confのコピーを作成
            conf_scores = conf_preds[i].clone()
            
            #画像クラスごとのループ(背景クラスのindexである0は計算せず、index=1から)
            for cl in range(1, num_classes):
                
                #2.confの閾値を超えたBBoxを取りだす。
                #confの閾値を超えたconfのインデックスをc_maskとして取得
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                #gtはGreater thanのこと。gtにより閾値を超えたものが1に、以下が0になる。
                #conf_scores: torch.Size([21, 8732])
                #c_mask: torch.Size([8732])
                
                #scoresはtorch.Size([閾値を超えたBBoxの数])
                scores = conf_scores[cl][c_mask]
                
                #閾値を超えたconfがない場合、つまりscores=[]のときは何もしない
                if scores.nelment() == 0:
                    continue
                    
                #c_maskを、decoded_boxesに適用できるようにサイズを変更する。
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                #l_mask : torch.Size([8732, 4])
                
                #l_maskをdecoded_boxesに適応する。
                boxes = decoded_boxes[l_mask].view(-1, 4)
                #deocded_boxes[l_mask]で1次元になってしまうので、
                #viewで(閾値を超えたBBox, 4)サイズに変形する
                
                #3. Non-Maximum Suppressionを実施し、被っているBBoxを取り除く
                ids, count = mm_suppression(
                     boxes, scores, self.nms_thresh, self.top_k)
                #ids : confの降順にNon-Maximum Suppressionを通過したindexが格納
                #count : Non-Maximum Suppressionを通過したBBoxの数
                
                #outputにNon-Maximum Suppressionを抜けた結果を格納
                output[i , cl,  :count] = torch.cat((scores[ids[:count]].unsqueeze(1), 
                                                     boxes[ids[:count]]), 1)
                
        
        return output #torch.Size([1, 21, 200, 5])          

