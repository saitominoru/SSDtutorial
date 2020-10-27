#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#オフセット情報を使い、DBoxをBBoxに変換する関数
import torch
def decode(loc, dbox_list):
    """
    オフセット情報を使い、DBoxをBBoxに変換する。
    
    Parameters
    -----------
    loc: [8732, 4]
        SSDモデルで推論するオフセット情報
    dbox_list: [8732, 4]
        DBoxの情報
        
    
    Returns
    -------
    boxes : [xmin, ymin, xmax, ymax]
       BBoxの情報
    """
    
    #DBoxは[cx, cy, width, height]で格納されている
    #locも[Δcx, Δcy, Δwidth, Δheight]で格納されている
    
    #オフセット情報からBBoxを求める
    boxes = torch.cat((
        dbox_list[:, :2] + loc[:, :2] * 0.1 * dbox_list[:, 2:],
        dbox[:, 2:] * torch.exp(loc[:, 2:] * 0.2)), dim = 1)
    #boxesのサイズはtorch.Size([8732, 4])になる
    
    #BBoxの座標情報を[cx, cy, width, height]から[xmin, ymin, xmax, ymax]に変換
    boxes[:, :2] -= boxes[:, 2:] / 2   #座標(xmin, ymin)へ変換
    boxes[:, 2:] += boxes[:, :2]       #座標(xmax, ymax)へ変換
    
    return boxes

