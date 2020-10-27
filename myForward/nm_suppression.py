#!/usr/bin/env python
# coding: utf-8


#Non-Maximum Suppressionを行う関数
import torch

def nm_suppression(boxes, scores, overlap=0.45, top_k=200):
    """
    Non-Maximum Suppressionを行う関数
    boxesのうち被り過ぎ(overlap以上)のBBoxを削除する。
    
    Parameters
    ----------
    boxes : [確信度閾値(0.01)を超えたBBoxの数,4]
         BBoxの情報。
    scores : [確信度閾値(0.01)を超えたBBoxの数]
         confの情報
         
    Returns
    -------
    keep : リスト
        confの降順にNon-Maximum Suppression(nms)を通過したindexが格納
    count : int
        nmsを通過したBBoxの数
    """
    
    #returnの雛形を作成
    count = 0
    keep = scores.new(scores.size(0)).zero_().long()
    #keep : torch.Size([確信度閾値を超えたBBoxの数]), 要素は全部0
    
    #各BBoxの面積areaを計算
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2-x1, y2-y1)
    
    #boxesをコピーする。後でBBoxの被り度合いIoUの計算に使用する際の雛形として用意
    tmp_x1 = boxes.new()
    tmp_y1 = boxes.new()
    tmp_x2 = boxes.new()
    tmp_y2 = boxes.new()
    tmp_w = boxes.new()
    tmp_h = boxes.new()
    
    #scoresを昇順に並び替える v:Tensor, idx, LongTensor
    v, idx = scores.sort(0) #vには昇順に並べられたあとのscoreが格納され、 
                            #idxには昇順に並ぶ前にscoresで何番目に存在したのかを返す
    
    
    #上位top_k個(200個)のBBoxのindexを取り出す(200個存在しない場合もある)
    idx = idx[-top_k:] #[-k:]で、後ろからk個取り出すという意味.小さい方が先頭
    
    #idxの要素が0でない限りループする
    while idx.numel() > 0:  #numel = テンソルの要素数を返す
        i = idx[-1] #現在のconf最大のindexをiに。
                    #[-1]でidxの一番最後の要素＝最大の確信度の要素を得る
        
        #keepの現在の最後にconf最大のindexを格納する
        #このindexのBBoxと被りが大きいBBoxをこれから削除する
        keep[count] = i
        count+=1
        
        #最後のBBoxになった場合はループを抜ける
        if idx.size(0) == 1:
            break
        
        #現在のconf最大のindexをkeepに格納したので、idxをひとつ減らす
        idx = idx[:-1]
        
        #---------
        #これからkeepに格納したBBoxと被りの大きいBBoxを抽出して除去する
        #----------
        #1つ減らしたidxまでのBBoxを、outに指定した変数として作成する
        torch.index_select(x1, 0, idx, out = tmp_x1)
        torch.index_select(y1, 0, idx, out = tmp_y1)
        torch.index_select(x2, 0, idx, out = tmp_x2)
        torch.index_select(y2, 0, idx, out = tmp_y2)
        #index_select(input, dim, index, out)
        #       -> inputの配列から、dim方向に向かって、
        #          idxで指定した順番に要素を取り出してoutで指定したリストに格納
        
        #すべてのBBoxに対して、現在のBBox=indexがiと被っている値までに設定(clamp)
        tmp_x1 = torch.clamp(tmp_x1, min=x1[i])
        tmp_y1 = torch.clamp(tmp_y1, min=y1[i])
        tmp_x2 = torch.clamp(tmp_x2, min=x2[i])
        tmp_y2 = torch.clamp(tmp_y2, min=y2[i])
        #clampとは、min, maxで指定した値からはみ出ている値をmin.maxにし、
        #            その他の値はそのままにする関数
        
        #wとhのテンソルサイズをindexを1減らしたものにする。
        tmp_w.resize_as_(tmp_x2)
        tmp_h.resize_as_(tmp_y2)
        
        
        #clampした状態でのBBoxの幅と高さを求める
        tmp_w = tmp_x2 - tmp_x1
        tmp_h = tmp_y2 - tmp_y1
        
        #幅や高さが負になっているものは0にする
        tmp_w = torch.clamp(tmp_w, min=0.0)
        tmp_h = torch.clamp(tmp_h, min=0.0)
        
        #clampされた状態での面積を求める
        inter = temp_w*tmp_h
        
        #IoU = intersect部分/ (area(a) + area(b) - intersect部分)の計算
        rem_areas = torch.index_select(area, 0, idx) #各BBoxの元の面積
        union = (rem_areas - inter) + area[i]
        IoU = inter/ union
        
        #IoUがoverlapより小さいidxのみを残す
        idx = idx[IoU.le(overlap)] #leはless than or equal to の処理をする演算
        #IoUがoverlapより大きいidxは、最初に選んでkeepに格納したidxと同じ物体に対して
        #BBoxを囲んでいるため消去
        
    #whileループを抜けたら終了
    
    return keep, count

