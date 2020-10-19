#!/usr/bin/env python
# coding: utf-8

import torch

def od_collate_fn(batch):
    """
    Datasetから取り出すアノテーションデータのサイズが画像ごとに異なる。
    画像内の物体数が２個であれば、(2,5)というサイズになるが、３個であれば(3,5)というサイズになる。
    この変化に対応するDataloaderを作成するためにカスタマイズしたcollate_fnを作成する。
    collate_fnはPytorchでリストからmini-batchを作成する関数である。
    ミニバッチ分の画像が並んでいるリスト変数batchにミニバッチ番号を指定する次元を先頭に１つ追加してリストの形を変形する。
    """
    
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0]) #sample[0]は画像img
        targets.append(torch.FloatTensor(sample[1])) #sample[1]はアノテーションgt
        
        
    #imgsはミニバッチサイズのリストになっている
    #リストの要素はtorch.Size([3,300,300])
    #このリストをtorch.Size([batch_num, 3, 300, 300])のテンソルに変換する
    imgs = torch.stack(imgs, dim=0)
    
    #*targetsはアノテーションデータの正解であるgtのリスト
    #*リストのサイズはミニバッチサイズ
    #*リストtargetsの要素は[n, 5]
    #*nは画像ごとに異なり、画像内にあるオブジェクトの数になる。
    #*5は[xmin,ymin,xmax,ymax,class,index]
    return imgs, targets