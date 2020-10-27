#!/usr/bin/env python
# coding: utf-8


#SSDのクラスを作る
import torch.nn as nn
from .make_vgg import make_vgg
from .make_extras import make_extras
from .L2Norm import L2Norm
from .DBox import DBox
from .make_loc_conf import make_loc_conf

class SSD(nn.Module):
    
    def __init__(self, phase, cfg):
        super(SSD, self).__init__()
        
        self.phase = phase # train or inference
        self.num_classes = cfg["num_classes"] #クラス数21
        
        #SSDのネットワークを作る
        self.vgg = make_vgg()
        self.extras = make_extras()
        self.L2Norm = L2Norm()
        self.loc, self.conf = make_loc_conf(
                    cfg["num_classes"], cfg["bbox_aspect_num"])
        
        #DBoxの作成
        dbox = DBox(cfg)
        self.dbox_list = dbox.make_dbox_list()
        
        #推論時はクラス「Detect」を用意する
        if phase == 'inference':
            self.detect = Detect()
