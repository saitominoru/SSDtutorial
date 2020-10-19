#!/usr/bin/env python
# coding: utf-8


import xml.etree.cElementTree as ET
import numpy as np
import cv2

#「XML形式のアノテーション」を、リスト形式に変換するクラス
class Anno_xml2list(object):
    """
    １枚の画像に対する「XML形式のアノテーションデータ」を、画像サイズで規格化してから
    リストに変換する
    
    Attributes
    ----------
    classes : リスト
        VOCのクラス名を格納したリスト
    """
    
    def __init__(self, classes):
        
        self.classes = classes
        
    def __call__(self, xml_path, width, height):
        """
        １枚の画像に対する「XML形式のアノテーションデータ」を、画像サイズで規格化して
        からリスト形式に変換する
        
        Parameters
        ----------
        xml_path : str
            xmlファイルへのパス
        width    : int
            対象画像の幅
        height   : int
            対象画像の高さ
        
        Returns
        -------
        ret : [[xmin, ymin, xmax, ymax, label_ind], ... ]
            物体のアノテーションデータを格納したリスト。画像内に存在する物体数の分だけ
            要素を持つ。
        """
        
        #画像内のすべての物体のアノテーションをこのリストに格納
        ret = []
        
        #xmlファイルを読み込む
        xml = ET.parse(xml_path).getroot()
        
        #画像内にある物体('object')の数だけループする
        for obj in xml.iter('object'):
            
            #アノテーションで検知がdifficultに設定されているものは除外
            difficult = int(obj.find('difficult').text)
            if difficult == 1:
                continue
                
            #１つの物体に対するアノテーションが格納されているリスト
            bndbox = []
            
            name = obj.find('name').text.lower().strip() #物体名
            print(name)
            bbox = obj.find('bndbox')  #バウンディングボックスの情報
            
            #アノテーションのxmin,ymin,xmax,ymaxを取得して0~1に正規化
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            
            for pt in (pts):
                #VOCは原点が(1,1)なので１を引き算して(0,0)に
                cur_pixel = int(bbox.find(pt).text) - 1
                
                #幅、高さで規格化
                if pt=='xmin' or pt=='xmax':  #x方向の時は幅で割り算
                    cur_pixel /= width
                else: #y方向の時は高さで割り算
                    cur_pixel /= height
                    
                bndbox.append(cur_pixel)
                
            #アノテーションのクラス名のインデックスを取得して追加
            label_idx = self.classes.index(name)
            bndbox.append(label_idx)
            
            #retに[xmin,ymin.xmax,ymax,label_ind]を足す
            ret += [bndbox]
            
        return np.array(ret)