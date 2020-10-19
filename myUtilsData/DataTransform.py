#!/usr/bin/env python
# coding: utf-8

#絶対パスによるインポート：https://note.nkmk.me/python-relative-import/
import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.data_augumentation import Compose, ConvertFromInts, ToAbsoluteCoords,PhotometricDistort, Expand, RandomSampleCrop, RandomMirror, ToPercentCoords, Resize, SubtractMeans

class DataTransform():
    """
    画像とアノテーションの前処理クラス。訓練と推論で異なる動作をする。
    画像のサイズを３００x３００にする。
    学習時はデータオーギュメンテーションする。
    
    Attributes
    ----------
    input_size : int
        リサイズ先の画像の大きさ。
    color_mean : (B, G, R)
        各色チャネルの平均値。
    """
    
    def __init__(self, input_size, color_mean):
        self.data_transform = {
            'train': Compose([
                ConvertFromInts(),         #intをfloat32に変換
                ToAbsoluteCoords(),        # アノテーションデータの規格化を戻す
                PhotometricDistort(),      #画像の色調などをランダムに変化
                Expand(color_mean),        #画像のキャンバスを広げる
                RandomSampleCrop(),        #画像内の部分をランダムに抜き出す
                RandomMirror(),            #画像を反転させる
                ToPercentCoords(),         #アノテーションデータを0-1に正規化
                Resize(input_size),        #画像サイズをinput_size*input_sizeに変形
                SubtractMeans(color_mean)  #BGRの色の平均値を引き算  
            ]), 
            
            'val': Compose([
                ConvertFromInts(),        #intをfloat32に変換
                Resize(input_size),       #画像サイズをinput_size * input_sizeに変形
                SubtractMeans(color_mean) #BGRの色の平均値を引き算
                
            ])
        }
    def __call__(self, img, phase, boxes, labels):
        """
        Parameters
        ----------
        phases :   'train' or 'val'
            前処理のモードを指定。
        """
        return self.data_transform[phase](img, boxes, labels)