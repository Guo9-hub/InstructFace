#coding:utf-8
import pandas as pd
from datasets.RAFDB import InstructRAFDB
import os
from PIL import Image
import torch
import numpy as np

# AffectNet 的 8 类表情
EXPRESSION_NAMES = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class InstructAffectNet(InstructRAFDB):
    def __init__(self, csv_file, mode, img_dir, transform, instruct_json_path, ablation='full'):
        self.ablation = ablation
        self.mode = mode
        self.img_dir = img_dir
        self.transform = transform
        self.imgs = []
        self.labels = []
        
        # 读取 CSV
        # 假设 csv 表头是: subDirectory_filePath, expression
        df = pd.read_csv(csv_file)
        for index, row in df.iterrows():
            expr = int(row['expression'])
            # 过滤掉非基础表情 (AffectNet 有时会有第8,9类)
            if expr > 7: continue 
            
            self.imgs.append(row['subDirectory_filePath'])
            self.labels.append(expr)
            
        print(f"[{mode.upper()}] AffectNet 加载完成: {len(self.imgs)} 张图片")
        
        # 指令相关逻辑可以复用 RAFDB 的，或者在这里重写
        self.instruct_dict = {} # 如果暂时没有指令文件，就留空