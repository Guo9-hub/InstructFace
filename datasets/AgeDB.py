# coding:utf-8
import os
import torch
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import json
import random

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class AgeDB(data.Dataset):
    def __init__(self, root=None, phase=None, transform=None, instruct_json_path=None, ablation='full', label_path=None, mode=None, img_dir=None):
        self.ablation = ablation
        self.transform = transform
        
        self.mode = mode if mode is not None else phase
        if self.mode is None: self.mode = 'train'
        
        self.img_dir = img_dir if img_dir else root
        if not self.img_dir: 
            raise ValueError("必须提供图片路径 (img_dir 或 root)")

        print(f"[{self.mode.upper()}] 正在扫描真正的官方 AgeDB 目录: {self.img_dir} ...")
        all_image_paths = []
        for root_dir, dirs, files in os.walk(self.img_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_image_paths.append(os.path.join(root_dir, file))
        
        all_image_paths.sort()
        
        self.data_list = []
        
       
        for i, path in enumerate(all_image_paths):
            basename = os.path.basename(path)
            name_no_ext = os.path.splitext(basename)[0]
            
            try:
                parts = name_no_ext.split('_')
                if len(parts) >= 3 and parts[-2].isdigit():
                    age = int(parts[-2])
                else:
                    continue # 名字不符合规范的废图直接跳过
                    
                # 过滤异常年龄数据，保证训练稳定
                if age < 0 or age > 120:
                    continue
                    
                # 8:2 划分训练集和测试集
                is_test = (i % 5 == 0)
                
                if self.mode == 'train' and not is_test:
                    self.data_list.append((path, age))
                elif (self.mode == 'test' or self.mode == 'val') and is_test:
                    self.data_list.append((path, age))
                    
            except Exception:
                continue

        # =================================================================
        # Few-shot (少样本) 
        # =================================================================
        if self.mode == 'train':
            #  在这里设置你的 few-shot 比例！
            # 0.10 代表 10% 数据，0.01 代表 1% 数据。如果要跑全量，改为 1.0 即可。
            ratio = 0.01
            
            if ratio < 1.0:
                # 必须固定随机种子，保证每次提取的 Few-shot 子集完全相同，确保消融实验严谨！
                random.seed(42) 
                random.shuffle(self.data_list)
                num_samples = int(len(self.data_list) * ratio)
                self.data_list = self.data_list[:num_samples]
                print(f"[{self.mode.upper()}]  Few-shot 模式启动！仅使用 {ratio*100}% 的训练数据。")
        # =================================================================

        print(f"[{self.mode.upper()}] 成功加载带有【真实年龄】的样本: {len(self.data_list)} 张")

        self.instruct_dict = {}
        if instruct_json_path and os.path.exists(instruct_json_path):
            try:
                with open(instruct_json_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        item = json.loads(line)
                        key = os.path.basename(item['image'])
                        self.instruct_dict[key] = item
            except Exception: pass

        self.general_prompts = [
            "Estimate the age of the person.",
            "How old is the person in this image?",
            "Predict the apparent age based on facial features."
        ]

    def __getitem__(self, index):
        path, age_label = self.data_list[index]
        fname = os.path.basename(path)
        
        try:
            image = pil_loader(path)
        except:
            image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        raw_data = self.instruct_dict.get(fname, {})
        raw_d = ""
        if 'description' in raw_data:
             raw_d = raw_data['description']
        elif 'conversations' in raw_data:
             raw_d = raw_data['conversations'][1]['value']
             
        # 【关键兜底】如果没找到 LLaVA 描述，直接用真实年龄生成一句官方指令
        if not raw_d: 
            raw_d = "A person's face showing their current age."

        raw_q = random.choice(self.general_prompts) if self.mode == 'train' else "Estimate the age."

        if self.ablation == 'only_q': 
            instruction = raw_q; description = ""
        elif self.ablation == 'only_d': 
            instruction = ""; description = raw_d
        else: 
            instruction = raw_q; description = raw_d

        return image, torch.tensor(age_label, dtype=torch.long), instruction, description

    def __len__(self):
        return len(self.data_list)

InstructAgeDB = AgeDB

# =====================================================================
# 新增：专用于 AgeDB 性别分类的 Dataset 类 (男=0, 女=1)
# =====================================================================
class AgeDBGender(data.Dataset):
    def __init__(self, root=None, phase=None, transform=None, instruct_json_path=None, ablation='full', label_path=None, mode=None, img_dir=None):
        self.ablation = ablation
        self.transform = transform
        self.mode = mode if mode is not None else phase
        if self.mode is None: self.mode = 'train'
        self.img_dir = img_dir if img_dir else root

        print(f"[{self.mode.upper()}] 正在扫描 AgeDB 目录 (性别分类任务): {self.img_dir} ...")
        all_image_paths = []
        for root_dir, dirs, files in os.walk(self.img_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_image_paths.append(os.path.join(root_dir, file))
        all_image_paths.sort()
        
        self.data_list = []
        for i, path in enumerate(all_image_paths):
            basename = os.path.basename(path)
            name_no_ext = os.path.splitext(basename)[0]
            try:
                parts = name_no_ext.split('_')
                if len(parts) >= 4:
                    gender_str = parts[-1].lower()
                    if gender_str == 'm':
                        label = 0 # 男
                    elif gender_str == 'f':
                        label = 1 # 女
                    else:
                        continue
                else:
                    continue

                # 8:2 划分训练集和测试集
                is_test = (i % 5 == 0)
                if self.mode == 'train' and not is_test:
                    self.data_list.append((path, label))
                elif (self.mode == 'test' or self.mode == 'val') and is_test:
                    self.data_list.append((path, label))
            except Exception: continue

        print(f"[{self.mode.upper()}] 成功加载【性别标签】样本: {len(self.data_list)} 张 (Male=0, Female=1)")

        # 读取你之前用 LLaVA 生成的 JSON 描述
        self.instruct_dict = {}
        if instruct_json_path and os.path.exists(instruct_json_path):
            try:
                with open(instruct_json_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        item = json.loads(line)
                        key = os.path.basename(item['image'])
                        self.instruct_dict[key] = item
            except Exception: pass

        self.general_prompts = [
            "Identify the gender of the person in this image.",
            "Is the person in the image male or female?",
            "Classify the gender based on the facial features."
        ]

    def __getitem__(self, index):
        path, gender_label = self.data_list[index]
        fname = os.path.basename(path)
        try:
            image = pil_loader(path)
        except:
            image = Image.new('RGB', (224, 224))

        if self.transform: image = self.transform(image)
        else: image = transforms.ToTensor()(image)

        raw_data = self.instruct_dict.get(fname, {})
        raw_d = ""
        if 'description' in raw_data: raw_d = raw_data['description']
        elif 'conversations' in raw_data: raw_d = raw_data['conversations'][1]['value']
             
        # 兜底：如果没有描述，自动生成一句性别的描述
        if not raw_d: 
             gender_text = "male" if gender_label == 0 else "female"
             raw_d = "A person's face showing their gender characteristics."

        raw_q = random.choice(self.general_prompts) if self.mode == 'train' else "Identify the gender."

        if self.ablation == 'only_q': instruction = raw_q; description = ""
        elif self.ablation == 'only_d': instruction = ""; description = raw_d
        else: instruction = raw_q; description = raw_d

        return image, torch.tensor(gender_label, dtype=torch.long), instruction, description

    def __len__(self): return len(self.data_list)

InstructAgeDBGender = AgeDBGender