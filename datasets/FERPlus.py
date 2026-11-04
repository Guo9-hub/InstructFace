# # coding:utf-8
# import os
# import csv
# import json
# import random
# import torch
# import numpy as np
# import torch.utils.data as data
# from PIL import Image
# import torchvision.transforms as transforms

# EXPRESSION_NAMES = ["Neutral", "Happiness", "Surprise", "Sadness", "Anger", "Disgust", "Fear", "Contempt"]

# def pil_loader(path):
#     with open(path, 'rb') as f:
#         with Image.open(f) as img:
#             return img.convert('RGB')

# class InstructFERPlus(data.Dataset):
#     def __init__(self, label_path=None, mode=None, img_dir=None, transform=None, instruct_json_path=None, ablation='full', root=None, phase=None,few_shot_ratio=1.0):
#         self.ablation = ablation
#         self.transform = transform
#         self.few_shot_ratio = few_shot_ratio

#         # 1. 参数大一统 (兼容 root/phase 写法)
#         self.mode = mode if mode else phase
#         if self.mode is None: self.mode = 'train'
        
#         # 路径兼容
#         self.img_dir = img_dir if img_dir else root
#         if not self.img_dir: raise ValueError("必须提供图片路径 (img_dir 或 root)")

#         # 2. 【核心大招】预先扫描所有图片，建立索引 (你的逻辑，非常棒！)
#         print(f"[{self.mode.upper()}] 正在扫描图片目录: {self.img_dir} ... ")
#         self.image_path_map = {}
#         for root_dir, _, files in os.walk(self.img_dir):
#             for file in files:
#                 if file.lower().endswith(('.png', '.jpg', '.jpeg')):
#                     # 建立映射: 'fer00001.png' -> '/完整路径/fer00001.png'
#                     self.image_path_map[file] = os.path.join(root_dir, file)
        
#         print(f"[{self.mode.upper()}] 扫描完成，共索引 {len(self.image_path_map)} 张图片。")

#         # 3. 自动定位 CSV 文件
#         self.csv_path = label_path
#         if not self.csv_path or not os.path.exists(self.csv_path):
#             # 自动推导: 优先找 fer2013new.csv
#             candidates = [
#                 os.path.join(self.img_dir, 'fer2013new.csv'),
#                 os.path.join(os.path.dirname(self.img_dir.rstrip('/')), 'fer2013new.csv'), # 上一级
#                 os.path.join(os.path.dirname(os.path.dirname(self.img_dir.rstrip('/'))), 'fer2013new.csv') # 上上级
#             ]
#             for c in candidates:
#                 if os.path.exists(c):
#                     self.csv_path = c
#                     break
        
#         if not self.csv_path or not os.path.exists(self.csv_path):
#              print(f"[严重警告] 找不到 fer2013new.csv！请手动指定 label_path。")
#              self.data_list = []
#              self.soft_labels = []
#         else:
#              self._load_csv(self.csv_path, self.mode)
#         if self.mode == 'train' and 0 < self.few_shot_ratio < 1.0:
#             combined = list(zip(self.data_list, self.soft_labels))
#             random.seed(1234) # 固定种子保证实验可重复
#             random.shuffle(combined)
            
#             num_samples = int(len(combined) * self.few_shot_ratio)
#             combined = combined[:num_samples]
            
#             # 重新解包
#             self.data_list, self.soft_labels = zip(*combined)
#             self.data_list = list(self.data_list)
#             self.soft_labels = list(self.soft_labels)
#             print(f"!!! Few-shot 模式: 已随机抽取 {self.few_shot_ratio*100}% 数据，剩余 {len(self.data_list)} 条")

#         print(f"[{self.mode.upper()}] 最终加载有效样本: {len(self.data_list)} 张")

#         # 4. 指令加载
#         self.instruct_dict = {}
#         if instruct_json_path and os.path.exists(instruct_json_path):
#             try:
#                 with open(instruct_json_path, 'r') as f:
#                     self.instruct_dict = {os.path.basename(item['image']): item for item in json.load(f)}
#             except: pass
        
#         self.general_prompts = [
#             "Identify the emotion.", "Classify the expression.", "What is the emotion?", "Analyze the mood."
#         ]
    
#     def _process_data(self, emotion_raw):
#         """
#         采用官方 FERPlus 的严格过滤逻辑：
#         1. 移除只有 1 票的离群值
#         2. 最大票数必须过半，否则归为 Unknown
#         3. 剔除最终类别为 Unknown 和 Non-Face 的样本
#         """
#         # emotion_raw 的长度现在应为 10（包含 unknown 和 NF）
#         size = len(emotion_raw)
#         emotion_unknown = [0.0] * size
#         emotion_unknown[-2] = 1.0  # 设置为 unknown 的模式

#         # 移除离群票数 (< 1.0 的票忽略，加上浮点数 epsilon 容差)
#         for i in range(size):
#             if emotion_raw[i] < 1.0 + 1e-6:
#                 emotion_raw[i] = 0.0

#         sum_list = sum(emotion_raw)
#         emotion = [0.0] * size

#         # 寻找最高票，判断是否过半
#         if sum_list > 0:
#             maxval = max(emotion_raw)
#             if maxval > 0.5 * sum_list:
#                 emotion[np.argmax(emotion_raw)] = maxval
#             else:
#                 emotion = emotion_unknown  # 票数不过半，强制标记为 unknown
#         else:
#             emotion = emotion_unknown

#         # 找到最终的类别索引
#         idx = np.argmax(emotion)

#         # 只要前 8 个基础表情 (idx < 8)，过滤掉 unknown 和 NF
#         if idx < 8:
#             # 标准化处理并只返回前 8 个维度的标签
#             sum_emo = sum(emotion)
#             normalized_emotion = [float(i) / sum_emo for i in emotion]
#             return normalized_emotion[:8]
#         else:
#             return None  # 返回 None 代表丢弃该样本

#     def _load_csv(self, label_path, mode):
#         self.data_list = []
#         self.soft_labels = []
        
#         # ====================== 官方标准划分 ======================
#         if mode == 'train':
#             target_usages = ['Training']
#         else:
#             target_usages = ['PrivateTest']

#         with open(label_path, 'r') as f:
#             reader = csv.reader(f)
#             header = next(reader, None)
#             if header and 'usage' not in header[0].lower():
#                 f.seek(0)
            
#             for row in reader:
#                 try:
#                     if row[0].strip() not in target_usages:
#                         continue

#                     img_name = row[1].strip()
                    
#                     if img_name in self.image_path_map:
#                         full_path = self.image_path_map[img_name]
                        
#                         # 【关键修改】：读取所有的 10 个类别列（从索引 2 到 11），用来进行完整判断
#                         votes = list(map(float, row[2:12])) 
                        
#                         soft_label = self._process_data(votes)
                        
#                         # 如果没有被过滤掉，则加入数据集
#                         if soft_label is not None:
#                             self.data_list.append(full_path)
#                             self.soft_labels.append(np.array(soft_label, dtype=np.float32))
#                 except:
#                     continue

#     # def _process_data(self, emotion_raw):
#     #     emotion_clean = [x if x >= 1.0 - 1e-6 else 0.0 for x in emotion_raw]
#     #     sum_votes = sum(emotion_clean)
#     #     if sum_votes < 1e-6: return None
#     #     return [val / sum_votes for val in emotion_clean]

#     # def _load_csv(self, label_path, mode):
#     #     self.data_list = []
#     #     self.soft_labels = []
        
#     #     target_usages = ['Training'] if mode == 'train' else ['PublicTest', 'PrivateTest']

#     #     with open(label_path, 'r') as f:
#     #         reader = csv.reader(f)
#     #         header = next(reader, None)
#     #         if header and 'usage' not in header[0].lower(): f.seek(0)
            
#     #         for row in reader:
#     #             try:
#     #                 if row[0].strip() not in target_usages: continue

#     #                 img_name = row[1].strip()
                    
#     #                 # 只要文件名在刚才扫描的字典里，就能找到！
#     #                 if img_name in self.image_path_map:
#     #                     full_path = self.image_path_map[img_name]
#     #                     votes = list(map(float, row[2:10]))
#     #                     soft_label = self._process_data(votes)
                        
#     #                     if soft_label:
#     #                         self.data_list.append(full_path)
#     #                         self.soft_labels.append(np.array(soft_label, dtype=np.float32))
#     #             except: continue
#     # def _load_csv(self, label_path, mode):
#     #     self.data_list = []
#     #     self.soft_labels = []
        
#     #     # ====================== 官方标准划分 ======================
#     #     # FERPlus 官方固定划分：
#     #     # train = Training       → 25045 样本
#     #     # val/test = PrivateTest → 3137 样本（PublicTest 弃用）
#     #     # ==========================================================
#     #     if mode == 'train':
#     #         target_usages = ['Training']
#     #     else:
#     #         target_usages = ['PrivateTest']  # 官方测试集 ONLY

#     #     with open(label_path, 'r') as f:
#     #         reader = csv.reader(f)
#     #         header = next(reader, None)
#     #         if header and 'usage' not in header[0].lower():
#     #             f.seek(0)
            
#     #         for row in reader:
#     #             try:
#     #                 # 按官方 usage 筛选
#     #                 if row[0].strip() not in target_usages:
#     #                     continue

#     #                 img_name = row[1].strip()
                    
#     #                 if img_name in self.image_path_map:
#     #                     full_path = self.image_path_map[img_name]
#     #                     votes = list(map(float, row[2:10]))
#     #                     soft_label = self._process_data(votes)
                        
#     #                     if soft_label:
#     #                         self.data_list.append(full_path)
#     #                         self.soft_labels.append(np.array(soft_label, dtype=np.float32))
#     #             except:
#     #                 continue

#     def __getitem__(self, index):
#         path = self.data_list[index]
#         soft_target = self.soft_labels[index]
        
#         try: image = pil_loader(path)
#         except: image = Image.new('RGB', (48, 48))

#         # 保持你的 Resize 逻辑 (这对 FERPlus 小图很有用)
#         w, h = image.size
#         target_size = 224
#         scale = target_size / max(w, h)
#         new_w, new_h = int(w * scale), int(h * scale)
#         image = image.resize((new_w, new_h), Image.BILINEAR)
#         image_new = Image.new("RGB", (target_size, target_size))
#         image_new.paste(image, ((target_size - new_w) // 2, (target_size - new_h) // 2))
        
#         if self.transform: image_new = self.transform(image_new)

#         target_tensor = torch.from_numpy(soft_target)

#         # ================= [核心修改区域开始] =================
#         # 1. 提取文件名，用于在字典中查找对应描述
#         fname = os.path.basename(path)
#         raw_data = self.instruct_dict.get(fname, {})
        
#         raw_d = ""
#         # 2. 优先读取 LLaVA 生成的安全描述
#         if 'description' in raw_data: 
#             raw_d = raw_data['description']
#         elif 'conversations' in raw_data: 
#             raw_d = raw_data['conversations'][1]['value']
             
#         # 3. 【致命修复】兜底逻辑绝对不能带入真实标签 (hard_label)！
#         if not raw_d: 
#             # 删除了 EXPRESSION_NAMES[hard_label]，改为通用的安全占位符
#             raw_d = "A person's face showing a specific emotion."

#         # 4. 生成提问指令
#         raw_q = random.choice(self.general_prompts) if self.mode == 'train' else "Identify emotion."
#         # ================= [核心修改区域结束] =================
        
#         if self.ablation == 'only_q': instruction = raw_q; description = ""
#         elif self.ablation == 'only_d': instruction = ""; description = raw_d
#         else: instruction = raw_q; description = raw_d

#         return image_new, target_tensor, instruction, description

#     def __len__(self):
#         return len(self.data_list)

# # =========================================================
# # 【关键】加上这一行，防止 ImportError
# # =========================================================
# FERPlus = InstructFERPlus



# #------------------------------------------------------------------------------------------------------------------------------------
# #少样本
# # coding:utf-8
# import os
# import csv
# import json
# import random
# import torch
# import numpy as np
# import torch.utils.data as data
# from PIL import Image
# import torchvision.transforms as transforms

# EXPRESSION_NAMES = ["Neutral", "Happiness", "Surprise", "Sadness", "Anger", "Disgust", "Fear", "Contempt"]

# def pil_loader(path):
#     with open(path, 'rb') as f:
#         with Image.open(f) as img:
#             return img.convert('RGB')

# class InstructFERPlus(data.Dataset):
#     def __init__(self, label_path=None, mode=None, img_dir=None, transform=None, instruct_json_path=None, ablation='full', root=None, phase=None, few_shot_ratio=0.1):
#         self.ablation = ablation
#         self.transform = transform
#         self.few_shot_ratio = few_shot_ratio

#         # 1. 参数大一统 (兼容 root/phase 写法)
#         self.mode = mode if mode else phase
#         if self.mode is None: self.mode = 'train'
        
#         self.img_dir = img_dir if img_dir else root
#         if not self.img_dir: raise ValueError("必须提供图片路径 (img_dir 或 root)")

#         # 2. 预先扫描所有图片，建立索引
#         print(f"[{self.mode.upper()}] 正在扫描图片目录: {self.img_dir} ... ")
#         self.image_path_map = {}
#         for root_dir, _, files in os.walk(self.img_dir):
#             for file in files:
#                 if file.lower().endswith(('.png', '.jpg', '.jpeg')):
#                     self.image_path_map[file] = os.path.join(root_dir, file)
        
#         print(f"[{self.mode.upper()}] 扫描完成，共索引 {len(self.image_path_map)} 张图片。")

#         # 3. 自动定位 CSV 文件
#         self.csv_path = label_path
#         if not self.csv_path or not os.path.exists(self.csv_path):
#             candidates = [
#                 os.path.join(self.img_dir, 'fer2013new.csv'),
#                 os.path.join(os.path.dirname(self.img_dir.rstrip('/')), 'fer2013new.csv'), 
#                 os.path.join(os.path.dirname(os.path.dirname(self.img_dir.rstrip('/'))), 'fer2013new.csv') 
#             ]
#             for c in candidates:
#                 if os.path.exists(c):
#                     self.csv_path = c
#                     break
        
#         if not self.csv_path or not os.path.exists(self.csv_path):
#              print(f"[严重警告] 找不到 fer2013new.csv！请手动指定 label_path。")
#              self.data_list = []
#              self.soft_labels = []
#         else:
#              self._load_csv(self.csv_path, self.mode)
        
#         # ====================================================================
#         # 【升级版】安全的分层少样本采样 (Stratified Few-Shot Sampling)
#         # ====================================================================
#         if self.mode == 'train' and 0 < self.few_shot_ratio < 1.0:
#             random.seed(1234) # 固定种子保证实验可重复
            
#             # 按类别将数据分类存放
#             class_dict = {i: [] for i in range(8)}
#             for path, soft_label in zip(self.data_list, self.soft_labels):
#                 hard_label = np.argmax(soft_label)
#                 class_dict[hard_label].append((path, soft_label))

#             new_data_list = []
#             new_soft_labels = []

#             # 遍历8个类别，每个类别各自按比例抽取
#             for c in range(8):
#                 cls_samples = class_dict[c]
#                 if len(cls_samples) > 0:
#                     random.shuffle(cls_samples)
#                     # 计算该类别的抽取数量，强制 max(1, ...) 保证绝对不可能出现 0 个样本的类别
#                     num_cls_samples = max(1, int(len(cls_samples) * self.few_shot_ratio))
#                     sampled = cls_samples[:num_cls_samples]
#                     for path, soft_label in sampled:
#                         new_data_list.append(path)
#                         new_soft_labels.append(soft_label)

#             # 将重新抽取好的数据合并并再次打乱
#             combined = list(zip(new_data_list, new_soft_labels))
#             random.shuffle(combined)
            
#             self.data_list, self.soft_labels = zip(*combined)
#             self.data_list = list(self.data_list)
#             self.soft_labels = list(self.soft_labels)
#             print(f"!!! Stratified Few-shot 模式: 已分层抽取 {self.few_shot_ratio*100}% 数据，保证所有类别不为空。当前剩余 {len(self.data_list)} 条")
#         # ====================================================================

#         print(f"[{self.mode.upper()}] 最终加载有效样本: {len(self.data_list)} 张")

#         # 4. 指令加载
#         self.instruct_dict = {}
#         if instruct_json_path and os.path.exists(instruct_json_path):
#             try:
#                 with open(instruct_json_path, 'r') as f:
#                     self.instruct_dict = {os.path.basename(item['image']): item for item in json.load(f)}
#             except: pass
        
#         self.general_prompts = [
#             "Identify the emotion.", "Classify the expression.", "What is the emotion?", "Analyze the mood."
#         ]
    
#     def _process_data(self, emotion_raw):
#         # 官方 FERPlus 的严格过滤逻辑
#         size = len(emotion_raw)
#         emotion_unknown = [0.0] * size
#         emotion_unknown[-2] = 1.0  

#         for i in range(size):
#             if emotion_raw[i] < 1.0 + 1e-6:
#                 emotion_raw[i] = 0.0

#         sum_list = sum(emotion_raw)
#         emotion = [0.0] * size

#         if sum_list > 0:
#             maxval = max(emotion_raw)
#             if maxval > 0.5 * sum_list:
#                 emotion[np.argmax(emotion_raw)] = maxval
#             else:
#                 emotion = emotion_unknown  
#         else:
#             emotion = emotion_unknown

#         idx = np.argmax(emotion)

#         if idx < 8:
#             sum_emo = sum(emotion)
#             normalized_emotion = [float(i) / sum_emo for i in emotion]
#             return normalized_emotion[:8]
#         else:
#             return None  

#     def _load_csv(self, label_path, mode):
#         self.data_list = []
#         self.soft_labels = []
        
#         if mode == 'train':
#             target_usages = ['Training']
#         else:
#             target_usages = ['PrivateTest']

#         with open(label_path, 'r') as f:
#             reader = csv.reader(f)
#             header = next(reader, None)
#             if header and 'usage' not in header[0].lower():
#                 f.seek(0)
            
#             for row in reader:
#                 try:
#                     if row[0].strip() not in target_usages:
#                         continue

#                     img_name = row[1].strip()
                    
#                     if img_name in self.image_path_map:
#                         full_path = self.image_path_map[img_name]
#                         votes = list(map(float, row[2:12])) 
#                         soft_label = self._process_data(votes)
                        
#                         if soft_label is not None:
#                             self.data_list.append(full_path)
#                             self.soft_labels.append(np.array(soft_label, dtype=np.float32))
#                 except:
#                     continue

#     def __getitem__(self, index):
#         path = self.data_list[index]
#         soft_target = self.soft_labels[index]
        
#         try: image = pil_loader(path)
#         except: image = Image.new('RGB', (48, 48))

#         w, h = image.size
#         target_size = 224
#         scale = target_size / max(w, h)
#         new_w, new_h = int(w * scale), int(h * scale)
#         image = image.resize((new_w, new_h), Image.BILINEAR)
#         image_new = Image.new("RGB", (target_size, target_size))
#         image_new.paste(image, ((target_size - new_w) // 2, (target_size - new_h) // 2))
        
#         if self.transform: image_new = self.transform(image_new)

#         target_tensor = torch.from_numpy(soft_target)

#         fname = os.path.basename(path)
#         raw_data = self.instruct_dict.get(fname, {})
        
#         raw_d = ""
#         if 'description' in raw_data: 
#             raw_d = raw_data['description']
#         elif 'conversations' in raw_data: 
#             raw_d = raw_data['conversations'][1]['value']
             
#         if not raw_d: 
#             raw_d = "A person's face showing a specific emotion."

#         raw_q = random.choice(self.general_prompts) if self.mode == 'train' else "Identify emotion."
        
#         if self.ablation == 'only_q': instruction = raw_q; description = ""
#         elif self.ablation == 'only_d': instruction = ""; description = raw_d
#         else: instruction = raw_q; description = raw_d

#         return image_new, target_tensor, instruction, description

#     def __len__(self):
#         return len(self.data_list)

# FERPlus = InstructFERPlus


#-------------------------------------------------------------------------------------------
#官方少样本
# coding:utf-8
import os
import sys
import csv
import json
import random
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

# 基础表情参考
EXPRESSION_NAMES = ["Neutral", "Happiness", "Surprise", "Sadness", "Anger", "Disgust", "Fear", "Contempt"]

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class InstructFERPlus(data.Dataset):
    def __init__(self, label_path=None, mode=None, img_dir=None, transform=None, instruct_json_path=None, ablation='full', root=None, phase=None, few_shot_ratio=0.1):
        # 1. 兼容 finetune 主脚本的传参
        self.ablation = ablation
        self.transform = transform
        self.few_shot_ratio = few_shot_ratio

        self.mode = mode if mode else phase
        if self.mode is None: self.mode = 'train'
        
        self.img_dir = img_dir if img_dir else root
        self.csv_path = label_path
        
        if not self.img_dir: raise ValueError("必须提供图片路径 (img_dir 或 root)")

        # 2. 预先扫描所有图片，建立索引 (解决你的图片可能不在同一个子文件夹的问题)
        print(f"[{self.mode.upper()}] 正在扫描图片目录: {self.img_dir} ... ")
        self.image_path_map = {}
        for root_dir, _, files in os.walk(self.img_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_path_map[file] = os.path.join(root_dir, file)

        if not self.csv_path or not os.path.exists(self.csv_path):
             print(f"[严重警告] 找不到 fer2013new.csv！请检查 label_path 路径: {self.csv_path}")
             self.data_list = []
             self.soft_labels = []
        else:
             self._load_csv(self.csv_path, self.mode)
        
        # ====================================================================
        # 3. 安全的分层少样本采样 (Stratified Few-Shot Sampling)
        # ====================================================================
        if self.mode == 'train' and 0 < self.few_shot_ratio < 1.0:
            random.seed(1234) # 固定种子保证实验可重复
            
            # 按最大概率的类别 (Hard Label) 将数据分类存放
            class_dict = {i: [] for i in range(8)}
            for path, soft_label in zip(self.data_list, self.soft_labels):
                hard_label = np.argmax(soft_label)
                class_dict[hard_label].append((path, soft_label))

            new_data_list = []
            new_soft_labels = []

            # 遍历 8 个类别，分别按比例抽取
            for c in range(8):
                cls_samples = class_dict[c]
                if len(cls_samples) > 0:
                    random.shuffle(cls_samples)
                    # 强制保底 1 张，防止极度不平衡类别（如 Disgust）变成 0 张导致模型 NaN 崩溃
                    num_cls_samples = max(1, int(len(cls_samples) * self.few_shot_ratio))
                    sampled = cls_samples[:num_cls_samples]
                    for path, soft_label in sampled:
                        new_data_list.append(path)
                        new_soft_labels.append(soft_label)

            # 将重新抽取好的数据合并并再次打乱
            combined = list(zip(new_data_list, new_soft_labels))
            random.shuffle(combined)
            
            self.data_list, self.soft_labels = zip(*combined)
            self.data_list = list(self.data_list)
            self.soft_labels = list(self.soft_labels)
            print(f"!!! Stratified Few-shot 模式: 已分层抽取 {self.few_shot_ratio*100}% 数据，保证所有类别不为空。当前剩余 {len(self.data_list)} 条")
        else:
            print(f"[{self.mode.upper()}] 最终加载有效样本: {len(self.data_list)} 张")

        # 4. 指令加载
        self.instruct_dict = {}
        if instruct_json_path and os.path.exists(instruct_json_path):
            try:
                with open(instruct_json_path, 'r') as f:
                    self.instruct_dict = {os.path.basename(item['image']): item for item in json.load(f)}
            except: pass
        
        self.general_prompts = [
            "Identify the emotion.", "Classify the expression.", "What is the emotion?", "Analyze the mood."
        ]
    
    @staticmethod
    def _process_data(emotion_raw):
        """
        官方 FERPlus 处理逻辑 (多数表决 Majority Voting)：
        1. 移除票数 <= 1.0 的噪声离群值
        2. 最大票数必须过半，否则归类为 Unknown
        3. 转换成 Soft Label 概率分布返回
        """
        size = len(emotion_raw)
        emotion_unknown = [0.0] * size
        emotion_unknown[-2] = 1.0  # Unknown index

        # 移除离群票数
        for i in range(size):
            if emotion_raw[i] < 1.0 + sys.float_info.epsilon:
                emotion_raw[i] = 0.0

        sum_list = sum(emotion_raw)
        emotion = [0.0] * size

        if sum_list > 0:
            maxval = max(emotion_raw)
            if maxval > 0.5 * sum_list:
                emotion[np.argmax(emotion_raw)] = maxval
            else:
                emotion = emotion_unknown  
        else:
            emotion = emotion_unknown

        idx = np.argmax(emotion)

        # 只要前 8 个基础表情 (idx < 8)，过滤掉 Unknown 和 Non-Face
        if idx < 8:
            sum_emo = sum(emotion)
            normalized_emotion = [float(i) / sum_emo for i in emotion]
            return normalized_emotion[:8]
        else:
            return None  

    def _load_csv(self, label_path, mode):
        self.data_list = []
        self.soft_labels = []
        
        # 官方标准的划分方式
        if mode == 'train':
            target_usages = ['Training']
        else:
            target_usages = ['PrivateTest', 'PublicTest']

        with open(label_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header and 'usage' not in header[0].lower():
                f.seek(0)
            
            for row in reader:
                try:
                    if row[0].strip() not in target_usages:
                        continue

                    img_name = row[1].strip()
                    
                    if img_name in self.image_path_map:
                        full_path = self.image_path_map[img_name]
                        # 读取 10 个维度的原始投票数据
                        votes = list(map(float, row[2:12])) 
                        soft_label = self._process_data(votes)
                        
                        if soft_label is not None:
                            self.data_list.append(full_path)
                            self.soft_labels.append(np.array(soft_label, dtype=np.float32))
                except:
                    continue

    def __getitem__(self, index):
        path = self.data_list[index]
        soft_target = self.soft_labels[index]
        
        try: 
            image = pil_loader(path)
        except: 
            image = Image.new('RGB', (48, 48))

        # ==========================================================
        # 【关键修复】：官方的纯净图像流 (丢弃老代码的 numpy 黑色边缘填充)
        # 把原始图片直接喂给外部的 transform_train，保证最高精度！
        # ==========================================================
        if self.transform: 
            image_new = self.transform(image)
        else:
            image_new = image

        target_tensor = torch.from_numpy(soft_target)

        fname = os.path.basename(path)
        raw_data = self.instruct_dict.get(fname, {})
        
        raw_d = ""
        # 优先读取 LLaVA 生成的安全描述
        if 'description' in raw_data: 
            raw_d = raw_data['description']
        elif 'conversations' in raw_data: 
            raw_d = raw_data['conversations'][1]['value']
             
        # 兜底逻辑：绝对不能泄露真实标签！
        if not raw_d: 
            raw_d = "A person's face showing a specific emotion."

        raw_q = random.choice(self.general_prompts) if self.mode == 'train' else "Identify emotion."
        
        # 文本消融实验控制
        if self.ablation == 'only_q': instruction = raw_q; description = ""
        elif self.ablation == 'only_d': instruction = ""; description = raw_d
        else: instruction = raw_q; description = raw_d

        return image_new, target_tensor, instruction, description

    def __len__(self):
        return len(self.data_list)

# 兼容你的 main 脚本
FERPlus = InstructFERPlus