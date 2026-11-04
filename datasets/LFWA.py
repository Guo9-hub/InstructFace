
# #原来用的代码
# #coding:utf-8
# from PIL import Image
# import torch.utils.data as data
# import numpy as np
# import os
# import torchvision.transforms as transforms
# import torch
# import json
# import re
# import random

# # 40个人脸属性作为模型预测的目标
# # ATTR_NAMES = [
# #     "Male", "Asian", "White", "Black", "Baby", "Child", "Youth", "Middle Aged", "Senior", 
# #     "Black Hair", "Blond Hair", "Brown Hair", "Bald", "No Eyewear", "Eyeglasses", "Sunglasses", 
# #     "Mustache", "Smiling", "Frowning", "Chubby", "Blurry", "Harsh Lighting", "Flash", 
# #     "Soft Lighting", "Outdoor", "Curly Hair", "Wavy Hair", "Straight Hair", "Receding Hairline", 
# #     "Bangs", "Sideburns", "Fully Visible Forehead", "Partially Visible Forehead", 
# #     "Obstructed Forehead", "Bushy Eyebrows", "Arched Eyebrows", "Narrow Eyes", "Eyes Open", 
# #     "Big Nose", "Pointy Nose"
# # ]
# ATTR_NAMES = [
#     "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
#     "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
#     "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
#     "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
#     "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
#     "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks",
#     "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
#     "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"
# ]

# def pil_loader(path):
#     with open(path, 'rb') as f:
#         with Image.open(f) as img:
#             return img.convert('RGB')

# class InstructLFWA(data.Dataset):
#     # def __init__(self, attr_dir, mode, img_dir, transform, instruct_json_path):
#     def __init__(self, attr_dir, mode, img_dir, transform, instruct_json_path,ablation='full'):
#         self.ablation = ablation
#         self.mode = mode
#         self.attr = np.zeros((6572 if mode == 'train' else 6571, 40))
        
#         # 适配 LLaVA conversations 格式
#         with open(instruct_json_path, 'r') as f:
#             instruct_list = json.load(f)
            
#         self.instruct_dict = {}
#         for item in instruct_list:
#             # conversations[0] 指令存为 q，conversations[1] 文本描述存为 d
#             img_key = os.path.basename(item['image'])
#             q_val = item['conversations'][0]['value']
#             d_val = item['conversations'][1]['value']
#             self.instruct_dict[img_key] = {
#                 'q': q_val,
#                 'd': d_val
#             }
        
#         # 2. 加载属性标签 ，将属性标签转换为二进制0 1
#         with open(attr_dir) as f:
#             f.readline(); f.readline()
#             lines = f.readlines()
#             flag = 0 if mode == 'train' else 1
#             self.imgs = []; idx = 0
#             for line in lines:
#                 flag = 1 - flag
#                 if flag % 2 == 0: continue
                
#                 vals = line.split()
#                 len_name = len(vals) - 74 
#                 person_name = "_".join(vals[:len_name])
#                 img_name = f"{person_name}/{person_name}_{str(vals[len_name]).zfill(4)}.jpg"
#                 self.imgs.append(img_name)

#                 temp = np.array([float(s) for s in vals[len_name+1 : len_name+41]])
#                 temp[temp >= 0] = 1
#                 temp[temp < 0] = 0
#                 self.attr[idx, :] = temp
#                 idx += 1

#         self.transform = transform
#         self.img_dir = img_dir

#         # 3. 论文级指令池：用于训练时的任务变换
#         self.general_prompts = [
#             "Please analyze the facial attributes of this person.",
#             "List all the facial features you can identify from this image.",
#             "Examine this face and describe its key characteristics.",
#             "What can you tell me about this person's appearance?",
#             "Identify the visual traits present in this facial image."
#         ]

# # class InstructLFWA(data.Dataset):
# #     def __init__(self, attr_dir, mode, img_dir, transform, instruct_json_path, ablation='full',
# #     train_ratio=0.8, seed=42):
# #         self.ablation = ablation
# #         self.mode = mode
# #         self.img_dir = img_dir
# #         self.transform = transform
        
# #         # 1. 加载 LLaVA 指令 JSON
# #         with open(instruct_json_path, 'r') as f:
# #             instruct_list = json.load(f)
# #         self.instruct_dict = {}
# #         for item in instruct_list:
# #             img_key = os.path.basename(item['image'])
# #             self.instruct_dict[img_key] = {
# #                 'q': item['conversations'][0]['value'],
# #                 'd': item['conversations'][1]['value']
# #             }

# #         # 2. 基于身份划分
# #         all_entries = [] # 存储格式: (person_name, img_name, attr_vector)
        
# #         with open(attr_dir) as f:
# #             f.readline(); f.readline() # 跳过前两行
# #             lines = f.readlines()
            
# #             for line in lines:
# #                 vals = line.split()
# #                 # 动态计算人名长度
# #                 len_name = len(vals) - 74 
# #                 person_name = "_".join(vals[:len_name])
# #                 img_name = f"{person_name}/{person_name}_{str(vals[len_name]).zfill(4)}.jpg"
                
# #                 # 提取前40个属性标签
# #                 temp_attr = np.array([float(s) for s in vals[len_name+1 : len_name+41]])
# #                 temp_attr[temp_attr >= 0] = 1
# #                 temp_attr[temp_attr < 0] = 0
                
# #                 all_entries.append((person_name, img_name, temp_attr))

# #         # 3. 按人名聚类并打乱
# #         # 获取所有唯一的人名
# #         unique_persons = sorted(list(set([e[0] for e in all_entries])))
# #         random.seed(seed) # 固定随机种子，保证每次运行划分一致
# #         random.shuffle(unique_persons)

# #         # 计算切分点
# #         split_idx = int(len(unique_persons) * train_ratio)
# #         train_persons = set(unique_persons[:split_idx])
# #         test_persons = set(unique_persons[split_idx:])

# #         # 4. 根据当前模式 (train/test) 过滤数据
# #         self.imgs = []
# #         final_attrs = []
        
# #         target_persons = train_persons if mode == 'train' else test_persons
        
# #         for p_name, i_name, a_vec in all_entries:
# #             if p_name in target_persons:
# #                 self.imgs.append(i_name)
# #                 final_attrs.append(a_vec)
        
# #         self.attr = np.array(final_attrs)
        
# #         # print(f"[*] {mode} 模式加载完成: {len(unique_persons)} 总人数, "
# #         #       f"当前模式人数: {len(target_persons)}, 图片数: {len(self.imgs)}")

# #         # 5. 论文级指令池 
# #         self.general_prompts = [
# #             "Please analyze the facial attributes of this person.",
# #             "List all the facial features you can identify from this image.",
# #             "Examine this face and describe its key characteristics.",
# #             "What can you tell me about this person's appearance?",
# #             "Identify the visual traits present in this facial image."
# #         ]

#     def __getitem__(self, index):
#         img_path_rel = self.imgs[index]
#         image = pil_loader(os.path.join(self.img_dir, img_path_rel))
        
#         # 4. 图像对齐与缩放逻辑 (Baseline)  
#         w, h = image.size
#         new_w = int(1.0 * w / h * 224)
#         image = transforms.Resize((224, new_w))(image)
#         image_np = np.array(image)
#         image_new = np.zeros((224, 224, 3))
#         add_num = int((224 - new_w) / 2)
#         image_new[:, add_num:add_num+image_np.shape[1], :] = image_np
#         image_new = Image.fromarray(np.uint8(image_new)).convert('RGB')
        
#         if self.transform: 
#             image_new = self.transform(image_new)

#         # 5. 提取语义对齐所需的 Q 和 D
#         img_filename = os.path.basename(img_path_rel)
#         raw_data = self.instruct_dict.get(img_filename, {})
        
#         # 获取原始文本数据
#         raw_d = raw_data.get('d', "A facial portrait.")
#         # 训练时使用随机指令增强鲁棒性，测试时使用固定指令保证公平
#         if self.mode == 'train':
#             raw_q = random.choice(self.general_prompts)
#         else:
#             raw_q = "Face Attribute: Identify specific facial attributes."

    
        
#         if self.ablation == 'only_q':
#             # 只有指令，没有描述
#             instruction = raw_q
#             vivid_description = "" 
#         elif self.ablation == 'only_d':
#             # 只有描述，没有指令
#             instruction = ""
#             vivid_description = raw_d
#         else: # 'full' 模式
#             # 指令 + 描述
#             instruction = raw_q
#             vivid_description = raw_d

#         # 返回四个值：图片，标签，指令 Q，文本描述 D
#         return image_new, torch.tensor(self.attr[index, :], dtype=torch.float32), instruction, vivid_description

#     def __len__(self):
#         return len(self.imgs)   
#-------------------------------------------------------------------------------------------------------------------------------------

#少样本
#coding:utf-8
from PIL import Image
import torch.utils.data as data
import numpy as np
import os
import torchvision.transforms as transforms
import torch
import json
import re
import random

ATTR_NAMES = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
    "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks",
    "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
    "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"
]

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class InstructLFWA(data.Dataset):
    # 新增 fewshot_rate 和 seed 参数
    def __init__(self, attr_dir, mode, img_dir, transform, instruct_json_path, ablation='full', fewshot_rate=0.05, seed=42):
        self.ablation = ablation
        self.mode = mode
        # LFWA 原本的 hardcode 初始化
        self.attr = np.zeros((6572 if mode == 'train' else 6571, 40))
        
        # 1. 适配 LLaVA conversations 格式
        with open(instruct_json_path, 'r') as f:
            instruct_list = json.load(f)
            
        self.instruct_dict = {}
        for item in instruct_list:
            # conversations[0] 指令存为 q，conversations[1] 文本描述存为 d
            img_key = os.path.basename(item['image'])
            q_val = item['conversations'][0]['value']
            d_val = item['conversations'][1]['value']
            self.instruct_dict[img_key] = {
                'q': q_val,
                'd': d_val
            }
        
        # 2. 加载属性标签 ，将属性标签转换为二进制0 1
        with open(attr_dir) as f:
            f.readline(); f.readline()
            lines = f.readlines()
            flag = 0 if mode == 'train' else 1
            self.imgs = []; idx = 0
            for line in lines:
                flag = 1 - flag
                if flag % 2 == 0: continue
                
                vals = line.split()
                len_name = len(vals) - 74 
                person_name = "_".join(vals[:len_name])
                img_name = f"{person_name}/{person_name}_{str(vals[len_name]).zfill(4)}.jpg"
                self.imgs.append(img_name)

                temp = np.array([float(s) for s in vals[len_name+1 : len_name+41]])
                temp[temp >= 0] = 1
                temp[temp < 0] = 0
                self.attr[idx, :] = temp
                idx += 1

        # =========================================================
        # 3. 新增：少样本 (Few-Shot) 数据采样逻辑
        # 因为 imgs 是列表，attr 是 numpy 矩阵，我们需要采样 index 来保持它们一一对应
        # =========================================================
        if self.mode == 'train' and fewshot_rate < 1.0:
            random.seed(seed)  # 固定种子
            num_samples = max(1, int(len(self.imgs) * fewshot_rate))
            
            # 随机抽取对应的索引
            sampled_indices = random.sample(range(len(self.imgs)), num_samples)
            
            # 同步切片更新 imgs 和 attr
            self.imgs = [self.imgs[i] for i in sampled_indices]
            self.attr = self.attr[sampled_indices, :]
            
            print(f"[{mode.upper()} FEW-SHOT] Randomly sampled {num_samples} images (Rate: {fewshot_rate*100}%).")
        else:
            print(f"[{mode.upper()}] Final loaded {len(self.imgs)} images for LFWA.")
        # =========================================================

        self.transform = transform
        self.img_dir = img_dir

        # 4. 论文级指令池：用于训练时的任务变换
        self.general_prompts = [
            "Please analyze the facial attributes of this person.",
            "List all the facial features you can identify from this image.",
            "Examine this face and describe its key characteristics.",
            "What can you tell me about this person's appearance?",
            "Identify the visual traits present in this facial image."
        ]

    def __getitem__(self, index):
        img_path_rel = self.imgs[index]
        image = pil_loader(os.path.join(self.img_dir, img_path_rel))
        
        # 4. 图像对齐与缩放逻辑 (Baseline)  
        w, h = image.size
        new_w = int(1.0 * w / h * 224)
        image = transforms.Resize((224, new_w))(image)
        image_np = np.array(image)
        image_new = np.zeros((224, 224, 3))
        add_num = int((224 - new_w) / 2)
        image_new[:, add_num:add_num+image_np.shape[1], :] = image_np
        image_new = Image.fromarray(np.uint8(image_new)).convert('RGB')
        
        if self.transform: 
            image_new = self.transform(image_new)

        # 5. 提取语义对齐所需的 Q 和 D
        img_filename = os.path.basename(img_path_rel)
        raw_data = self.instruct_dict.get(img_filename, {})
        
        # 获取原始文本数据
        raw_d = raw_data.get('d', "A facial portrait.")
        # 训练时使用随机指令增强鲁棒性，测试时使用固定指令保证公平
        if self.mode == 'train':
            raw_q = random.choice(self.general_prompts)
        else:
            raw_q = "Face Attribute: Identify specific facial attributes."

        if self.ablation == 'only_q':
            # 只有指令，没有描述
            instruction = raw_q
            vivid_description = "" 
        elif self.ablation == 'only_d':
            # 只有描述，没有指令
            instruction = ""
            vivid_description = raw_d
        else: # 'full' 模式
            # 指令 + 描述
            instruction = raw_q
            vivid_description = raw_d

        # 返回四个值：图片，标签，指令 Q，文本描述 D
        return image_new, torch.tensor(self.attr[index, :], dtype=torch.float32), instruction, vivid_description

    def __len__(self):
        return len(self.imgs)
# #--------------------------------------------------------------------


# #coding:utf-8
# from PIL import Image
# import torch.utils.data as data
# import numpy as np
# import os
# import torchvision.transforms as transforms
# import torch
# import json
# import random

# # 40个人脸属性作为模型预测的目标 (供参考)
# ATTR_NAMES = [
#     "Male", "Asian", "White", "Black", "Baby", "Child", "Youth", "Middle Aged", "Senior", 
#     "Black Hair", "Blond Hair", "Brown Hair", "Bald", "No Eyewear", "Eyeglasses", "Sunglasses", 
#     "Mustache", "Smiling", "Frowning", "Chubby", "Blurry", "Harsh Lighting", "Flash", 
#     "Soft Lighting", "Outdoor", "Curly Hair", "Wavy Hair", "Straight Hair", "Receding Hairline", 
#     "Bangs", "Sideburns", "Fully Visible Forehead", "Partially Visible Forehead", 
#     "Obstructed Forehead", "Bushy Eyebrows", "Arched Eyebrows", "Narrow Eyes", "Eyes Open", 
#     "Big Nose", "Pointy Nose"
# ]

# def pil_loader(path):
#     with open(path, 'rb') as f:
#         with Image.open(f) as img:
#             return img.convert('RGB')

# class InstructLFWA(data.Dataset):
#     def __init__(self, attr_dir, mode, img_dir, transform, instruct_json_path, ablation='full'):
#         self.ablation = ablation
#         self.mode = mode
#         self.transform = transform
#         self.img_dir = img_dir

#         # 1. 数组大小完美对齐基线
#         if mode == 'train':
#             self.attr = np.zeros((6572, 40))
#         else:
#             self.attr = np.zeros((6571, 40))

#         # 2. 加载 LLaVA 生成的指令和描述 (JSON)
#         with open(instruct_json_path, 'r') as f:
#             instruct_list = json.load(f)
            
#         self.instruct_dict = {}
#         for item in instruct_list:
#             # 统一提取文件名（例如 "Aaron_Eckhart_0001.jpg"）作为匹配键
#             img_key = os.path.basename(item['image'])
#             q_val = item['conversations'][0]['value']
#             d_val = item['conversations'][1]['value']
#             self.instruct_dict[img_key] = {'q': q_val, 'd': d_val}

#         # 3. 读取属性，完美对齐基线切分和映射逻辑
#         with open(attr_dir) as f:
#             f.readline()
#             f.readline()
#             lines = f.readlines()

#             if mode == 'train':
#                 flag = 0
#             else:
#                 flag = 1

#             self.imgs = []
#             idx = 0

#             for line in lines:
#                 # 完美对齐基线：使用奇偶行划分 train 和 test
#                 flag = 1 - flag
#                 if flag % 2 == 0:
#                     continue

#                 vals = line.split()
#                 len_name = len(vals) - 74

#                 # 完美对齐基线：拼接人名与路径
#                 person_name = vals[0]
#                 for i in range(1, len_name):
#                     person_name += '_' + vals[i]
#                 img_name = person_name + '/' + person_name + '_' + str(vals[len_name]).zfill(4) + '.jpg'

#                 self.imgs.append(img_name)

#                 # 完美对齐基线：特定索引抽取 (非常关键！如果你不加这个，模型学到的全是乱码标签)
#                 index = [65, 36, 56, 60, 13, 30, 41, 39, 10, 11,
#                          21, 12, 35, 20, 49, 15, 47, 59, 61, 69,
#                          1,  43, 17, 37, 46, 51, 64, 40, 29, 62,
#                          31, 18, 28, 27, 71, 50, 67, 73, 72, 7]

#                 temp = np.array([float(s) for s in vals[-74:]])
#                 temp[temp >= 0] = 1
#                 temp[temp < 0] = 0

#                 self.attr[idx, :] = [temp[i] for i in index]

#                 # 完美对齐基线：特殊人工修正 (必不可少)
#                 if temp[57] > 0:
#                     self.attr[idx, 2] = 1

#                 idx += 1

#         self.length = len(self.imgs)

#         # 4. 论文级指令池
#         self.general_prompts = [
#             "Please analyze the facial attributes of this person.",
#             "List all the facial features you can identify from this image.",
#             "Examine this face and describe its key characteristics.",
#             "What can you tell me about this person's appearance?",
#             "Identify the visual traits present in this facial image."
#         ]
        
#         print(f"[{mode.upper()}] Loaded {len(self.imgs)} images from LFWA.")

#     def __getitem__(self, index):
#         img_path_rel = self.imgs[index]
#         image = pil_loader(os.path.join(self.img_dir, img_path_rel))
        
#         # ==========================================================
#         # 完全复刻基线的图像处理逻辑 (包含原版的 np.resize 特性)
#         # ==========================================================
#         w, h = image.size
#         new_w = int(1.0 * w / h * 224)
#         image = transforms.Resize((224, new_w))(image)
#         image = np.array(image)

#         image_new = np.zeros((224, 224, 3))
#         add_num = (224 - new_w) / 2

#         list1 = image_new[:, int(add_num):int(add_num+new_w), :]
#         image_new = np.resize(image, (list1.shape[0], list1.shape[1], list1.shape[2]))

#         image_new = Image.fromarray(np.uint8(image_new))
#         image_new = image_new.convert('RGB')

#         if self.transform is not None:
#             image_new = self.transform(image_new)
#         # ==========================================================

#         # 获取指令和描述
#         img_filename = os.path.basename(img_path_rel)
#         raw_data = self.instruct_dict.get(img_filename, {})
#         raw_d = raw_data.get('d', "A facial portrait.")
        
#         if self.mode == 'train':
#             raw_q = random.choice(self.general_prompts)
#         else:
#             raw_q = "Face Attribute: Identify specific facial attributes."

#         # 消融实验
#         if self.ablation == 'only_q':
#             instruction = raw_q
#             vivid_description = "" 
#         elif self.ablation == 'only_d':
#             instruction = ""
#             vivid_description = raw_d
#         else: 
#             instruction = raw_q
#             vivid_description = raw_d

#         return image_new, torch.tensor(self.attr[index, :], dtype=torch.float32), instruction, vivid_description

#     def __len__(self):
#         return self.length
