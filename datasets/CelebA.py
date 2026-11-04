# # 原来都是用的这个跑的 
# # #coding:utf-8
# from PIL import Image
# import torch.utils.data as data
# import numpy as np
# import os
# import torchvision.transforms as transforms
# import torch
# import json
# import random
# from torchvision.transforms import InterpolationMode
# # CelebA 40 属性名 (供参考)
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

# class InstructCelebA(data.Dataset):
#     def __init__(self, attr_dir, mode, img_dir, transform, instruct_json_path, part_dir, ablation='full'):
#         """
#         attr_dir: list_attr_celeba.txt 的路径
#         part_dir: list_eval_partition.txt 的路径 (CelebA 专用划分文件)
#         mode: 'train', 'val', or 'test'
#         """
#         self.ablation = ablation
#         self.mode = mode
#         self.img_dir = img_dir
#         self.transform = transform
        
#         # 1. 定义 partition 映射
#         # CelebA 标准: 0=Train, 1=Val, 2=Test
#         if mode == 'train':
#             target_partition = '0'
#         elif mode == 'val':
#             target_partition = '1'
#         elif mode == 'test':
#             target_partition = '2'
#         else:
#             raise ValueError(f"Unknown mode: {mode}")

#         # 2. 读取 partition 文件，确定哪些图片属于当前 mode
#         self.imgs = []
#         with open(part_dir, 'r') as f:
#             lines = f.readlines()
#             for line in lines:
#                 fname, partition = line.strip().split()
#                 if partition == target_partition:
#                     self.imgs.append(fname)
        
#         # 3. 加载属性 (Attr)
#         # CelebA 的 attr 文件前两行是 header，后面每一行对应一张图 (按文件名排序)
#         # 我们需要构建一个字典或者直接按顺序索引，因为 partition 文件和 attr 文件的图片顺序是一致的
        
#         # 为了稳健，我们先读取所有属性存入字典: {filename: [attrs]}
#         self.attr_dict = {}
#         with open(attr_dir, 'r') as f:
#             num_imgs = int(f.readline().strip()) # 第一行是数量
#             attr_names = f.readline().strip().split() # 第二行是属性名
            
#             for line in f:
#                 vals = line.strip().split()
#                 fname = vals[0]
#                 # 属性值是 1 或 -1，需要转成 0/1
#                 # vals[1:] 是 40 个属性
#                 attrs = np.array([1 if int(x) == 1 else 0 for x in vals[1:]], dtype=np.float32)
#                 self.attr_dict[fname] = attrs

#         # 4. 加载 LLaVA 生成的指令和描述 (JSON)
#         # 假设 JSON 格式和你之前生成的一样: [{"image": "000001.jpg", "conversations": [...]}, ...]
#         with open(instruct_json_path, 'r') as f:
#             instruct_list = json.load(f)
            
#         self.instruct_dict = {}
#         for item in instruct_list:
#             # 兼容处理：无论 JSON 里的 image 字段是 "000001.jpg" 还是 "img_align_celeba/000001.jpg"
#             # 我们都只取文件名
#             img_key = os.path.basename(item['image']) 
            
#             q_val = item['conversations'][0]['value']
#             d_val = item['conversations'][1]['value']
            
#             self.instruct_dict[img_key] = {
#                 'q': q_val,
#                 'd': d_val
#             }

#         # 5. 论文级指令池 (Instruction Pool)
#         self.general_prompts = [
#             "Please analyze the facial attributes of this person.",
#             "List all the facial features you can identify from this image.",
#             "Examine this face and describe its key characteristics.",
#             "What can you tell me about this person's appearance?",
#             "Identify the visual traits present in this facial image."
#         ]
        
#         print(f"[{mode.upper()}] Loaded {len(self.imgs)} images from CelebA.")

#     def __getitem__(self, index):
#         # 获取文件名 (e.g., "000001.jpg")
#         img_fname = self.imgs[index]
        
#         # 加载图片
#         # CelebA 图片通常在 img_align_celeba 目录下
#         # 这里的 self.img_dir 应该指向 .../img_align_celeba/
#         image = pil_loader(os.path.join(self.img_dir, img_fname))
        
#         # # 1. 图像预处理 (保持你的 Padding 逻辑)
#         # w, h = image.size
#         # new_w = int(1.0 * w / h * 224)
#         # image = transforms.Resize((224, new_w))(image)
#         # image_np = np.array(image)
        
#         # # 创建 224x224 画布
#         # image_new = np.zeros((224, 224, 3), dtype=np.uint8)
#         # add_num = int((224 - new_w) / 2)
        
#         # # 防止越界 (robustness)
#         # if image_np.ndim == 2: # 灰度图处理
#         #     image_np = np.stack((image_np,)*3, axis=-1)
        
#         # # 填入画布
#         # # 注意：这里如果 new_w 算出来是 224，add_num=0，逻辑也没问题
#         # # 使用切片赋值比 resize 更安全，保持长宽比
#         # # 你的旧代码逻辑有点乱，我帮你整理清晰了：
#         # valid_w = min(new_w, 224)
#         # image_new[:, add_num : add_num + valid_w, :] = image_np[:, :valid_w, :]
        
#         # image_new = Image.fromarray(image_new).convert('RGB')
        
#         # if self.transform: 
#         #     image_new = self.transform(image_new)
#         # image = transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC)(image)

#         image = transforms.Resize((224, 224))(image)
        
#         if self.transform: 
#             image_new = self.transform(image)
#         else:
#             image_new = transforms.ToTensor()(image)
#         # if self.transform:
#         #     image_new = self.transform(image)
#         # else:
#         #     image_new = transforms.ToTensor()(image)

#         # 2. 获取属性标签
#         label = self.attr_dict[img_fname] # 从字典里取，更安全
        
#         # 3. 获取文本指令和描述
#         raw_data = self.instruct_dict.get(img_fname, {})
#         raw_d = raw_data.get('d', "A photo of a face.") # 默认描述
        
#         # 训练时随机指令，测试时固定指令
#         if self.mode == 'train':
#             raw_q = random.choice(self.general_prompts)
#         else:
#             raw_q = "Face Attribute: Identify specific facial attributes."

#         # 4. 消融实验逻辑 (Ablation)
#         if self.ablation == 'only_q':
#             instruction = raw_q
#             vivid_description = "" 
#         elif self.ablation == 'only_d':
#             instruction = ""
#             vivid_description = raw_d
#         else: # 'full'
#             instruction = raw_q
#             vivid_description = raw_d

#         return image_new, torch.tensor(label, dtype=torch.float32), instruction, vivid_description

#     def __len__(self):
#         return len(self.imgs)
#---------------------------------------------------------------------------------------------------------------------------------------------
#可能是还需要验证一下
# #coding:utf-8
# from PIL import Image
# import torch.utils.data as data
# import numpy as np
# import os
# import torchvision.transforms as transforms
# import torch
# import json
# import random

# # CelebA 40 属性名 (供参考)
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

# class InstructCelebA(data.Dataset):
#     def __init__(self, attr_dir, mode, img_dir, transform, instruct_json_path, part_dir, ablation='full'):
#         self.ablation = ablation
#         self.mode = mode
#         self.img_dir = img_dir
#         self.transform = transform
        
#         # 1. 定义 partition 映射 (0=Train, 1=Val, 2=Test)
#         if mode == 'train':
#             target_partition = '0'
#         elif mode == 'val':
#             target_partition = '1'
#         elif mode == 'test':
#             target_partition = '2'
#         else:
#             raise ValueError(f"Unknown mode: {mode}")

#         # 2. 读取 partition 文件
#         self.imgs = []
#         with open(part_dir, 'r') as f:
#             lines = f.readlines()
#             for line in lines:
#                 fname, partition = line.strip().split()
#                 if partition == target_partition:
#                     self.imgs.append(fname)
        
#         # 3. 加载属性 (Attr) - 完美对齐基线的 1 和 -1 读取方式
#         self.attr_dict = {}
#         with open(attr_dir, 'r') as f:
#             num_imgs = int(f.readline().strip()) 
#             attr_names = f.readline().strip().split() 
            
#             for line in f:
#                 vals = line.strip().split()
#                 fname = vals[0]
#                 # 完全对齐第一种方式：原封不动读取 1 和 -1
#                 attrs = np.array([int(x) for x in vals[1:]], dtype=np.float32)
#                 self.attr_dict[fname] = attrs

#         # 4. 加载 LLaVA 生成的指令和描述 (JSON)
#         with open(instruct_json_path, 'r') as f:
#             instruct_list = json.load(f)
            
#         self.instruct_dict = {}
#         for item in instruct_list:
#             img_key = os.path.basename(item['image']) 
#             q_val = item['conversations'][0]['value']
#             d_val = item['conversations'][1]['value']
#             self.instruct_dict[img_key] = {'q': q_val, 'd': d_val}

#         # 5. 论文级指令池
#         self.general_prompts = [
#             "Please analyze the facial attributes of this person.",
#             "List all the facial features you can identify from this image.",
#             "Examine this face and describe its key characteristics.",
#             "What can you tell me about this person's appearance?",
#             "Identify the visual traits present in this facial image."
#         ]
        
#         print(f"[{mode.upper()}] Loaded {len(self.imgs)} images from CelebA.")

#     def __getitem__(self, index):
#         img_fname = self.imgs[index]
#         image = pil_loader(os.path.join(self.img_dir, img_fname))
        
#         # ==========================================================
#         # 完全复刻第一种 (基线) 的图像处理逻辑，一字不差
#         # ==========================================================
#         w, h = image.size
#         new_w = int(1.0 * w / h * 224)
#         image = transforms.Resize((224, new_w))(image)
#         image = np.array(image)

#         image_new = np.zeros((224, 224, 3))
#         add_num = (224 - new_w) / 2

#         list1 = image_new[:, int(add_num):int(add_num + new_w), :]
#         image_new = np.resize(image, (list1.shape[0], list1.shape[1], list1.shape[2]))

#         image_new = Image.fromarray(np.uint8(image_new))
#         image_new = image_new.convert('RGB')

#         if self.transform is not None:
#             image_new = self.transform(image_new)
#         # ==========================================================

#         # 获取属性标签
#         label = self.attr_dict[img_fname] 
        
#         # 获取文本指令和描述
#         raw_data = self.instruct_dict.get(img_fname, {})
#         raw_d = raw_data.get('d', "A photo of a face.") 
        
#         if self.mode == 'train':
#             raw_q = random.choice(self.general_prompts)
#         else:
#             raw_q = "Face Attribute: Identify specific facial attributes."

#         # 消融实验逻辑 (Ablation)
#         if self.ablation == 'only_q':
#             instruction = raw_q
#             vivid_description = "" 
#         elif self.ablation == 'only_d':
#             instruction = ""
#             vivid_description = raw_d
#         else: 
#             instruction = raw_q
#             vivid_description = raw_d

#         return image_new, torch.tensor(label, dtype=torch.float32), instruction, vivid_description

#     def __len__(self):
#         return len(self.imgs)


#-----------------------------------------------------------------------------------------------------
#第一版少样本
#coding:utf-8
from PIL import Image
import torch.utils.data as data
import numpy as np
import os
import torchvision.transforms as transforms
import torch
import json
import random

# CelebA 40 属性名 (供参考)
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

class InstructCelebA(data.Dataset):
    def __init__(self, attr_dir, mode, img_dir, transform, instruct_json_path, part_dir, ablation='full', fewshot_rate=0.002, seed=42):
        """
        attr_dir: list_attr_celeba.txt 的路径
        part_dir: list_eval_partition.txt 的路径 (CelebA 专用划分文件)
        mode: 'train', 'val', or 'test'
        fewshot_rate: 少样本比例，例如 0.01 代表使用 1% 的训练数据。默认 1.0 (100%)
        seed: 随机种子，确保每次采样的少样本数据一致
        """
        self.ablation = ablation
        self.mode = mode
        self.img_dir = img_dir
        self.transform = transform
        
        # 1. 定义 partition 映射
        # CelebA 标准: 0=Train, 1=Val, 2=Test
        if mode == 'train':
            target_partition = '0'
        elif mode == 'val':
            target_partition = '1'
        elif mode == 'test':
            target_partition = '2'
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # 2. 读取 partition 文件，确定哪些图片属于当前 mode
        self.imgs = []
        with open(part_dir, 'r') as f:
            lines = f.readlines()
            for line in lines:
                fname, partition = line.strip().split()
                if partition == target_partition:
                    self.imgs.append(fname)
        
        # =========================================================
        # 3. 新增：少样本 (Few-Shot) 数据采样逻辑
        # =========================================================
        if self.mode == 'train' and fewshot_rate < 1.0:
            random.seed(seed)  # 必须固定种子，保证实验可重复
            # 计算需要采样的数量，最少保留 1 张
            num_samples = max(1, int(len(self.imgs) * fewshot_rate))
            self.imgs = random.sample(self.imgs, num_samples)
            print(f"[{mode.upper()} FEW-SHOT] Randomly sampled {num_samples} images (Rate: {fewshot_rate*100}%).")
        # =========================================================

        # 4. 加载属性 (Attr)
        self.attr_dict = {}
        with open(attr_dir, 'r') as f:
            num_imgs = int(f.readline().strip()) # 第一行是数量
            attr_names = f.readline().strip().split() # 第二行是属性名
            
            for line in f:
                vals = line.strip().split()
                fname = vals[0]
                # 属性值是 1 或 -1，需要转成 0/1
                attrs = np.array([1 if int(x) == 1 else 0 for x in vals[1:]], dtype=np.float32)
                self.attr_dict[fname] = attrs

        # 5. 加载 LLaVA 生成的指令和描述 (JSON)
        with open(instruct_json_path, 'r') as f:
            instruct_list = json.load(f)
            
        self.instruct_dict = {}
        for item in instruct_list:
            img_key = os.path.basename(item['image']) 
            q_val = item['conversations'][0]['value']
            d_val = item['conversations'][1]['value']
            
            self.instruct_dict[img_key] = {
                'q': q_val,
                'd': d_val
            }

        # 6. 论文级指令池 (Instruction Pool)
        self.general_prompts = [
            "Please analyze the facial attributes of this person.",
            "List all the facial features you can identify from this image.",
            "Examine this face and describe its key characteristics.",
            "What can you tell me about this person's appearance?",
            "Identify the visual traits present in this facial image."
        ]
        
        print(f"[{mode.upper()}] Final loaded {len(self.imgs)} images for CelebA.")

    def __getitem__(self, index):
        img_fname = self.imgs[index]
        image = pil_loader(os.path.join(self.img_dir, img_fname))
        
        # 1. 图像预处理 (保持你的 Padding 逻辑)
        w, h = image.size
        new_w = int(1.0 * w / h * 224)
        image = transforms.Resize((224, new_w))(image)
        image_np = np.array(image)
        
        # 创建 224x224 画布
        image_new = np.zeros((224, 224, 3), dtype=np.uint8)
        add_num = int((224 - new_w) / 2)
        
        # 防止越界 (robustness)
        if image_np.ndim == 2: # 灰度图处理
            image_np = np.stack((image_np,)*3, axis=-1)
        
        valid_w = min(new_w, 224)
        image_new[:, add_num : add_num + valid_w, :] = image_np[:, :valid_w, :]
        
        image_new = Image.fromarray(image_new).convert('RGB')
        
        if self.transform: 
            image_new = self.transform(image_new)

        # 2. 获取属性标签
        label = self.attr_dict[img_fname] 
        
        # 3. 获取文本指令和描述
        raw_data = self.instruct_dict.get(img_fname, {})
        raw_d = raw_data.get('d', "A photo of a face.") 
        
        # 训练时随机指令，测试时固定指令
        if self.mode == 'train':
            raw_q = random.choice(self.general_prompts)
        else:
            raw_q = "Face Attribute: Identify specific facial attributes."

        # 4. 消融实验逻辑 (Ablation)
        if self.ablation == 'only_q':
            instruction = raw_q
            vivid_description = "" 
        elif self.ablation == 'only_d':
            instruction = ""
            vivid_description = raw_d
        else: # 'full'
            instruction = raw_q
            vivid_description = raw_d

        return image_new, torch.tensor(label, dtype=torch.float32), instruction, vivid_description

    def __len__(self):
        return len(self.imgs)


# # # #---------------------------------------------------------------------------------
# #官方少样本
# # coding:utf-8
# from PIL import Image
# import torch.utils.data as data
# import numpy as np
# import os
# import torch
# import json
# import random

# def pil_loader(path):
#     with open(path, 'rb') as f:
#         img = Image.open(f)
#         return img.convert('RGB')

# class InstructCelebA(data.Dataset):
#     def __init__(self, attr_dir, mode, img_dir, transform, instruct_json_path, part_dir, ablation='full', few_shot_ratio=0.002):
#         self.ablation = ablation
#         self.mode = mode
#         self.img_dir = img_dir
#         self.transform = transform
#         self.few_shot_ratio = few_shot_ratio

#         # 1. 解析 partition 划分 (0=Train, 1=Val, 2=Test)
#         if mode == 'train':
#             target_partition = '0'
#         elif mode == 'val':
#             target_partition = '1'
#         elif mode == 'test':
#             target_partition = '2'
#         else:
#             raise ValueError(f"Unknown mode: {mode}")

#         self.imgs = []
#         with open(part_dir, 'r') as f:
#             for line in f:
#                 vals = line.strip().split()
#                 if len(vals) == 2:
#                     fname, partition = vals
#                     if partition == target_partition:
#                         self.imgs.append(fname)

#         # 2. 少样本采样 (Few-Shot) 
#         if self.mode == 'train' and 0 < self.few_shot_ratio < 1.0:
#             random.seed(1234) # 固定种子，保证实验完全可复现
#             num_samples = max(1, int(len(self.imgs) * self.few_shot_ratio))
#             self.imgs = random.sample(self.imgs, num_samples)
#             print(f"!!! Few-shot 模式: 已随机抽取 {self.few_shot_ratio*100}% 数据，剩余 {len(self.imgs)} 张")
#         else:
#             print(f"[{mode.upper()}] 正在加载 {len(self.imgs)} 张图片。")

#         # 3. 解析属性 (对齐官方逻辑: 从 {-1, 1} 映射到 {0, 1})
#         self.attr_dict = {}
#         with open(attr_dir, 'r') as f:
#             num_imgs = int(f.readline().strip())
#             attr_names = f.readline().strip().split()
#             for line in f:
#                 vals = line.strip().split()
#                 if len(vals) > 1:
#                     fname = vals[0]
#                     # 提取数值并利用 (x+1)/2 巧妙地将 -1 变 0, 1 还是 1
#                     attrs = np.array(vals[1:], dtype=np.float32)
#                     attrs = (attrs + 1) / 2.0 
#                     self.attr_dict[fname] = attrs

#         # 4. 指令与描述加载 (LLaVA JSON)
#         self.instruct_dict = {}
#         if instruct_json_path and os.path.exists(instruct_json_path):
#             with open(instruct_json_path, 'r') as f:
#                 instruct_list = json.load(f)
#             for item in instruct_list:
#                 img_key = os.path.basename(item['image'])
#                 self.instruct_dict[img_key] = {
#                     'q': item['conversations'][0]['value'],
#                     'd': item['conversations'][1]['value']
#                 }

#         self.general_prompts = [
#             "Please analyze the facial attributes of this person.",
#             "List all the facial features you can identify from this image.",
#             "Examine this face and describe its key characteristics.",
#             "What can you tell me about this person's appearance?",
#             "Identify the visual traits present in this facial image."
#         ]

#     def __getitem__(self, index):
#         img_fname = self.imgs[index]
        
#         # 1. 纯粹地读取原始图片，不做任何多余的 Numpy 或 Resize 操作
#         image = pil_loader(os.path.join(self.img_dir, img_fname))
        
#         # 2. 直接交给主脚本传进来的 transform_train 进行官方标准的裁剪和归一化
#         if self.transform is not None: 
#             image_new = self.transform(image)
#         else:
#             image_new = image

#         # 3. 获取属性标签 (此时 __init__ 里已经帮你转好 0 和 1 了)
#         label = self.attr_dict[img_fname] 
        
#         # 4. 获取文本指令和描述
#         raw_data = self.instruct_dict.get(img_fname, {})
#         raw_d = raw_data.get('d', "A photo of a face.") 
        
#         if self.mode == 'train':
#             raw_q = random.choice(self.general_prompts)
#         else:
#             raw_q = "Face Attribute: Identify specific facial attributes."

#         if self.ablation == 'only_q': 
#             instruction = raw_q
#             vivid_description = "" 
#         elif self.ablation == 'only_d': 
#             instruction = ""
#             vivid_description = raw_d
#         else: 
#             instruction = raw_q
#             vivid_description = raw_d

#         return image_new, torch.tensor(label, dtype=torch.float32), instruction, vivid_description

#     def __len__(self):
#         return len(self.imgs)