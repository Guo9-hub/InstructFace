#coding:utf-8
from PIL import Image
import torch.utils.data as data
import numpy as np
import os
import torchvision.transforms as transforms

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def make_img(part_dir, partition):
    img = []
    with open(part_dir) as f:
        lines = f.readlines()
        for line in lines:
            pic_dir, num = line.split()
            if num == partition: 
                img.append(pic_dir)
    return img
def get_ids(ids_dir=None):
    id=np.zeros((202600, 1))
    with open(ids_dir) as f:
        lines=f.readlines()
        for i in range(len(lines)):
            id[i+1]=lines[i].split()[1]
    return id

class CelebAIdentity(data.Dataset):
    def __init__(self, part_dir, attr_dir, partition, img_dir, transform,id_dir="/home/Zhuxueyan/code/Dataset/CelebA/Anno/identity_CelebA.txt"):
        self.attr = np.zeros((202600, 40))
        with open(attr_dir) as f:
            f.readline()
            f.readline()
            lines = f.readlines()
            id = 0
            for line in lines:
                vals = line.split()
                id += 1
                for j in range(40):
                    self.attr[id, j] = int(vals[j+1])
        self.identity=get_ids(id_dir)


        self.img= make_img(part_dir, partition)
        self.length = len(self.img)
        self.transform = transform
        self.img_dir = img_dir

    def __getitem__(self, index):



        # 画出变化后的bounding box观测是否正确
        # figure(2)
        # draw_fll(image, self.fll[index, :])
        # imshow(image)
        #
        # figure(3)
        # draw_bounding_box(image, self.box[index, :])
        # imshow(image)
        # show()

        image = pil_loader(os.path.join(self.img_dir, self.img[index]))
        # print image
        w, h = image.size
        # m = min(w, h)
        new_w = int(1.0 * w / h * 224)
        image = transforms.Resize((224, new_w))(image)
        image = np.array(image)

        
        # image : [H , W , C] 
        # 作者的目的应该是将任意大小的image做一个数据拷贝，并且实现图片大小的归一化
        # 但是这个image用广播机制是不满足要求的（查广播机制的语法）
        # 新思路： 首先用numpy中的归一化（np.resize）将Image归一化为指定大小（打印分析）；然后赋值给image_new
        # resize(a, b , c)   a = 224,b = ..., c = 3
        # image_new = np.resize(a, b, c)


        image_new = np.zeros((224, 224, 3))
        add_num = (224 - new_w)/2

        # image_new[:, int(add_num):int(add_num+new_w), :] = image.copy()        #如果是其他图片的大小

        list1 = image_new[:,int(add_num):int(add_num+new_w), :]
        image_new = np.resize(image,(list1.shape[0],list1.shape[1],list1.shape[2]))

        image_new = Image.fromarray(np.uint8(image_new))
        image_new = image_new.convert('RGB')




        # image_new.show()


        # plt.imshow(image_new)
        # plt.show()
        # print image.shape
        if self.transform is not None:
            image_new = self.transform(image_new)    #这里的输入好像必须是image类型
        id = int(self.img[index].split('.')[0])
        return image_new, (self.attr[id, :],self.identity[id,:])


    def __len__(self):
        return self.length

