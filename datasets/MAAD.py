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

def make_img(part_dir):
    with open(part_dir) as f:
        lines = f.readlines()
        return lines

class MAAD(data.Dataset):
    def __init__(self, part_dir, attr_dir,img_dir, transform):
        self.img = make_img(part_dir)
        self.attr = np.zeros((3308040, 47))
        self.nameToAttr={}
        with open(attr_dir) as f:
            f.readline()
            lines = f.readlines()
            id = 0
            for line in lines:
                vals = line.split(",")
                print(len(vals))
                for j in range(0,47):
                    self.attr[id, j] = int(vals[j+2])
                self.nameToAttr[vals[0]]=id
                if id%10==0:
                    print(id)
                id += 1

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
        try:
            image = pil_loader(os.path.join(self.img_dir, self.img[index][:-1]))
        except:
            print("===============================图片读取出现错误===================================")
            import random
            return self.__getitem__(random.randint(0,len(self.img)-1))


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
        try:
            id = self.nameToAttr[self.img[index][:-1]]
        except:
            print("===============================标签读取错误===================================")
            import random
            return self.__getitem__(random.randint(0, len(self.img) - 1))
        return image_new, self.attr[id, :]


    def __len__(self):
        return self.length
def getTrainLists(dir="/data2/Datasets/lmx_data/VGG-Face2/"+"data/test/"):
    import os
    lists = os.listdir(dir)
    with open(os.path.join("/home/Zhuxueyan/code/Dataset/MAAD/", f"train_test.txt"), 'a') as f:
        for list in lists:
            names = os.listdir(dir + f"{list}")
            for name in names:
                f.write(f'{list}/{name}\n')
            print(list)
if __name__ == '__main__':
    # getTrainLists()
    try:
        x = pil_loader("/data2/Datasets/lmx_data/VGG-Face2/data/train/n004218/0381_01.jpg")
    except:
        print("hahhfahhfhdsjf")
    from torchvision import transforms
    import torch

    transform=transforms.Compose([transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    # data_dir="/home/Leimingxuan/DataSet/maad/VGG-Face2/"
    data_dir="//dataset/VGG-Face2/"

    #test
    val_dataset = MAAD('/home/Zhuxueyan/code/Dataset/MAAD/test_list.txt', '/home/Zhuxueyan/code/vision_transformer/datasets/MAAD_Face.csv',
                     data_dir + 'data/test/', transform)

    #train
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=64,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=8)
    for idx, (images, labels) in enumerate(val_loader):
        print("++++++++++++++++++++++++++++++++++++++++++++")
        print(idx)
    train_dataset = MAAD('/home/Zhuxueyan/code/Dataset/MAAD/train_list.txt', '/home/Zhuxueyan/code/vision_transformer/datasets/MAAD_Face.csv',
                     data_dir + 'data/train/', transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=64,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=8)

