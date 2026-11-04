#coding:utf-8
from PIL import Image
import torch.utils.data as data
import numpy as np
import os
import torchvision.transforms as transforms
import argparse
import torch

# import matplotlib.pyplot as plt
def get_resized_label(img_size, resized_size, label):

    #leftx, lefty, width, hight
    label_copy = label.copy()
    label_copy[::2] = label_copy[::2] * 1.0 * resized_size[0] / img_size[0]
    label_copy[1::2] = label_copy[1::2] * 1.0 * resized_size[1] / img_size[1]
    return label_copy
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
def get_imageNameDict(path_dir='/home/Zhuxueyan/code/Dataset/LFWA/lfw/'):
    dirs = os.listdir(path_dir)
    dict={}
    for i in range(len(dirs)):
        dict[dirs[i]]=i
    return dict

class LFWA_Identity(data.Dataset):
    def __init__(self, attr_dir, mode, img_dir, transform):
        #self.image是一个字符数组 每个元素形如 Miyako_Miyazaki/Miyako_Miyazaki_0001.jpg
        # 参数包括 attr_dir 属性标签文件
        # mode： train 和 test
        self.imageNameDict=get_imageNameDict()
        if mode == 'train':
            #训练集把funnel和原始的图片都加入作为训练
            self.attr = np.zeros((6572*2, 40))
        else:
            self.attr = np.zeros((6571, 40))

        with open(attr_dir) as f:
            f.readline()
            f.readline()
            lines = f.readlines()

            if mode =='train':
                flag = 0
            else:
                flag = 1

            self.imgs = []
            self.imgs_identity=[]
            idx = 0

            for line in lines:
                flag = 1 - flag
                if flag % 2 == 0:
                    continue
                # 每一行有可能有76-79个，其中前2个或者3个是人名，后边依次是每个人图片的数量和73个属性

                #把图像的路径加入到self.imgs

                vals = line.split()
                len_name = len(vals) - 74 #73个属性加上 1个num(这个人的第几张图片)

                person_name = vals[0]
                for i in range(1, len_name):
                    person_name += '_' + vals[i]
                img_name = person_name + '/' + person_name + '_' + str(vals[len_name]).zfill(4)+'.jpg'

                self.imgs.append(img_name)
                self.imgs_identity.append(person_name)
                #把剪切过和未剪切的图片都计入训练集防止过拟合
                if mode == 'train':
                    self.imgs.append('/home/Zhuxueyan/code/Dataset/LFWA/lfw/' + img_name)
                    self.imgs_identity.append(person_name)

                #把图像的标签加入到self.attr
                index = [65, 36, 56, 60, 13, 30 ,41, 39, 10, 11,
                         21, 12, 35, 20, 49, 15, 47, 59, 61, 69,
                         1,  43, 17, 37, 46, 51, 64, 40, 29, 62,
                         31, 18, 28, 27, 71, 50, 67, 73, 72, 7]

                #如果是列表好像不支持temp[temp>0] = 1 的操作
                temp = np.array([float(s) for s in vals[-74:]])   # temp 形如 [num, 0.001, -0.221, ...]

                temp[temp >= 0] = 1
                temp[temp < 0] = 0                     # temp 形如[num=1, -1, 1, -1...]
                # print temp
                self.attr[idx, :] = [temp[i] for i in index]



                # print len(temp)
                # print temp
                #
                # find = [1.1661, -0.1145, 0.7793, -0.7256, -0.9735, -0.2575, -1.1202, 0.0689,
                #  -0.7196, -0.6324, -1.6557, 0.4648, -0.9226, -0.2190, -0.7053, -1.2967,
                #  -0.4987, -0.7693, -1.8206, -1.2999, 1.5683, -1.3080, -0.6847, 1.4612,
                #  0.8324, 0.3742, 0.3617, 1.2679, -0.1611, -2.0730, -0.0888, -0.8650,
                #  -0.4977, -0.3857, -1.1449, -0.5157, -1.1400, -0.8266, 0.6940, -0.8356]
                # print find
                # print len(find)
                # for i in range(len(find)):
                #     for j in range(len(temp)):
                #         if abs(find[i] - temp[j]) < 0.0001:
                #             print j
                # print temp[56]
                # print temp[57]
                # exit(0)
                #由于56和57是男性女性有魅力，所以只要有一个为1就认为attractive属性为1

                if temp[57] > 0:
                    self.attr[idx, 2] = 1
                if mode == 'train':
                    self.attr[idx+1, :] = self.attr[idx, :]
                    idx += 2  #1
                else:
                    idx += 1

        # 只要用到训练集的fll标注即可，这里为了存取方便，把fll初始值赋为-1,表示图片中没有识别人脸
        if mode == 'train':
            self.fll = np.ones((6572*2, 144))*(-1)
        else:
            self.fll = np.ones((6571, 144))*(-1)

        path = os.path.dirname(__file__)
        if mode == 'train':
            flag = 0
        else:
            flag = 1

        # print(path)
        # 增大数据集后landmarks的标签txt需要重新选取，尚未完成
        # with open(path + '/landmarks_lfw.txt') as f2:
        #     idx = 0
        #     lines = f2.readlines()
        #     for line in lines:
        #         flag = 1 - flag
        #         if flag % 2 == 0:
        #             continue
        #
        #         vals = line.split()
        #         img_path = vals[0]
        #         # print img_path
        #         for j in range(144):
        #             self.fll[idx, j] = int(vals[j + 1])
        #         idx += 1



        self.length = len(self.imgs)
        self.transform = transform
        self.img_dir = img_dir
        # print self.imgs[1]

    def __getitem__(self, index):
        image = pil_loader(os.path.join(self.img_dir, self.imgs[index]))
        image_identity=self.imageNameDict[self.imgs_identity[index]]


        # print("id={}".format(id))
        # print("index ={}".format(index)) 训练集的时候两者相差１测试集的时候两者
        # print index , type(self.attr[id,:])
        w, h = image.size
        # m = min(w, h)
        new_w = int(1.0 * w / h * 224)
        image = transforms.Resize((224, new_w))(image)
        image = np.array(image)

        image_new = np.zeros((224, 224, 3))
        add_num = (224 - new_w) / 2

        list1 = image_new[:,int(add_num):int(add_num+new_w), :]
        image_new = np.resize(image,(list1.shape[0],list1.shape[1],list1.shape[2]))

        image_new = Image.fromarray(np.uint8(image_new))
        image_new = image_new.convert('RGB')

        # image_new[:, add_num:add_num + new_w, :] = image.copy()  # 如果是其他图片的大小
        # image_new = Image.fromarray(np.uint8(image_new))

        # image_new.show()

        # plt.imshow(image_new)
        # plt.show()
        # print image.shape
        if self.transform is not None:
            image_new = self.transform(image_new)  # 这里的输入好像必须是image类型


        # resize操作改变的fll标签
        fll = get_resized_label((w, h), (1, 1), self.fll[index, :])  # 函数里边是copy数组,避免改变原来的self,fll下个epoch有错误
        # print fll
        # 在两侧添加像素改变的fll标签, 由于celebA 218*178因此只有横坐标的标签需要变化
        # print fll

        # print("id={}".format(id))
        # print("index ={}".format(index)) 训练集的时候两者相差１测试集的时候两者
        # print index , type(self.attr[id,:])
        return image_new, (self.attr[index, :],image_identity)


    def __len__(self):
        return self.length

if __name__ == '__main__':
    transform_train = transforms.Compose([
        # 　取消固定大小的输入
        # transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    transform_val = transforms.Compose([
        # 取消固定大小的输入
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--batchSize', type=int, default=64)
    parser.add_argument('--nepoch', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gpu', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    opt = parser.parse_args()
    print(opt)

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    trainset = LFWA('/home/Zhuxueyan/code/Dataset/LFWA/lfw_attributes.txt', 'train',
                    '/home/Zhuxueyan/code/Dataset/LFWA/lfw', transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers)

    print(len(trainset))
    print(len(trainloader))
    print(trainset[0])
    print(trainset[1])

    testset = LFWA('/home/Zhuxueyan/code/Dataset/LFWA/lfw_attributes.txt', 'test', '/home/Zhuxueyan/code/Dataset/LFWA/lfw',
                   transform_val)
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers)

    # transform_train = transforms.Compose([
    #     # 　取消固定大小的输入
    #     # transforms.Resize((224, 224)),
    #     # transforms.CenterCrop((224, 224)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])
    #
    # transform_val = transforms.Compose([
    #     # 取消固定大小的输入
    #     # transforms.Resize((224, 224)),
    #     # transforms.CenterCrop((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--workers', type=int, default=2)
    # parser.add_argument('--batchSize', type=int, default=64)
    # parser.add_argument('--nepoch', type=int, default=15)
    # parser.add_argument('--lr', type=float, default=0.001)
    # parser.add_argument('--gpu', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    # opt = parser.parse_args()
    # print(opt)
    #
    # os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    #
    # img_root = '/home/maolongbiao/DataSet/CelebA/'
    #
    # # trainset = CelebA(img_root + 'Eval/list_eval_partition.txt', img_root + 'Anno/list_attr_celeba.txt', '0',
    # #                   img_root + 'img_align_celeba/', transform_train)
    # # trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers)
    # #
    # # valset = CelebA(img_root + 'Eval/list_eval_partition.txt', img_root + 'Anno/list_attr_celeba.txt', '1',
    # #                 img_root + 'img_align_celeba/', transform_val)
    # # valloader = torch.utils.data.DataLoader(valset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers)
    #
    # testset = CelebA(img_root + 'Eval/list_eval_partition.txt', img_root + 'Anno/list_attr_celeba.txt', '0',
    #                  img_root + 'img_align_celeba/', transform_val)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batchSize, shuffle=False, num_workers=opt.workers)

    # print testset[100043]
    # print testset[10394]
    # print testset[1]
    # print testset[2]
    # print testset[3]
    # print testset[4]
    # print testset[5]
    # print testset[6]

    # print(len(trainset))
    # print(len(trainloader))
    # print(len(valset))
    # print(len(valloader))
    # print(len(testset))
    # print(len(testloader))
