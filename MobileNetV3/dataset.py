import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from PIL import Image
import os
import numpy as np
import cv2
import random
from my_utils import RandomErasing
kernel = np.ones((5, 5), dtype=np.uint8)
mean = [ 0.485, 0.456, 0.406 ]
std  = [ 0.229, 0.224, 0.225 ]
def opencv_transform(img,re_size):
    img = cv2.resize(img,re_size)
    img = np.float32(img)
    cv2.normalize(img, dst=img, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX)
    B,G,R = cv2.split(img)
    R = (R-mean[0])/std[0]
    G = (G-mean[1])/std[1]
    B = (B-mean[2])/std[2]
    img = cv2.merge([R,G,B])
    return img
    
#以torch.utils.data.Dataset为基类创建MyDataset
class MyDataset(Dataset):
    #stpe1:初始化
    def __init__(self, txt,path, transform=None, target_transform=None,re_size = (64,64)):
        fh = open(txt, 'r')#打开标签文件
        imgs = []#创建列表，装东西
        for line in fh:#遍历标签文件每行
            line = line.rstrip()#删除字符串末尾的空格
            words = line.split()#通过空格分割字符串，变成列表
            # templist = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],dtype="float32")
            # templist[int(words[1])-1] = templist[int(words[1])-1]+1
            # templist[int(words[2])+1] = templist[int(words[2])+1]+1
            # templist[int(words[3])+14] = templist[int(words[3])+14]+1

            # imgs.append((words[0],templist))#把图片名words[0]，标签int(words[1])放到imgs里
            templist = np.array([int(words[1])-1,int(words[2])-1,int(words[3])-1,int(words[4])],dtype="float32")
            imgs.append((words[0],templist))
        self.re_size = re_size
        self.path = path
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.RE = RandomErasing(p=0.5)
 
    def __getitem__(self, index):#检索函数
        fn, label = self.imgs[index]#读取文件名、标签
        label = np.array(label)
        complete_path = os.path.join(self.path,fn)
        if self.transform is not None:
            img = cv2.imread(complete_path)#通过PIL.Image读取图片
            img = self.RE(img)
            img = self.transform(img)
        else:
            # #加载图像并转换为opencv格式
            img = cv2.imread(complete_path)
            #图像预处理操作
            img = opencv_transform(img,self.re_size)
            #转换预处理后的图像为Image格式
            img = np.asarray(img)
        return img,label
 
    def __len__(self):
        return len(self.imgs)
