import os
import torch
import torch.utils.data as data
from PIL import Image
import matplotlib.pyplot as plt
import torch.utils
import torchvision
import numpy as np
import scipy.io as scio
import torchvision.transforms as transforms
import cv2

def default_loader(path):
    return cv2.imread(path)


class myImageFloder(data.Dataset):
    def __init__(self, root, label, transform=None, target_transform=None, loader=default_loader, mode='train'):

        fn = scio.loadmat(label)
        imgs = []
        if mode == 'train':
            testlabel = fn['train_label']
            testimg = fn['train_images_name']
        if mode == 'test':
            testlabel = fn['test_label']
            testimg = fn['test_images_name']
        if mode == 'validate':
            testlabel = fn['val_label']
            testimg = fn['val_images_name']
        count = 0
        for name in testimg:
            # print name[0][0]
            if os.path.isfile(os.path.join(root, name[0][0])):
                imgs.append((name[0][0], testlabel[count]))
            count = count + 1

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.classes = fn['attributes']
        pass
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        label =  np.delete(label,(12,15,16,17,18,19,20,21,25),0)
        idx = [0,7,8,9,10,11,1,2,3,4,5,6,12,13,14,15,16]
        label = label[idx]#重新排列
        img = self.loader(os.path.join(self.root, fn))
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.Tensor(label) 

    def __len__(self):
        return len(self.imgs)

    def getName(self):
        return self.classes


def imshow(imgs):
    grid = torchvision.utils.make_grid(imgs)
    plt.imshow(grid.numpy().transpose(1, 2, 0))
    plt.title("bat")
    plt.show()


if __name__ == '__main__':
    mytransform = transforms.Compose([

        transforms.Resize(256),
        transforms.ToTensor(),  # mmb
    ]
    )

    # torch.utils.data.DataLoader
    set = myImageFloder(root="./data/PA-100K/release_data/release_data",
                        label="./data/PA-100K/annotation/annotation.mat", transform=mytransform)
    imgLoader = torch.utils.data.DataLoader(
        set,
        batch_size=1, shuffle=False, num_workers=2)

    len(set)
    dataiter = iter(imgLoader)
    images, labels = dataiter.next()
    imshow(images)