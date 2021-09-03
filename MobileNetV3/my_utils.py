import os
import re
import random
import math
import torch
import torch.nn as nn
import numpy as np
import cv2
INTER_MODE = {'NEAREST': cv2.INTER_NEAREST, 'BILINEAR': cv2.INTER_LINEAR, 'BICUBIC': cv2.INTER_CUBIC}
class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self,num_classes, epsilon,gamma,weight=None):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.softmax = nn.Softmax(dim=1)
        self.gamma = gamma
        self.weight = weight
    def forward(self, inputs, targets):
        probs = self.softmax(inputs)
        log_probs = torch.log(probs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        # loss = (-targets * log_probs).mean(0).sum()
        if(self.weight==None):
            loss = (-(torch.pow((1-probs), self.gamma))*log_probs).mean(0).sum()#mean(0)按列求均值
        else:
            weight = self.weight.expand(inputs.shape)
            loss = (-weight*(torch.pow((1-probs), self.gamma))*log_probs).mean(0).sum()#mean(0)按列求均值
        return loss


class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss
def mixup_data(x, y, alpha=0.2, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
 
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
 
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
 
 
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class RandomErasing:
    """Random erasing the an rectangle region in Image.
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    Args:
        sl: min erasing area region 
        sh: max erasing area region
        r1: min aspect ratio range of earsing region
        p: probability of performing random erasing
    """
 
    def __init__(self, p=0.5, sl=0.02, sh=0.4, r1=0.3):
 
        self.p = p
        self.s = (sl, sh)
        self.r = (r1, 1/r1)
    
 
    def __call__(self, img):
        """
        perform random erasing
        Args:
            img: opencv numpy array in form of [w, h, c] range 
                 from [0, 255]
        Returns:
            erased img
        """
        assert len(img.shape) == 3, 'image should be a 3 dimension numpy array'
        if random.random() > self.p:
            return img
        else:
            while True:
                Se = random.uniform(*self.s) * img.shape[0] * img.shape[1]
                re = random.uniform(*self.r) 
 
                He = int(round(math.sqrt(Se * re)))
                We = int(round(math.sqrt(Se / re)))
 
                xe = random.randint(0, img.shape[1])
                ye = random.randint(0, img.shape[0])
 
                if xe + We <= img.shape[1] and ye + He <= img.shape[0]:
                    img[ye : ye + He, xe : xe + We, :] = np.random.randint(low=0, high=255, size=(He, We, img.shape[2]))
 
                    return img

if __name__ == "__main__":
    img = cv2.imread("test.jpg")
    RE = RandomErasing(p=0.5)
    for i in range(20):
        img1 = RE(img.copy())
        cv2.imshow("test", img1)
        cv2.waitKey(1000)
