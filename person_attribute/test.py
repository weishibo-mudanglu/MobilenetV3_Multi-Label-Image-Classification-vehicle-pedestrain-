import os
import torch
import torch.utils.data as data
import torch.utils
import torch.nn as nn
from dataset import myImageFloder
import torchvision.transforms as transforms
from torch.autograd import Variable
import pandas as pd
from model.moblenet_v3_CBAM import MobileNetV3
# from visdom import Visdom
import numpy as np
import cvtransforms

batch_size = 5
attribute = [u'Female', u'Hat',u'Glasses', u'HandBag', u'ShoulderBag', 
            u'HoldObjectsInFront',u'AgeOver60', u'Age18-60', u'AgeLess18', 
            u'Front', u'Side', u'Back',u'ShortSleeve', u'LongSleeve', 
            u'Trousers', u'Shorts',u'Skirt&Dress']
def binary_np(input_np):
    index_max = np.argmax(input_np,axis=1)
    temp = np.zeros(input_np.shape)
    for i in range(len(index_max)):
        temp[i,index_max[i]] = temp[i,index_max[i]] + 1
    return temp
def pricision_recall(TP,FN,TN,FP):
    pricision = TP/(TP+TN)
    recall = TP/(TP+FP)
    accuracy = (TP+FN)/10000
    f1_score = 2*(recall*pricision)/(recall+pricision)
    return accuracy,pricision,recall,f1_score

mytransform = cvtransforms.Compose([

    # transforms.RandomHorizontalFlip(),
    cvtransforms.Resize((224, 128)),
    # transforms.RandomCrop(299),
    # transforms.Resize((299,299)),       #TODO:maybe need to change1
    cvtransforms.ToTensor(),  # mmb,
    cvtransforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                            std  = [ 0.229, 0.224, 0.225 ])
]
)

set = myImageFloder(root=r"D:\work\datas\PA-100K-20210716T100412Z-001\PA-100K\release_data\release_data", label=r"D:\work\datas\PA-100K-20210716T100412Z-001\PA-100K\annotation.mat",
                    transform=mytransform,mode="test")
imgLoader = torch.utils.data.DataLoader(
    set,
    batch_size=batch_size, shuffle=True, num_workers=0)

net = MobileNetV3(type='large', num_classes=17)
net.load_state_dict(torch.load('./checkpoint/checkpoint_epoch_24'))
net.cuda()
net.eval()
record_matrix = np.zeros((4,17))
score_matrix = np.zeros((17,4))
Accuracy = 0
for batch, data in enumerate(imgLoader, 0):
    # get the inputs
    inputs, labels = data

    inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

    outputs = net(inputs)

    pre_lable = np.zeros((batch_size,17))
    for i in range(10):
        if(i<6):
            pre_lable[:,i] = torch.max(outputs[i+10].data,1)[1].cpu().numpy().T
        
        elif i ==6:
            pre_lable[:,6:9] = binary_np(outputs[i+10].data.cpu().numpy())
        elif i ==7:
            pre_lable[:,9:12] = binary_np(outputs[i+10].data.cpu().numpy())
        elif i == 8:
            pre_lable[:,12:14] = binary_np(outputs[i+10].data.cpu().numpy())
        elif i ==9:
            pre_lable[:,14:17] = binary_np(outputs[i+10].data.cpu().numpy())

    labels = labels.cpu()
    # outputs = outputs.data.numpy()
    # outputs= np.where(outputs>-1, 1, 0)
    labels = labels.data.numpy()
    for i in range(batch_size):
        if((pre_lable[i,:] == labels[i,:]).all()):
            Accuracy = Accuracy + 1
    for k in range(batch_size):
        for j in range(len(labels[k,:])):
            if(labels[k,j]==1):
                if(pre_lable[k,j]==labels[k,j]):
                    record_matrix[0,j] = record_matrix[0,j] + 1
                else:
                    record_matrix[3,j] = record_matrix[3,j] + 1
            else:
                if(pre_lable[k,j]==labels[k,j]):
                    record_matrix[1,j] = record_matrix[1,j] + 1
                else:
                    record_matrix[2,j] = record_matrix[2,j] + 1
    print("已完成%.2f%%"%((batch*batch_size)/100))
Acc = Accuracy/10000
for i in range(17):
    temp = record_matrix[:,i]
    accuracy,pricision,recall,f1_score = pricision_recall(record_matrix[0,i],record_matrix[1,i],record_matrix[2,i],record_matrix[3,i])
    score_matrix[i,:] = [accuracy,pricision,recall,f1_score]
    print("||%-20s||accuracy:%.4f     ||pricision:%.4f     ||recall:%.4f     ||f1 score:%.4f     ||"%(attribute[i],accuracy,pricision,recall,f1_score))
score_df = pd.DataFrame(score_matrix)
data_df = pd.DataFrame(record_matrix.astype(int))
score_df.columns = ["accuracy","pricision","recall","f1_score"]
score_df.index = attribute
data_df.columns = attribute
data_df.index = ["TP","FN","TN","FP"]
writer = pd.ExcelWriter('./checkpoint/errors.xls')
data_df.to_excel(writer,'sheet1',float_format='%d')
score_df.to_excel(writer,'sheet2',float_format='%.4f')
writer.save()
print("准确率为%.4f%%"%(Acc*100))
print(record_matrix.astype(int))



