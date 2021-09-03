import os
import torch
import platform
import torch.utils.data as data
import torch.utils
import torch.nn as nn
from dataset_txt import myImageFloder
import torchvision.transforms as transforms
from torch.autograd import Variable
from draw_data import  draw_loss
from model.moblenet_v3_CBAM import MobileNetV3
import torchvision
import pandas as pd
# from visdom import Visdom
import numpy as np
from utils import CenterLoss,CrossEntropyLabelSmooth
import cvtransforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def checkpoint(epoch,net):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    path = "./checkpoint/checkpoint_epoch_{}".format(epoch)
    torch.save(net.state_dict(), path)
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

weight = torch.Tensor([1.7226262226969686, 2.6802565029531618, 1.0682133644154836, 2.580801475214588,
                    1.8984257687918218, 2.046590013290684, 1.9017984669155032, 2.6014006200502586,
                    2.272458988404639, 2.2625669787021203, 2.245380512162444, 2.3452980639899033,
                    1.5128949487853383, 1.7967419553099035, 1.3377813061118478, 2.284449325734624,
                    2.417810793601295])#去除了几个样本非常不均衡的类别[13，16，17，18，19，20，21，22，26]
if(platform.system()=='Windows'):
    path_data = r"D:\work\datas\PA-100K-20210716T100412Z-001\PA-100K\release_data"
    # path_label= r"D:\work\datas\PA-100K-20210716T100412Z-001\PA-100K\annotation.mat"
    path_label = r"D:\work\datas\PA-100K-20210716T100412Z-001"
elif(platform.system()=='Linux'):
    path_data = r"/home/Glasssix-LQJ/wei_data/PA-100K/release_data"
    path_label= r"/home/Glasssix-LQJ/wei_data/PA-100K/release_data"
else:
    print("地址变更错误，当前操作系统为%s"%(platform.system()))
def train():
    FSCORE = 0
    BATCH_SIZE = 5
    mytransform = cvtransforms.Compose([
        cvtransforms.RandomHorizontalFlip(),#随机翻转
        cvtransforms.Resize((224, 128)),#顺序为h,w
        cvtransforms.ColorJitter(brightness=0.3, contrast=0.3,saturation=0.3,hue=0.3),#含有颜色标签，使用色相或则色调偏移不利于检测
        cvtransforms.ToTensor(),  # mmb,
        cvtransforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],std  = [ 0.229, 0.224, 0.225 ])
    ])
    set = myImageFloder(root=path_data, label=path_label,transform=mytransform,folders=['train_name_label.txt','Market_1501.txt','DukeMTMC.txt'])
    imgLoader = torch.utils.data.DataLoader(set,batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    net = MobileNetV3(type='large', num_classes=17)
    net.to(device)
    net.train()
    #criterion_2 = CrossEntropyLabelSmooth(5,1)
    criterion_2 = nn.CrossEntropyLoss()
    criterion_2.to(device)
    centerloss = CenterLoss(10,10)
    centerloss.to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.001)
    running_loss = 0.0
    loss_record = []
    EPOCHS = 100
    for epoch in range(EPOCHS):
        for i, data in enumerate(imgLoader, 0):
            # get the inputs
            inputs, labels = data
            #-----------------------根据标记对数据分类-----------------------#
            index_0 = torch.nonzero((labels[:,17]==0),as_tuple=False).squeeze(-1)#全标注
            index_1 = torch.nonzero((labels[:,17]==1),as_tuple=False).squeeze(-1)#Market_1501(性别，年龄，背包，袖子，帽子)
            index_2 = torch.nonzero((labels[:,17]==2 ),as_tuple=False).squeeze(-1)#DukeMTMC(性别，背包，袖子，帽子)
            
            #----------------------依照分类组合求损失-----------------------------#
            #有对应标签的类组合
            index_female         = torch.cat((index_0,index_1,index_2))
            index_hat            = torch.cat((index_0,index_1,index_2))
            index_glass          = index_0
            index_handbag        = torch.cat((index_0,index_1,index_2))
            index_shoulderbag    = torch.cat((index_0,index_1,index_2))
            index_carry          = index_0
            index_age            = torch.cat((index_0,index_1))
            index_dirction       = index_0
            index_upclothing     = torch.cat((index_0,index_1,index_2))
            index_downclothing   = index_0
            # wrap them in Variable
            inputs=inputs.to(device)
            labels=labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            # print(outputs)
            label = labels.long()
            if(len(index_female)!=0):
                loss_female         = criterion_2(outputs[10][index_female], label[:,0][index_female])
                center_loss_female  = centerloss(label[:,0][index_female],outputs[0][index_female])
            else:
                loss_female         = 0
                center_loss_female  = 0

            if(len(index_hat)!=0):
                loss_hat            = criterion_2(outputs[11][index_hat], label[:,1][index_hat])
                center_loss_hat     = centerloss(label[:,1][index_hat],outputs[1][index_hat])
            else:
                loss_hat        = 0
                center_loss_hat = 0

            if(len(index_glass)!=0):
                loss_glass          = criterion_2(outputs[12][index_glass], label[:,2][index_glass])
                center_loss_glass   = centerloss(label[:,2][index_glass],outputs[2][index_glass])
            else:
                loss_glass        = 0
                center_loss_glass = 0

            if(len(index_handbag)!=0):
                loss_handbag        = criterion_2(outputs[13][index_handbag], label[:,3][index_handbag])
                center_loss_handbag = centerloss(label[:,3][index_handbag],outputs[3][index_handbag])
            else:
                loss_handbag        = 0
                center_loss_handbag = 0
            
            if(len(index_shoulderbag)!=0):
                loss_shoulderbag         = criterion_2(outputs[14][index_shoulderbag], label[:,4][index_shoulderbag])
                center_loss_shoulderbag  = centerloss(label[:,4][index_shoulderbag],outputs[4][index_shoulderbag])
            else:
                loss_shoulderbag        = 0
                center_loss_shoulderbag = 0

            if(len(index_carry)!=0):
                loss_carry          = criterion_2(outputs[15][index_carry], label[:,5][index_carry])
                center_loss_carry   = centerloss(label[:,5][index_carry],outputs[5][index_carry])
            else:
                loss_carry        = 0
                center_loss_carry = 0
            
            if(len(index_age)!=0):
                loss_age            = criterion_2(outputs[16][index_age], torch.max(label[:,6:9].data,1)[1][index_age])
                center_loss_age     = centerloss(torch.max(label[:,6:9].data,1)[1][index_age],outputs[6][index_age])
            else:
                loss_age        = 0
                center_loss_age = 0

            if(len(index_dirction)!=0):
                loss_dirction        = criterion_2(outputs[17][index_dirction], torch.max(label[:,9:12].data,1)[1][index_dirction])
                center_loss_dirction = centerloss(torch.max(label[:,9:12].data,1)[1][index_dirction],outputs[7][index_dirction])
            else:
                loss_dirction        = 0
                center_loss_dirction = 0
            
            if(len(index_upclothing)!=0):
                loss_upclothing         = criterion_2(outputs[18][index_upclothing], torch.max(label[:,12:14].data,1)[1][index_upclothing])
                center_loss_upclothing  = centerloss(torch.max(label[:,12:14].data,1)[1][index_upclothing],outputs[8][index_upclothing])
            else:
                loss_upclothing        = 0
                center_loss_upclothing = 0

            if(len(index_downclothing)!=0):
                loss_downclothing         = criterion_2(outputs[19][index_downclothing], torch.max(label[:,14:17].data,1)[1][index_downclothing])
                center_loss_downclothing  = centerloss(torch.max(label[:,14:17].data,1)[1][index_downclothing],outputs[9][index_downclothing])
            else:
                loss_downclothing        = 0
                center_loss_downclothing = 0

        
            loss = loss_female+loss_hat+loss_glass+loss_handbag+loss_shoulderbag+loss_carry+loss_age+loss_dirction+loss_upclothing+loss_downclothing  \
                +center_loss_female+center_loss_hat+center_loss_glass+center_loss_handbag+center_loss_shoulderbag+center_loss_carry+center_loss_age+center_loss_dirction+center_loss_upclothing+center_loss_downclothing
            # print(loss)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data
            if i % 100 == 0:  # print every 1000 mini-batches
                print('epoch:%d  i:%d loss: %.6f' % (epoch, i*BATCH_SIZE + 1, running_loss / 100))
                loss_record.append(running_loss)
                running_loss = 0.0
    #         #draw_loss(loss_record,EPOCHS,'loss_record.jpg')
        if epoch % 5 == 0 and epoch !=0:
            val_batch_size = 16
            set = myImageFloder(root=path_data, label=path_label,transform=mytransform,folders=['val_name_label.txt'])
            imgLoader = torch.utils.data.DataLoader(set,batch_size=val_batch_size, shuffle=True, num_workers=0)
            net.eval()

            record_matrix = np.ones((4,17))
            score_matrix = np.zeros((17,4))
            Accuracy = 0
            pre_lable = np.zeros((val_batch_size,17))
            for batch, data in enumerate(imgLoader, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = net(inputs)
                #由于这里是放入固定大小的数组中，所以batchsize一定能被测试集整除，否则最后一个batchsize会报错
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
                labels = labels.data.numpy()
                for i in range(val_batch_size):
                    if((pre_lable[i,:] == labels[i,:-1]).all()):
                        Accuracy = Accuracy + 1
                for k in range(val_batch_size):
                    for j in range(len(labels[k,:-1])):
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
            net.train()
            Acc = Accuracy/10000
            for i in range(17):
                accuracy,pricision,recall,f1_score = pricision_recall(record_matrix[0,i],record_matrix[1,i],record_matrix[2,i],record_matrix[3,i])
                print(record_matrix[:,i])
                score_matrix[i,:] = [accuracy,pricision,recall,f1_score]
            temp_F1_socre = score_matrix[:,3].sum()/17
            if(temp_F1_socre>FSCORE):
                print("准确率为%.4f%%,F1_score从%.4f%%提高为%.4f%%,保存新的权重"%(Acc*100,FSCORE,temp_F1_socre))
                path = "./checkpoint/checkpoint_epoch_{}.xlxs".format(epoch)
                FSCORE = temp_F1_socre
                score_df = pd.DataFrame(score_matrix)
                data_df = pd.DataFrame(record_matrix.astype(int))
                score_df.columns = ["accuracy","pricision","recall","f1_score"]
                score_df.index = attribute
                data_df.columns = attribute
                data_df.index = ["TP","FN","TN","FP"]
                writer = pd.ExcelWriter(path)
                data_df.to_excel(writer,'sheet1',float_format='%d')
                score_df.to_excel(writer,'sheet2',float_format='%.4f')
                writer.save()
                checkpoint(epoch,net)
            else:
                print("准确率为%.4f%%,F1_score为%.4f%%,没有保存新的权重"%(Acc*100,temp_F1_socre))
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.95
if __name__=='__main__':
    train()