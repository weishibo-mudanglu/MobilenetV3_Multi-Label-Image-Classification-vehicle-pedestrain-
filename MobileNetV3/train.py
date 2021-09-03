import sys
sys.path.append('./data')
sys.path.append('./model')
import platform
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from matplotlib import pyplot as plt
from dataset import MyDataset
from model import MobileNetV3_large
from model import MobileNetV3_small
import torchvision
from torch.autograd import Variable
# import pandas as pd
from draw_data import draw_loss
from my_utils import CrossEntropyLabelSmooth
import cvtransforms
from CenterLoss import  CenterLoss



#宏定义一些数据，如epoch数，batchsize等
MAX_EPOCH=100
BATCH_SIZE=5
LR=0.001
re_size = (64,64)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ============================ step 1/5 数据 ============================
if(platform.system()=='Windows'):
    train_path = r"../XML_resolver/train_plus"
    test_path = r"../XML_resolver/test_val"
    val_path = r"../XML_resolver/val"
elif(platform.system()=='Linux'):
    train_path = r"/home/Glasssix-LQJ/wei_data/vehicle_attribute/train_plus"
    test_path = r"/home/Glasssix-LQJ/wei_data/vehicle_attribute/test_val"
    val_path = r"/home/Glasssix-LQJ/wei_data/vehicle_attribute/val"
# test_path = r"D:\work\datas\vehicle_attri\dataset\VeRi\image_test"

#对训练集所需要做的预处理
train_transform=cvtransforms.Compose([
    cvtransforms.Resize(re_size),
    cvtransforms.RandomRotation(10),#随机角度旋转
    cvtransforms.ColorJitter(brightness=0.5, contrast=0.5,saturation=0.0001,hue=0.0001),#含有颜色标签，使用色相或则色调偏移不利于检测
    cvtransforms.RandomHorizontalFlip(),#随机裁剪
    cvtransforms.ToTensor(),
    cvtransforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                            std  = [ 0.229, 0.224, 0.225 ]),
])

#对验证集所需要做的预处理，用于验证的数据只需要resize和归一化
valid_transform=cvtransforms.Compose([
    cvtransforms.Resize(re_size),
    # cvtransforms.RandomRotation(10),#随机角度旋转
    # cvtransforms.ColorJitter(brightness=0.5, contrast=0.5,saturation=0.001,hue=0.001),#含有颜色标签，使用色相或则色调偏移不利于检测
    # cvtransforms.RandomHorizontalFlip(),#随机裁剪
    cvtransforms.ToTensor(),
    cvtransforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                            std  = [ 0.229, 0.224, 0.225 ]),
])



def train():
    record_loss = []
    record_dirction_loss = []
    record_type_loss = []
    record_color_loss = []
    log_interval=3
    val_interval=1
    # 构建MyDataset实例
    train_data=MyDataset(txt=r'./data/train_plus.txt',path = train_path, transform=train_transform)
    val_data = MyDataset(txt=r'./data/val.txt',path = val_path, transform=train_transform)
    # 构建DataLoader
    # 训练集数据最好打乱
    # DataLoader的实质就是把数据集加上一个索引号，再返回
    train_loader=DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=6)
    val_loader=DataLoader(dataset=val_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=6)
    # ============================ step 2/5 模型 ============================
    net=MobileNetV3_small(num_classes=20)
    #temp = torch.load('./weights/best.pkl')
    #net_dict = net.state_dict()
    #pretrained_dict = {k: v for k, v in temp.items() if k !='linear2.weight' and k !='linear2.bias'} 
    #net_dict.update(pretrained_dict)
    #net.load_state_dict(temp)#加载前面的模型继续训练
    net.to(device)
    #=============================模型微调================================
    # for name, param in net.named_parameters():
    #     if "bn1" in name:
    #         param.requires_grad = False
    #     if "layers" in name:
    #         param.requires_grad = False
    #     if "conv2" in name:
    #         param.requires_grad = False
    #     if "bn2" in name:
    #         param.requires_grad = False
    #     if "conv3" in name:
    #         param.requires_grad = False
    # ============================ step 3/5 损失函数 ============================
    #交叉熵损失函数设置为三个不同的调用名，方便对不同的标签赋予权重
    weight_t = torch.Tensor([0.833639004,2.530824891,1.27649635,2.020229885,0.91752361,2.020229885,2.020229885,1.520229885]).to(device)
    weight_c = torch.Tensor([1.938802661,2.97808219,2.313227513,0.877292576,1.162765957,2.449309665,0.855653986,2.16744186,2.431055901,0.708050026]).to(device)
    # criterion=torch.nn.BCEWithLogitsLoss()#二值交叉熵
    criterion_d =CrossEntropyLabelSmooth(10, 0.1,0)
    criterion_t =CrossEntropyLabelSmooth(10, 0.1,0)
    criterion_c =CrossEntropyLabelSmooth(10, 0.1,0)
    criterion_t.to(device)
    criterion_c.to(device)
    centerloss_d  = CenterLoss(2,10).to(device)
    centerloss_t  = CenterLoss(8,10).to(device)
    centerloss_c  = CenterLoss(10,10).to(device)
    # ============================ step 4/5 优化器 ============================
    optimizer=optim.Adam(net.parameters(),lr=LR, betas=(0.9, 0.99))# 选择优化器
    # ============================ step 5/5 训练 ============================
    
    
    accurancy_global=0.0
    for epoch in range(MAX_EPOCH):
        correct=0.
        total=0.
        for i,data in enumerate(train_loader):
            net.train()
            img,label=data
            #------------------对数据按标记进行划分-----------------------#
            index_0 = torch.nonzero((label[:,3]==0),as_tuple=False).squeeze(-1)#方向+类型+颜色 
            index_1 = torch.nonzero((label[:,3]==1),as_tuple=False).squeeze(-1)#方向
            index_2 = torch.nonzero((label[:,3]==2),as_tuple=False).squeeze(-1)#类型
            index_3 = torch.nonzero((label[:,3]==3),as_tuple=False).squeeze(-1)#颜色
            index_4 = torch.nonzero((label[:,3]==4),as_tuple=False).squeeze(-1)#方向+类型
            index_5 = torch.nonzero((label[:,3]==5),as_tuple=False).squeeze(-1)#方向+颜色
            index_6 = torch.nonzero((label[:,3]==6),as_tuple=False).squeeze(-1)#类型+颜色
            

            #-------------------依照标记分组求损失------------------------#
            index_dirction = torch.cat((index_0,index_1,index_4,index_5))
            index_type     = torch.cat((index_0,index_2,index_4,index_6))
            index_color    = torch.cat((index_0,index_3,index_5,index_6))
            #------------------全部数据送入网络训练-----------------------#
            img=img.to(device)
            label=label.to(device)
            label = label.long()
            f_dirction,f_type,f_color,out1,out2,out3=net(img)

            optimizer.zero_grad()  # 归0梯度

            #------------------对结果按标记分组求损失函数并相加------------#
            if(len(index_dirction)!=0):
                loss1=criterion_d(out1[index_dirction],label[:,0][index_dirction])#得到损失函数
                centerloss_dirction = centerloss_d(label[:,0][index_dirction],f_dirction[index_dirction])
            else:
                loss1= 0
                centerloss_dirction = 0
            if(len(index_type)!=0):
                loss2=criterion_t(out2[index_type],label[:,1][index_type])
                centerloss_type     = centerloss_t(label[:,1][index_type],f_type[index_type])
            else:
                loss2= 0
                centerloss_type  = 0
            if(len(index_color)!=0):
                loss3=criterion_c(out3[index_color],label[:,2][index_color])
                centerloss_color    = centerloss_c(label[:,2][index_color],f_color[index_color])
            else:
                loss3= 0
                centerloss_color    =  0
            loss = loss1+loss2+loss3*1.5 + centerloss_dirction + centerloss_type + centerloss_color
            loss.backward()#反向传播
            if (i+1)%log_interval==0:
                print('epoch:{},loss:{:.6f}'.format(epoch+1,loss.data.item()))
            optimizer.step()#优化
        
        for i,data in enumerate(val_loader):
            net.eval()
            img,label=data

            #------------------全部数据送入网络训练-----------------------#
            img=img.to(device)
            label=label.to(device)
            label = label.long()
            f_dirction,f_type,f_color,out1,out2,out3=net(img)

            #记录损失率
            record_loss.append(loss.data.item())
            record_dirction_loss.append(loss1.data.item())
            record_type_loss.append(loss2.data.item())
            record_color_loss.append(loss3.data.item())
            
            _,direction_pre = torch.max(out1.data, 1)
            _,type_pre = torch.max(out2.data, 1)
            _,color_pre = torch.max(out3.data, 1)


            prediction = torch.cat((direction_pre.unsqueeze(1),type_pre.unsqueeze(1),color_pre.unsqueeze(1)),1)
            for j in range(prediction.shape[0]):
                if(torch.equal(prediction[j,:],label[j,0:-1])):
                    correct = correct +1
                else:
                    if(label[j,3]==2 and prediction[j,1]==label[j,1]):
                        correct = correct +1
            total += label.size(0)
    
        accurancy=correct / total
        if accurancy>accurancy_global:
            torch.save(net.state_dict(), './weights/best.pkl')
            print("准确率由：", accurancy_global, "上升至：", accurancy, "已更新并保存权值为weights/best.pkl")
            accurancy_global=accurancy
        print('第%d个epoch的识别准确率为：%.4f%%' % (epoch + 1, 100*accurancy))
    torch.save(net.state_dict(), './weights/last.pkl')
    torch.save(optimizer.state_dict,'./weights/last_op.pkl')
    # draw_loss(record_loss,MAX_EPOCH,'./weights/train_loss.jpg')
    # draw_loss(record_dirction_loss,MAX_EPOCH,'./weights/train_dloss.jpg')
    # draw_loss(record_type_loss,MAX_EPOCH,'./weights/train_tloss.jpg')
    # draw_loss(record_color_loss,MAX_EPOCH,'./weights/train_closs.jpg')
    print("训练完毕，权重已保存为：weights/last.pkl")
def test():
    direction_correct=0.
    type_correct = 0.
    color_correct = 0.
    correct=0.
    total=0.
    total1=0.
    type_error_map = np.zeros((10,9))
    color_error_map = np.zeros((12,11))
    #加载测试数据
    test_data=MyDataset(txt=r'.\data\test_val.txt',path = test_path,transform=valid_transform)
    # test_data=MyDataset(txt=r'C:\road01_ins\car1.txt',path = r'C:\road01_ins\car1',transform=valid_transform)
    # test_data=MyDataset(txt=r'.\data\test_all.txt',path = test_path,re_size=re_size)
    # test_data=MyDataset(txt=r'.\data\train_all.txt',path = train_path, transform=valid_transform)

    valid_loader=DataLoader(dataset=test_data,batch_size=10,shuffle=True)
    #加载网络模型和参数
    PATH = "./weights/best.pkl"
    net = MobileNetV3_small(num_classes=20)
    net.load_state_dict(torch.load(PATH,map_location='cuda:0'))

    net.to(device)
    net.eval()
    # mobilenetv3_small = torch.jit.trace(net, torch.rand(1,3,64,64))
    # a = torch.randn(2,3,64,64)
    # print(mobilenetv3_small(a))
    # mobilenetv3_small.save('mobilenetv3_small.pt')
    
    for i,data in enumerate(valid_loader):
        img,label=data
        # img = Variable(img)
        # label = Variable(label)
        index_0 = torch.nonzero((label[:,3]==0),as_tuple=False).squeeze(-1)
        img=img.to(device)
        label=label.to(device)
        # 前向传播
        # img = img.permute(0,3,1,2)
        _,_,_,out1,out2,out3=net(img)
        _,direction_pre = torch.max(out1.detach(), 1)#正常情况对张量的操作会被梯度跟踪，这里量化时使用.detach()操作可以避免梯度跟踪
        _,type_pre = torch.max(out2.detach(), 1)
        _,color_pre = torch.max(out3.detach(), 1)

        # _,direction_label = torch.max(label[:,:2].data, 1)
        # _,type_label = torch.max(label[:,2:15].data, 1)
        # _,color_label = torch.max(label[:,15:].data, 1)
        prediction = torch.cat((direction_pre.unsqueeze(1),type_pre.unsqueeze(1),color_pre.unsqueeze(1)),1)
        # temp_label = torch.cat((direction_label.unsqueeze(1),type_label.unsqueeze(1),color_label.unsqueeze(1)),1)
        # direction_correct += (direction_pre == direction_label).sum()
        # type_correct += (type_pre == type_label).sum()
        # color_correct += (color_pre == color_label).sum()
        direction_correct += (direction_pre == label[:,0]).sum()
        type_correct += (type_pre == label[:,1]).sum()
        color_correct += (color_pre == label[:,2]).sum()
        for i in range(prediction.shape[0]):
            # if(torch.equal(prediction[i,:], temp_label[i,:])):
            type_error_map[9,int(label[i,1])] = type_error_map[9,int(label[i,1])] +1
            if(label[i,3]==0):
                color_error_map[11,int(label[i,2])] = color_error_map[11,int(label[i,2])] +1
            if(torch.equal(prediction[i,:], label[i,0:-1])):
                correct = correct +1
            else:
                if(not torch.equal(prediction[i,1], label[i,1])):
                    type_error_map[int(prediction[i,1]),int(label[i,1])] = type_error_map[int(prediction[i,1]),int(label[i,1])] +1
                if(not torch.equal(prediction[i,2], label[i,2]) and label[i,3]==0):
                    color_error_map[int(prediction[i,2]),int(label[i,2])] = color_error_map[int(prediction[i,2]),int(label[i,2])] +1
                if(prediction[i,1]==label[i,1] and label[i,3]==2):
                    correct = correct +1
        total += label.size(0)
        total1 += len(index_0)
        accurancy=correct / total
        print('全标签准确率：%f，方向：%f，类型：%f，颜色：%f' % (100*accurancy,100*direction_correct/ total1,100*type_correct/ total,100*color_correct/ total1))
    for i in range(8):
        type_error_map[8,i] = np.sum(type_error_map[0:8,i])
        type_error_map[i,8] = np.sum(type_error_map[i,:])
    for i in range(10):
        color_error_map[10,i] = np.sum(color_error_map[0:9,i])
        color_error_map[i,10] = np.sum(color_error_map[i,:])
    # 分类错误统计，保存为excel表格
    data_df1 = pd.DataFrame(type_error_map)
    data_df2 = pd.DataFrame(color_error_map)
    data_df1.index = ["sedan","bicycle","van","bus","truck",'tricycle','motor','others',"sum","all"]
    data_df1.columns = ["sedan","bicycle","van","bus","truck",'tricycle','motor','others',"sum"]

    data_df2.index = ["yellow","orange","green","gray","red","blue","white","golden","brown","black","sum","all"]
    data_df2.columns = ["yellow","orange","green","gray","red","blue","white","golden","brown","black","sum"]
    writer = pd.ExcelWriter('./weights/errors.xlsx')  #关键2，创建名称为hhh的excel表格
    data_df1.to_excel(writer,'page_1',float_format='%.5f')
    data_df2.to_excel(writer,'page_2',float_format='%.5f')
    writer.save()

if __name__ == '__main__':
    train()
    # test()
