import os
import torch
import torch.utils.data as data
import torch.utils
import torch.nn as nn
from dataset import myImageFloder
import torchvision.transforms as transforms
from torch.autograd import Variable
from draw_data import  draw_loss
import model.HP_net as HP_net
from model.moblenet_v3 import MobileNetV3
import torchvision
# from visdom import Visdom
import numpy as np
from CenterLoss import CenterLoss

def checkpoint(epoch):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    path = "./checkpoint/checkpoint_epoch_{}".format(epoch)
    torch.save(net.state_dict(), path)


mytransform = transforms.Compose([

    transforms.RandomHorizontalFlip(),#随机翻转
    transforms.Resize((224, 128)),#顺序为h,w
    torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5),#含有颜色标签，使用色相或则色调偏移不利于检测
    # transforms.RandomCrop(299),
    # transforms.Resize((299,299)),       #TODO:maybe need to change1
    transforms.ToTensor(),  # mmb,
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                            std  = [ 0.229, 0.224, 0.225 ]),
]
)

# torch.utils.data.DataLoader
set = myImageFloder(root=r"D:\work\datas\PA-100K-20210716T100412Z-001\PA-100K\release_data\release_data", label=r"D:\work\datas\PA-100K-20210716T100412Z-001\PA-100K\annotation.mat",
                    transform=mytransform)
imgLoader = torch.utils.data.DataLoader(set,batch_size=2, shuffle=True, num_workers=0)


net = MobileNetV3(type='large', num_classes=17)
# net.load_state_dict(torch.load('./checkpoint/checkpoint_epoch_95'))
# path = "./checkpoint5/checkpoint_epoch_40"  # FIXME:
# net.load_state_dict(torch.load(path))
# for param in net.MNet.parameters():
#     param.requires_grad = False
#
# for param in net.AF1.parameters():
#     param.requires_grad = False
#
# for param in net.AF2.parameters():
#     param.requires_grad = False
#
# for param in net.AF3.parameters():
#     param.requires_grad = False

net.cuda()
net.train()

#weight控制样本权重
# weight = torch.Tensor([1.7226262226969686, 2.6802565029531618, 1.0682133644154836, 2.580801475214588,
#                        1.8984257687918218, 2.046590013290684, 1.9017984669155032, 2.6014006200502586,
#                        2.272458988404639, 2.2625669787021203, 2.245380512162444, 2.3452980639899033,
#                        2.692210221689372, 1.5128949487853383, 1.7967419553099035, 2.5832221110933764,
#                        2.3302195718894034, 2.438480257574324, 2.6012705532709526, 2.704589108443237,
#                        2.6704246374231753, 2.6426970354162505, 1.3377813061118478, 2.284449325734624,
#                        2.417810793601295, 2.7015143874115033])
weight = torch.Tensor([1.7226262226969686, 2.6802565029531618, 1.0682133644154836, 2.580801475214588,
                       1.8984257687918218, 2.046590013290684, 1.9017984669155032, 2.6014006200502586,
                       2.272458988404639, 2.2625669787021203, 2.245380512162444, 2.3452980639899033,
                       1.5128949487853383, 1.7967419553099035, 1.3377813061118478, 2.284449325734624,
                       2.417810793601295])#去除了几个样本非常不均衡的类别[13，16，17，18，19，20，21，22，26]

# criterion_1 = nn.BCEWithLogitsLoss(weight=weight)  # TODO:1.learn 2. weight
criterion_2 = nn.CrossEntropyLoss()
centerloss = CenterLoss(10,10)
centerloss.cuda()
# criterion_1.cuda()

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.001)

running_loss = 0.0
loss_record = []
EPOCHS = 100
for epoch in range(EPOCHS):
    for i, data in enumerate(imgLoader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        # print(outputs)
        label = labels.long()
        te1 = outputs[10]
        loss_female         = criterion_2(outputs[10], label[:,0])
        loss_hat            = criterion_2(outputs[11], label[:,1])
        loss_glass          = criterion_2(outputs[12], label[:,2])
        loss_handbag        = criterion_2(outputs[13], label[:,3])
        loss_shoulderbag    = criterion_2(outputs[14], label[:,4])
        loss_carry          = criterion_2(outputs[15], label[:,5])
        loss_age            = criterion_2(outputs[16], torch.max(label[:,6:9].data,1)[1])
        loss_dirction       = criterion_2(outputs[17], torch.max(label[:,9:12].data,1)[1])
        loss_upclothing     = criterion_2(outputs[18], torch.max(label[:,12:14].data,1)[1])
        loss_downclothing   = criterion_2(outputs[19], torch.max(label[:,14:17].data,1)[1])
        center_loss_female         = centerloss(label[:,0],outputs[0])
        center_loss_hat            = centerloss(label[:,1],outputs[1])
        center_loss_glass          = centerloss(label[:,2],outputs[2])
        center_loss_handbag        = centerloss(label[:,3],outputs[3])
        center_loss_shoulderbag    = centerloss(label[:,4],outputs[4])
        center_loss_carry          = centerloss(label[:,5],outputs[5])
        center_loss_age            = centerloss(torch.max(label[:,6:9].data,1)[1],outputs[6])
        center_loss_dirction       = centerloss(torch.max(label[:,9:12].data,1)[1],outputs[7])
        center_loss_upclothing     = centerloss(torch.max(label[:,12:14].data,1)[1],outputs[8])
        center_loss_downclothing   = centerloss(torch.max(label[:,14:17].data,1)[1],outputs[9])
        loss = loss_female+loss_hat+loss_glass+loss_handbag+loss_shoulderbag+loss_carry+loss_age+loss_dirction+loss_upclothing+loss_downclothing  \
            +center_loss_female+center_loss_hat+center_loss_glass+center_loss_handbag+center_loss_shoulderbag+center_loss_carry+center_loss_age+center_loss_dirction+center_loss_upclothing+center_loss_downclothing
        # print(loss)
        loss = loss/20
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data
        if i % 100 == 0:  # print every 1000 mini-batches
            print('epoch:%d  i:%d loss: %.6f' % (epoch, i + 1, running_loss / 100))
            # viz.updateTrace(
            #     X=np.array([epoch + i / 8000.0]),
            #     Y=np.array([running_loss]),
            #     win=win,
            #     name="1"
            # )
            loss_record.append(running_loss)
            running_loss = 0.0
    draw_loss(loss_record,EPOCHS,'loss_record.jpg')
    if epoch % 5 == 0:
        checkpoint(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.95
