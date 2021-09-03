from scipy.io import loadmat
import numpy as np
# m = loadmat(r"D:\li_data\li_person_attri\Market-1501-v15.09.15\Market-1501_Attribute-master\market_attribute")
# l = loadmat(r"D:\li_data\li_person_attri\DukeMTMC-reID\DukeMTMC-attribute-master\duke_attribute")
# attribute = m['market_attribute'][0][0]
# attri = l['duke_attribute'][0][0]
# # print(attribute[1][0][0][-1][0], len(attribute[0][0][0]))
# print(len(attri[1][0][0]))
# a = np.array(attri[1][0][0][-1][0], dtype=np.int)
# print(a)
# print(np.where(a==5))
# print('------------:', attribute[1])
# print(attribute[0][0][0])
# print(attribute[0], len(attribute))
# print(attri[0],len(attri))
# print(l['duke_attribute'][0][0])
# print('keys', l.keys())
# print(m)
# import torch
# a = torch.load('model/pa100k_epoch_8.pth')
# # print(a.keys())
# a = torch.tensor([[1,2]])
# # b =
#################################################
path = r'D:\li_data\li_person_attri\pa_100k\annotation'
info = m = loadmat(path)#['attributes']
print(info.keys())
print(info['train_images_name'][43])
print('label', info['train_label'][43])