from __future__ import print_function,division
import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils import data
from torchvision import transforms as T
from utils import labelToOneHot

class imageSentiment(data.Dataset):

    def __init__(self,root ,transforms = None,train = True,test = False):
        '''
        获取csv文件中的数据集，并根据训练集，测试集，验证集划分数据
        '''
        self.test = test
        data_frame = pd.read_csv(root,dtype='a')
        image_nums = len(np.array(data_frame['feature']))
        self.image_data = np.array(data_frame['feature'])
        image_idxs = [i for i in range(image_nums)]

        if not self.test:
            image_label = np.array(data_frame['label'])
            image_label = [int(i) for i in image_label]
            # label_tensor = labelToOneHot(image_label)
            self.label_tensor = image_label

        #shuffle imgs
        np.random.seed(100)
        image_idxs = np.random.permutation(image_idxs)

        if self.test:
            self.img_idxs = image_idxs
        elif train:
            self.img_idxs = image_idxs[:int(0.8*image_nums)]
        else:
            self.img_idxs = image_idxs[int(0.8*image_nums):]

        if transforms is None:
            self.transforms = T.Compose([
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ])

    def __getitem__(self, index):
        '''
        返回一张图片的数据
        '''
        index = self.img_idxs[index]
        data = self.image_data[index]
        data = data.split()
        input = [float(i) for i in data]
        input = np.array(input)
        input = input.reshape([48,48])/255.
        image = Image.fromarray(input) #生成image对象
        data = self.transforms(image)
        if self.test:
            id = index
            return id,data
        else:
            # label = self.image_label[index]
            #完成多类别的label转化为one-hot label vector
            # label_vector = torch.zeros(1,7).scatter_(1,label,1)
            label_vector = self.label_tensor[index]
            return label_vector,data

    def __len__(self):
        return len(self.img_idxs)

