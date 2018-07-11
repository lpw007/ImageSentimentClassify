import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule

class LeNet1(BasicModule):
    '''
    定义LeNet网络
    '''
    def __init__(self):
        super(LeNet1, self).__init__()

        #定义网络的特征提取层
        self.features = nn.Sequential(
            nn.Conv2d(1, 7, 5,padding=1),
            nn.BatchNorm2d(7),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(),
            nn.Conv2d(7, 21, 3),
            nn.BatchNorm2d(21),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout()
        )

        self.classifier = nn.Sequential(
            nn.Linear(21 * 10 * 10, 120),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 7),
        )

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),21*10*10)
        x = self.classifier(x)
        return x