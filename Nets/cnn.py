import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule

class CNN(BasicModule):
    '''
    定义CNN网络
    '''
    def __init__(self):
        super(CNN, self).__init__()

        #定义网络的特征提取层
        self.features = nn.Sequential(
            nn.Conv2d(1,64,3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.3),
            nn.Conv2d(64,128,3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3),
            nn.Conv2d(128,256,3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3),
            # nn.Conv2d(256, 512, 3),
            # nn.BatchNorm2d(512),
            # nn.ReLU(),
            # nn.MaxPool2d(2, 2),
            # nn.Dropout(0.4),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256*4*4,512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512,7),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1,256*4*4)
        x = self.classifier(x)
        return F.log_softmax(x,dim=1)