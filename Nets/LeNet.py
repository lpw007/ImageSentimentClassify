import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule

class LeNet(BasicModule):
    def __init__(self):
        super(LeNet, self).__init__()
        self.con1 = nn.Conv2d(1,7,5)
        self.pool = nn.MaxPool2d(2,2)
        self.con2 = nn.Conv2d(7,21,3)
        self.fc1 = nn.Linear(21*10*10,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,7)

    # def forward(self, x):
    #     x = self.pool(F.relu(self.con1(x)))
    #     x = self.pool(F.relu(self.con2(x)))
    #     x = x.view(-1, 21 * 10 * 10)
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     return x

    def forward(self, x):
        x = self.con1(x)