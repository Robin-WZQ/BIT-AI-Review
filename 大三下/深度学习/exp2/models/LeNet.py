import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1 = nn.Conv2d(3,16,5)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16,32,5)
        self.pool2 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(32*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
    
    def forward(self, x):
        x = F.elu(self.conv1(x))#输入(3,32,32)
        x = self.pool1(x) 
        x = F.elu(self.conv2(x)) 
        x = self.pool2(x) 
        x = x.view(-1,32*5*5) 
        x = F.elu(self.fc1(x)) 
        x = F.elu(self.fc2(x)) 
        x = self.fc3(x) #输出10
        return x

