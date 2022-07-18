from re import X
import time
from test import test

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import init
from torch.utils.data import DataLoader
from torchsummary import summary

from datasets.fname_dataset import CifarDataset
from models.LeNet import LeNet
from models.GoogLeNet import GoogLeNet
from models.AlexNet import AlexNet
from models.ResNet_X import ResNet18,ResNet34,ResNet50,ResNet101,ResNet152
from models.VGG_X import VGG
from opts import parse_opts
from tra_val import train
from utils.label_smoothing import LabelSmoothing
from utils.transform import transform


def main():
    # 指定设备使用情况
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 配置文件
    opt = parse_opts()
    print(opt)

    # 定义数据增强方法
    train_transform,test_transform = transform()

    # 构建数据集
    train_set = CifarDataset(dataset="train",transform=train_transform)
    dev_set = CifarDataset(dataset="valid",transform=test_transform)
    test_set = CifarDataset(dataset="test",transform=test_transform)

    # 构建数据加载器
    train_loader = DataLoader(dataset=train_set,batch_size=opt.batch_size,shuffle=False)
    dev_loader = DataLoader(dataset=dev_set,batch_size=opt.batch_size,shuffle=False)
    test_loader = DataLoader(dataset=test_set,batch_size=opt.batch_size,shuffle=False)

    # 初始化模型对象
    net = ResNet34().to(device)

    # 初始化模型参数
    for name, param in net.named_parameters():
        if 'weight' in name:
            init.normal_(param, mean=0, std=0.01) # 网络层间权重正态分布
        if 'bias' in name:
            init.constant_(param, val=0) # 网络偏置为常数0
    
    # pip install tensorboardX
    writer = SummaryWriter('runs')

    # pip install torchsummary
    summary(net, input_size=(3, 32, 32))

    # 定义损失函数
    criterion = LabelSmoothing(smoothing=0.2)
    # criterion = torch.nn.CrossEntropyLoss()

    # 定义优化器
    optimizer = optim.SGD(net.parameters(), lr=opt.lr,momentum=opt.momentum,weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                   milestones=[int(opt.epoch_num * 0.56), int(opt.epoch_num * 0.78)],
                   gamma=opt.gamma, last_epoch=-1)


    # 模型训练
    print("开始训练")
    since = time.time() # 训练计时
    train(opt,net,device,criterion,optimizer,train_loader,dev_loader,writer,scheduler)
    time_elapsed = time.time() - since
    print('训练结束！用时:{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # 保存模型参数
    torch.save(net.state_dict(),"saved_models/GoogleNet_model.pt")

    # 对模型进行测试，并生成预测结果
    test(net,device,opt.batch_size,test_loader)

if __name__ == "__main__":
    main()
    
