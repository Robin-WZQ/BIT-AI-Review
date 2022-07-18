from nbformat import write
from test import alltest

import torch
import torchvision.transforms as transforms
from torch.nn import init
from torch.utils.data import DataLoader
from torchsummary import summary
import time
from tensorboardX import SummaryWriter

from datasets.fname_dataset import HandWritingNumberRecognize_Dataset
from models.MLP import HandWritingNumberRecognize_Network
from opts import parse_opts
from tra_val import train, validation


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #使用何种训练设备

    opt = parse_opts() #配置文档
    print(opt) #打印配置文档
    
    # 构建数据集，参数和值需自行查阅相关资料补充。
    dataset_train = HandWritingNumberRecognize_Dataset(opt,dataset = "train",transform=transforms.ToTensor())

    dataset_val = HandWritingNumberRecognize_Dataset(opt,dataset = "val",transform=transforms.ToTensor())

    dataset_test = HandWritingNumberRecognize_Dataset(opt,dataset = "test",transform=transforms.ToTensor())

    print("-------------------------")
    print("读入数据完毕，其中\n训练集:{:1}张\n验证集:{:2}张\n测试集:{:3}张".format(len(dataset_train),len(dataset_val),len(dataset_test)))
    print("------------------------")

    # 构建数据加载器，参数和值需自行完善。
    data_loader_train = DataLoader(dataset=dataset_train,batch_size=opt.batch_size,shuffle=False) # 参数依次为：数据、batch大小、是否打乱

    data_loader_val = DataLoader(dataset=dataset_val,batch_size=opt.batch_size,shuffle=False)

    data_loader_test = DataLoader(dataset=dataset_test,batch_size=opt.batch_size,shuffle=False)

    # 初始化模型对象
    model = HandWritingNumberRecognize_Network()

    if opt.pretrain == "TRUE": # 提供导入训练好的模型
        model.load_state_dict(torch.load("saved_models/mlp_model.pt"))

    # 初始化模型权重
    for name, param in model.named_parameters():
        if 'weight' in name:
            init.normal_(param, mean=0, std=0.01) # 权重使用正态分布
        if 'bias' in name:
            init.constant_(param, val=0) # 偏置使用常数

    model = model.to(device)

    # pip install tensorboardX
    writer = SummaryWriter('runs') # 记录训练损失随step变化

    # pip install torchsummary
    summary(model, input_size=(1, 28, 28)) #记录模型相关参数，如参数量、层数以及层间关系等

    # 损失函数设置
    loss_function = torch.nn.CrossEntropyLoss()  

    # 优化器设置
    optimizer = torch.optim.RMSprop(model.parameters(), lr=opt.lr) 

    # 然后开始进行训练
    print("开始训练")
    since = time.time() # 记录训练时长
    for epoch in range(opt.max_epoch):
        train(epoch,data_loader_train,device,model,loss_function,optimizer,writer)
        # 在训练数轮之后开始进行验证评估
        if epoch % opt.num_val == 0:
            validation(data_loader_val,device,model)
    time_elapsed = time.time() - since
    print('训练结束！用时:{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    torch.save(model.state_dict(),"saved_models/mlp_model.pt") #保存模型

    # 自行完善测试函数，并通过该函数生成测试结果
    alltest(data_loader_test,device,model,opt.batch_size)

    print("测试结束")

    


if __name__ == "__main__":
    main()
