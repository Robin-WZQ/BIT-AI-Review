import torch

# 定义 train 函数
def train(opt,model,device,loss_function,optimizor,train_loader,dev_loader,writer,scheduler):
    # 参数设置
    epoch_num = opt.epoch_num
    val_num = opt.val_num

    for epoch in range(epoch_num):  # loop over the dataset multiple times
        train_l_sum, train_acc_sum, n, m = 0.0, 0.0, 0, 0
        for index, data in enumerate(train_loader, 0):
            # 该部分添加训练的主要内容
            images, labels = data
            images = images.to(device) # 加载进 cpu or gpu
            true_labels = labels.to(device) # 加载进 cpu or gpu
            pre_labels = model(images) # 输入模型，输出预测结果
            l = loss_function(pre_labels,true_labels) # 与真实label计算损失
            
            optimizor.zero_grad() # 梯度清零

            l.backward() # 计算梯度

            optimizor.step()  # 随机梯度下降算法, 更新参数

            train_l_sum += l.item() #利用.item()转成int类型进行累加求和损失结果
            # 这么写可以快速计算正确预测结果数
            train_acc_sum += (pre_labels.argmax(dim=1) == true_labels).sum().item() 
            n += true_labels.shape[0] # 统计计算了多少个样本 一轮样本数==batchsize大小
            m += 1 
            
            if m%100==0: # 如果进行100次更新，输出结果
                writer.add_scalar('loss/step', train_l_sum / m,
                     global_step= epoch * len(train_loader) + m, walltime=None)
                print('epoch %d [%d/%d], loss %.4f, train acc %.3f'% (epoch + 1, m,
                        len(train_loader),train_l_sum / m, train_acc_sum / n))
        # 模型训练n轮之后进行验证
        if epoch % val_num == 0:
            validation(model,dev_loader,device)
        
        # 学习率衰减项
        scheduler.step()

    print('Finished Training!')


# 定义 validation 函数
def validation(model,dev_loader,device):
    correct = 0
    total = 0
    model.eval() # 测试模型，去掉dropout和BN影响
    with torch.no_grad():
        for data in dev_loader:
            images, labels = data
            images = images.to(device) # 加载进 cpu or gpu
            true_labels = labels.to(device) # 加载进 cpu or gpu
            correct += (model(images).argmax(dim=1) == true_labels).float().sum().item()
            total += true_labels.shape[0]

    print("--------------------------------------------------------")
    print("验证集数据总量：", total, "预测正确的数量：", correct)
    print("当前模型在验证集上的准确率为：", correct / total)
    print("--------------------------------------------------------")