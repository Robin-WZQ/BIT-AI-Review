import torch
from tensorboardX import SummaryWriter

def validation(data_loader_val,device,model):
    # 验证函数，任务是在训练经过一定的轮数之后，对验证集中的数据进行预测并与真实结果进行比对，生成当前模型在验证集上的准确率
    correct = 0 #正确个数
    total = 0 #总共多少个
    accuracy = 0 #正确率
    model.eval() # 测试模型，保证dropout和BN层失效
    with torch.no_grad():  # 该函数的意义需在实验报告中写明
        for data in data_loader_val:
            images, true_labels = data # 按元组读取数据
            images = images.to(device) #数据转到相应设备
            true_labels = true_labels.to(device) #数据转到相应设备
            correct += (model(images).argmax(dim=1) == true_labels).float().sum().item() # 计算所有预测正确个数
            total += true_labels.shape[0] # 计算总共看了多少个
        accuracy = correct/total #计算正确率

    print("--------------------------------------------------------")
    print("验证集数据总量：", total, "预测正确的数量：", correct)
    print("当前模型在验证集上的准确率为：", accuracy)
    print("--------------------------------------------------------")



def train(epoch,data_loader_train,device,model,loss_function,optimizor,writer):
    train_l_sum, train_acc_sum, n, m = 0.0, 0.0, 0, 0 #依次为训练损失、训练正确率、多少样本、多少步
    
    for index, data in enumerate(data_loader_train, 0):
        # 该部分添加训练的主要内容
        images, true_labels = data
        images = images.to(device) # 加载进 cpu or gpu
        true_labels = true_labels.to(device) # 加载进 cpu or gpu
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
                     global_step= epoch * len(data_loader_train) + m, walltime=None)
            print('epoch %d [%d/%d], loss %.4f, train acc %.3f'% (epoch + 1, m,
                        len(data_loader_train),train_l_sum / m, train_acc_sum / n))
        