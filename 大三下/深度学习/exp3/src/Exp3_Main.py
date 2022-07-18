"""
模型入口，程序执行的开始，在该文件中配置必要的训练步骤
"""


from Exp3_Config import Training_Config
from Exp3_DataSet import TextDataSet, TestDataSet
from torch.utils.data import DataLoader
from Exp3Model.TextCNN import TextCNN_Model
# from Exp3Model.BiRNN import BiRNN
# from Exp3Model.BiLSTMatttion import BiLSTM_Attention 
from sklearn.metrics import f1_score
import torch
from label_smoothing import LabelSmoothing
import json

train_l = []
train_acc = []
val_acc = []

def record():
    f1 = open("record/acc.txt","w") # 训练集正确率
    f2 = open("record/loss.txt",'w') # 训练集损失
    f3 = open("record/val_acc.txt",'w') # 验证集正确率

    for i in range(len(train_l)):
        f1.write(str(train_acc[i])+"\n")
        f2.write(str(train_l[i])+"\n")
        f3.write(str(val_acc[i])+"\n")

    f1.close()
    f2.close()
    f3.close()

def compute(tensor1,tensor2):
    '''
    计算F1值\n
    方法来自sklearn（三分类下的）
    '''
    y = tensor1.argmax(dim=1)
    y_pred = y.tolist() # 首先转换成列表
    y_true = tensor2.tolist() # 首先转换成列表
    f1 = f1_score( y_true, y_pred, average='macro' )

    return f1

def train(epoch,model, loader,device,optimizor,scheduler):
    model.train()
    f_all,train_l_sum,train_acc_sum,n,m = 0.0,0.0,0.0,0,0

    for index, data in enumerate(loader):
        entity1,entity2,relation,sentences = data[0],data[1],data[2],data[3]
        pos1,pos2=data[4],data[5]
        entity1 = entity1.to(device)
        entity2 = entity2.to(device)
        true_labels = relation.to(device)
        sentences = sentences.to(device)
        pos1 = pos1.to(device)
        pos2 = pos2.to(device)
        pre_labels = model(sentences, entity1, entity2, pos1, pos2)
        l = loss_function(pre_labels,true_labels) # 与真实label计算损失
            
        optimizor.zero_grad() # 梯度清零

        l.backward() # 计算梯度

        optimizor.step()  # 随机梯度下降算法, 更新参数

        train_l_sum += l.item() #利用.item()转成int类型进行累加求和损失结果
        # 这么写可以快速计算正确预测结果数
        train_acc_sum += (pre_labels.argmax(dim=1) == true_labels).sum().item()
        f_all += compute(pre_labels,true_labels)
        n += true_labels.shape[0] # 统计计算了多少个样本 一轮样本数==batchsize大小
        m += 1 
        
        if m%100==0: # 如果进行100次更新，输出结果
            print('epoch %d [%d/%d], loss %.4f, train acc %.3f, trian f1 %.3f'% (epoch + 1, m,
                    len(train_loader),train_l_sum / m, train_acc_sum / n, f_all / m))

    train_l.append((epoch ,train_l_sum / m))
    train_acc.append((epoch,train_acc_sum / n))
    scheduler.step()

def validation(epoch,model, loader,device):
    correct = 0
    f_all = 0
    total = 0
    m = 0
    model.eval()
    with torch.no_grad():
        for index, data in enumerate(loader):
            entity1,entity2,relation,sentences,pos1,pos2 = data[0],data[1],data[2],data[3],data[4],data[5]
            entity1 = entity1.to(device)
            entity2 = entity2.to(device)
            true_labels = relation.to(device)
            sentences = sentences.to(device)
            pos1 = pos1.to(device)
            pos2 = pos2.to(device)
            pre_labels = model(sentences, entity1, entity2, pos1, pos2)
            f_all += compute(pre_labels,true_labels)
            

            correct += (pre_labels.argmax(dim=1) == true_labels).sum().item() 
            total += true_labels.shape[0] # 统计计算了多少个样本 一轮样本数==batchsize大小
            m+=1
                
    print("--------------------------------------------------------")
    print("验证集数据总量：", total, "预测正确的数量：", correct)
    print("当前模型在验证集上的准确率为：", correct / total)
    print("当前模型在验证集上的F1值为：", f_all / m)
    print("--------------------------------------------------------")
    val_acc.append((epoch,correct / total))


def predict(model, loader,device,batchsize):
    f = open("predict_labels_1120190892.txt","w")

    model.eval()

    with torch.no_grad():  
        for index, data in enumerate(loader):
            entity1,entity2,sentences,pos1,pos2 = data[0],data[1],data[2],data[3],data[4]
            entity1 = entity1.to(device)
            entity2 = entity2.to(device)
            sentences = sentences.to(device)
            pos1 = pos1.to(device)
            pos2 = pos2.to(device)
            pre_labels = model(sentences, entity1, entity2, pos1, pos2)
            for i in range(batchsize):
                try:
                    out = pre_labels[i].argmax(dim=0)
                    f.write(str(out.item())+"\n")
                except:
                    pass

    f.close()

def label_balance(train_dataset):
    '''
    标签平衡函数
    '''
    Y = []
    with open("data/rel2id.json",'r',encoding='utf-8') as load_f:
        rel2id_table = json.load(load_f) 
    filepath = "data/data_train.txt"
    lines = open(filepath, 'r', encoding='utf-8').readlines()
    d = {}
    for line in lines:
        line = line.split('\t')
        d[rel2id_table[1][line[2]]] = d.get(rel2id_table[1][line[2]], 0)+1

    for i in range(0,44):
        Y.append(d[i]) # 统计每类别种类
    # -------------------------------------------------------------------
    class_sample_counts = Y
    weights = 1./ torch.tensor(class_sample_counts, dtype=torch.float)
    # 这个 get_classes_for_all_imgs是关键
    train_targets = train_dataset.get_classes_for_all_imgs()
    samples_weights = weights[train_targets]
    # 使用该函数进行按分布采样
    sampler = torch.utils.data.WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)
    # -------------------------------------------------------------------
    return sampler

if __name__ == "__main__":
    config = Training_Config()

    if config.cuda == True:
        device = 'cuda'
    else:
        device = 'cpu'

    # 训练集验证集
    train_dataset = TextDataSet(filepath="./data/data_train.txt",configs=config)
    sampler = label_balance(train_dataset)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size,shuffle=False)

    val_dataset = TextDataSet(filepath="./data/data_val.txt",configs=config)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size,shuffle=True)

    # 测试集数据集和加载器
    test_dataset = TestDataSet("./data/test_exp3.txt",configs=config)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size,shuffle=False)

    # 初始化模型对象
    Text_Model = TextCNN_Model(configs=config)
    # 损失函数设置
    # loss_function = torch.nn.CrossEntropyLoss()  # torch.nn中的损失函数进行挑选，并进行参数设置
    loss_function = LabelSmoothing(smoothing=0.2)
    # 优化器设置
    optimizer = torch.optim.Adam(params=Text_Model.parameters(),lr=config.lr)  # torch.optim中的优化器进行挑选，并进行参数设置
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                   milestones=[int(config.epoch * 0.56), int(config.epoch * 0.78)],
                   gamma=0.1, last_epoch=-1)

    print("训练开始！")
    Text_Model = Text_Model.to(device)

    # 训练和验证
    for i in range(config.epoch):
        train(i,Text_Model,train_loader,device,optimizer,scheduler)
        if i % config.num_val == 0:
            validation(i,Text_Model, loader=val_loader,device=device)

    # 预测（测试）
    predict(Text_Model, test_loader,device,config.batch_size)

    record()


