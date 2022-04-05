"""
线性回归-vaccine接种预测-mindspore版
构建一层3->1的网络
"""
import numpy as np
import pandas as pd
from mindspore import Model, Tensor
from mindspore import dataset as ds
from mindspore import nn

from Net import LinearNet


def get_train(X,y):
    '''
    获取训练数据
    param: 
        X: 特征(pandas读取类型)
        y: 标签(pandas读取类型)
    '''
    X,y=np.array(X).astype(np.float32),np.array(y).astype(np.float32) # 太坑了
    for i in range(len(X)):
        yield [X[i]],[y[i]]
        

def get_test(X,y):
    '''
    获取测试数据
    '''
    X,y=np.array(X).astype(np.float32),np.array(y).astype(np.float32)# 太坑了

    return X,y
        
def create_dataset(X_train,y_train,batch_size=16,repeat_size=1):
    '''
    创建数据迭代器
    '''
    a = list(get_train(X_train,y_train))
    input_data=ds.GeneratorDataset(a,column_names=['data','label'])
    input_data=input_data.batch(batch_size) # 设置数据批次
    input_data=input_data.repeat(repeat_size) # 设置数据重复次数
    return input_data

def mse(y_predict,y_test):
    error = 0
    for i in range(y_predict.shape[0]):
        error+=(y_predict[i]-y_test[i])**2
    error /= y_predict.shape[0]
    error = error**0.5
    print("测试集的mse为：",error)
    return error


def test_all(net,X_test,y_test):
    '''
    测试函数，输出测试集的mse
    '''
    weight = net.trainable_params()[0]
    bias = net.trainable_params()[1]
    x_test,y_test = get_test(X_test,y_test)
    a= Tensor(weight).asnumpy()[0]
    a = np.expand_dims(a, 1)
    x_test = np.expand_dims(x_test, 1)
    b = np.matmul(x_test, a)
    y_predict =  b + Tensor(bias).asnumpy()[0]
    mse(y_predict,y_test)

def main():
    # ===================================================
    # 读入数据->绝对路径
    df = pd.read_csv("F:/junior/xia/ML/实验-mindspore/Chapter-2/疫苗接种预测/vaccine.csv")
    features = df["Year"]
    target = df["Values"]
    split_num = int(len(features)*0.7)

    X_train = features[:split_num]
    y_train = target[:split_num]    

    X_test = features[split_num:]
    y_test = target[split_num:]
    # ===================================================

    # ===================================================
    # 初始化超参数
    batch_number=10
    repeat_number=1
    epoch = 1
    # ===================================================

    # 创建数据
    ds_train=create_dataset(X_train,y_train,batch_number,repeat_number)
    print(ds_train)

    # ===================================================
    # 创建模型
    net=LinearNet()
    net_loss=nn.loss.MSELoss()
    opt=nn.Momentum(net.trainable_params(),learning_rate=1e-4,momentum=0.01)

    model=Model(net,net_loss,opt)
    # ===================================================

    # ===================================================
    # 训练+测试
    model.train(epoch, ds_train, dataset_sink_mode=False)

    test_all(net,X_test,y_test)
    # ===================================================

    # 打印线性回归参数
    for net_param in net.trainable_params():
        print(net_param, net_param.asnumpy())

if __name__ == "__main__":
    main()


