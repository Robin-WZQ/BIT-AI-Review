# 第二章作业说明

王中琦 1120190892

### Code Tree（以函数形式编写的代码风格）

|ML源码

​		|Linear Regression-Boston.ipynb # 波士顿房价预测

​		|Linear_regression.ipynb # sklearn线性回归

​		|regression2.ipynb # 多项式特征

​		|vaccine_regression.ipynb # 疫苗接种预测

|mindspore版

​		|波士顿房价预测

​				|linear-Boston.py # 主函数

​				|Net.py # 网络->模拟直线

​		|疫苗接种预测

​				|linear-vaccine.py # 主函数

​				|Net.py # 网络->模拟直线（多元线性回归）

|Bike sharing

​				|Bike_sharing.ipynb # 共享单车预测

------

### linear-Boston.py

#### 代码说明

1. 功能：波士顿房价预测
2. mindspore化：构建一层3->1的网络模拟多元线性回归

#### 函数说明

- **get_train**(X,y)：==获取训练数据迭代器==
  - 输入：
    - X: 特征(pandas读取类型)
    - y: 标签(pandas读取类型)
  - 输出：
    - X[i]：大小为[1，]
    - [[y[i]]：大小为[1，1]
- **get_test**(X,y)：==获取所有测试数据，Tensor类型==
  - 输入：
    - X: 特征(pandas读取类型)
    - y: 标签(pandas读取类型)
  - 输出：
    - x：特征（numpy.float32类型）
    - y：特征（numpy.float32类型）
- **create_dataset**(X_train,y_train,batch_size=16,repeat_size=1)：==创建数据迭代器==
  - 输入：
    - X_train：训练集X值
    - y_train：训练集y值
    - batch_size：批处理大小
    - repeat_size：数据重复次数
  
  - 输出：
    - input_data：数据迭代器
  
- mse(y_predict,y_test)：==MSE计算==
  - 输入：
    - y_predict：模型预测的y值
    - y_test：测试集的真实值
  - 输出：
    - 测试集mse
- test_all(net,X_test,y_test)：==测试函数，输出测试集的mse==
  - 输入：
    - net：模型（Class类型）
    - X_test：测试集x值
    - y_test：测试集y值
  - 输出：
    - None

### Net.py

#### 模型说明：

```python
class LinearNet(nn.Cell):
    def __init__(self):
        super(LinearNet,self).__init__()
        # 定义一个线形层，同时初始化权重和偏置
        self.fc=nn.Dense(3,1,Normal(0.02),Normal(0.02),has_bias=True) 
```

使用类进行封装，继承mindspore的nn.Cell父类。并定义3x1线性层模拟多元线性回归。

------

### linear-vaccine.py

#### 代码说明

1. 功能：疫苗接种预测（代码和波士顿房价预测略有差异）
2. mindspore化：构建一层1->1的网络模拟一元线性回归

#### 函数说明

- **get_train**(X,y)：==获取训练数据迭代器==
  - 输入：
    - X: 特征(pandas读取类型)
    - y: 标签(pandas读取类型)
  - 输出：
    - [X[i]]：大小为[1，1]
    - [[y[i]]：大小为[1，1]
- **get_test**(X,y)：==获取所有测试数据，Tensor类型==
  - 输入：
    - X: 特征(pandas读取类型)
    - y: 标签(pandas读取类型)
  - 输出：
    - x：特征（numpy.float32类型）
    - y：特征（numpy.float32类型）
- **create_dataset**(X_train,y_train,batch_size=16,repeat_size=1)：==创建数据迭代器==
  - 输入：
    - X_train：训练集X值
    - y_train：训练集y值
    - batch_size：批处理大小
    - repeat_size：数据重复次数
  - 输出：
    - input_data：数据迭代器

- mse(y_predict,y_test)：==MSE计算==
  - 输入：
    - y_predict：模型预测的y值
    - y_test：测试集的真实值
  - 输出：
    - 测试集mse
- test_all(net,X_test,y_test)：==测试函数，输出测试集的mse==
  - 输入：
    - net：模型（Class类型）
    - X_test：测试集x值
    - y_test：测试集y值
  - 输出：
    - None

### Net.py

#### 模型说明：

```python
class LinearNet(nn.Cell):
    def __init__(self):
        super(LinearNet,self).__init__()
        # 定义一个线形层，同时初始化权重和偏置
        self.fc=nn.Dense(1,1,Normal(0.02),Normal(0.02),has_bias=True) 
```

使用类进行封装，继承mindspore的nn.Cell父类。并定义1x1线性层模拟一元线性回归。

------

### Bike_sharing.ipynb

#### 代码说明：

1. 使用sklearn的多种回归方法，进行共享单车数据的预测任务

#### 结果如下：

- 随机森林结果最好

![output](C:\Users\24857\Desktop\output.png)

- 随机森林模型调参结果



|     特征名称     | 数值  |    结果    |
| :--------------: | :---: | :--------: |
|   n_estimator    |  10   |   0.3600   |
|                  |  100  | 0.3515 |
|                  |  200  |   0.3505   |
|                  |  400  |   **0.3500**   |
|    max_depth     |   3   |   0.8122   |
|                  |   5   |   0.6272   |
|                  |   8   |   0.4530   |
|                  |  15   | 0.3536 |
|                  |  25   |   0.3518   |
|                  |  50   |   **0.3512**   |
| min_samples_leaf |   1   | **0.3539** |
|                  |   2   |   0.3589   |
|                  |   3   |   0.3544   |
|   max_features   |   1   |   0.8451   |
|                  |   3   |   0.5308   |
|                  |   5   |   0.4181   |
|                  |   8   |   0.3550   |
|                  |   9   |   0.3517   |
|                  |  10   |   0.3502   |
|                  |  11   | **0.3490** |
|                  |  12   |   0.3505   |
|                  |  13   |   0.3522   |
|  max_leaf_nodes  |  100  |   0.6406   |
|                  | 1000  |   0.3981   |
|                  | 5000  |   0.3511   |
|                  | 10000 |   0.3505   |
|                  | 15000 |   0.3515   |
|                  | 20000 | **0.3490** |
