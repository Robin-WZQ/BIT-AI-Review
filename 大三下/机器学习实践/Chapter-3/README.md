# 第三章作业说明

王中琦 1120190892

### Code Tree

|ML源码

​		|DBSCAN.ipynb #DBSCAN聚类

​		|Kmeans.ipynb # Kmeans聚类

|mindspore版

​		|DBSCAN.py #DBSCAN聚类

​		|Kmeans.py # Kmeans聚类

|Kernel- Kmeans.ipynb # Kernel Kmeans聚类

|谱聚类.ipynb

------

### DBSCAN.py

#### 代码说明：

- 使用mindspore的Tensor进行修改，并使用了自己定义的accuary计算机制

#### 函数说明：

- **findNeighbor**(j , X, eps)： ==寻找eps领域内的点==
  - 输入：
    - j：第j个点
    - X：所有点的X值
    - eps：邻域范围大小
  - 输出：
    - N：eps邻域内的点
- **dbscan**(X, eps, min_Pts)： ==聚类主函数==
  - 输入：
    - X(Tensor):样本数据
    - eps(float):eps邻域半径
    - min_Pts(int):eps邻域内最少点个数
  - 输出：
    - cluster(list):聚类结果
- **score**(y_predict,y_test) ：==吻合度计算函数==
  - 输入：
    - y_predict(list)：模型预测结果
    - y_test(list)：真实结果
  - 输出：
    - score：吻合度

#### 结果如下

![output3](C:\Users\24857\Desktop\output3.png)

------

### Kernel Kmeans.ipynb(以类的风格编写的代码)

#### 代码说明：

- mindspore没改成功，太难改了

#### 函数说明：

- **get_data**() ： ==获取数据并可视化展示==

  - 输入：
    - None
  - 输出：
    - x：横坐标

- **process**(_x)：==使用高斯核映射至高维空间==

  - 输入：
    - _x：原始x坐标

  - 输出：
    - Z：映射至高位空间后的坐标

- **euclidean_distance**(one_sample, X)：==欧式距离计算==
  - 输入：
    - one_sample：样本
    - X：中心节点
  - 输出：
    - distances：样本距中心点各距离
- **Kmeans**：==Kmeans类==
  - 输入：
    - k：类别个数
    - max_iterations：最大迭代次数
    - varepsilon：阈值

#### 结果如下

![output4](C:\Users\24857\Desktop\output4.png)

------

### 谱聚类.ipynb（以函数风格编写的代码）

#### 代码说明：

- mindspore没改成功+1，太难改了

#### 函数说明：

- **get_data**() ： ==获取数据并可视化展示==
  - 输入：
    - None
  - 输出：
    - x：横坐标

- **euclidDistance**(x1, x2, sqrt_flag=False)：==计算两坐标间的欧氏距离==
  - 输入：
    - x1：第一个点
    - x2：另一个点
    - sqrt_flag：是否需要开方
  - 输出：
    - x：横坐标

- **calEuclidDistanceMatrix**(X)：==初始化邻接矩阵，以距离作为权重==

  - 输入：
    - X：样本的X坐标值

  - 输出：
    - S：以距离作为权重的邻接矩阵

- **myKNN**(S, k, sigma=1.0)：==K近邻法构建邻接矩阵，$W_{ij}=exp(-\frac{||x_i-x_j||^2_2}{2\sigma ^2})$==

  - 输入：
    - S：上个函数中得到欧式距离邻接矩阵
    - k：前k个距离最小
    - sigma：系数
  - 输出：
    - A：K近邻计算过后的邻接矩阵

- **calLaplacianMatrix**(adjacentMatrix)：==计算拉普拉斯矩阵（度矩阵+拉普拉斯矩阵+标准化）==

  - 输入：
    - adjacentMatrix：上一个函数中得到的k近邻邻接矩阵
  - 输出：
    - 标准化的拉普拉斯矩阵

- **normlize**(L_sys)：==特征向量标准化==

  - 输入：
    - L_sys：上一个函数中得到的标准化拉普拉斯矩阵
  - 输出：
    - H：$n*k$ 维的特征矩阵

#### 结果如下：

![output5](C:\Users\24857\Desktop\output5.png)