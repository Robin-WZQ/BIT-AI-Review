'''
@ author: 王中琦
使用mindspore的Tensor进行修改，并使用了自己定义的accuary计算机制
'''

import random
from mindspore import Tensor
import numpy as np
# 数据生成
from sklearn import datasets

# 寻找eps领域内的点
def findNeighbor(j , X, eps):
    X = X.asnumpy()
    N=[]
    for p in range(X.shape[0]): # 找到所有邻域内对象
        temp = np.sqrt(np.sum(np.square(X[j]-X[p])))
        temp = Tensor(temp)
        if(temp<eps):              # 欧式距离
            N.append(p)
    return N

def dbscan(X, eps, min_Pts):
    '''
    input:X(Tensor):样本数据
          eps(float):eps邻域半径
          min_Pts(int):eps邻域内最少点个数
    output:cluster(list):聚类结果
    '''
    k = -1
    NeighborPts = [] #array,某点领域内的对象
    Ner_NeighborPts = []
    fil = [] # 初始时已访问对象列表
    gama = [x for x in range(len(X))] # 初始时将所有点标记为未访问
    cluster = [-1 for y in range(len(X))]
    
    while len(gama)>0:
        j = random.choice(gama)
        gama.remove(j)
        fil.append(j)
        NeighborPts = findNeighbor(j, X, eps)
        if len(NeighborPts) < min_Pts:
            cluster[j] = -1
        else:
            k=k+1
            cluster[j] = k
            for i in NeighborPts:
                if i not in fil:
                    gama.remove(i)
                    fil.append(i)
                    Ner_NeighborPts = findNeighbor(i, X, eps)
                    if len(Ner_NeighborPts) >= min_Pts:
                        for a in Ner_NeighborPts:
                            if a not in NeighborPts:
                                NeighborPts.append(a)
                    if(cluster[i] == -1):
                        cluster[i] = k
                    
    return cluster

def score(y_predict,y_test):
    score = y_test == y_predict
    score = score.sum()
    return score

if __name__ == "__main__":
    X,y = datasets.make_moons(n_samples=100, noise=0.005, random_state=666)

    X = Tensor(X.astype(np.float32))

    # dbscan模型预测
    y_pred = dbscan(X, eps=0.5, min_Pts=10)
    # 计算吻合度
    acc = score(y, y_pred)

    print("聚类的吻合度：{:.2f}%".format(acc))  



