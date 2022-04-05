'''
这里实在是太难用mindspore修改了，没有改
'''

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine
import numpy as np
from mindspore import Tensor


def euclidean_distance(one_sample, X):
    one_sample = one_sample.reshape(1,-1)
    distances = np.power(np.tile(one_sample, (X.shape[0],1)) - X, 2).sum(axis = 1)
    return distances

class Kmeans():
    '''
    Kmeans 的聚类算法
    Parameters:
    ------------
    k: int
        聚类的数目
    max_iterations: int
        最大迭代次数
    varepsilon： float
        判断是否收敛，如果上一次的所有k个聚类中心与本次的所有k个聚类中心的差都小于varpsilon，
        则说明算法已经收敛
    '''
    def __init__(self, k=1 ,max_iterations=500, varepsilon=0.0001):
        self.k = k
        self.max_iterations = max_iterations
        self.varepsilon = varepsilon
        np.random.seed(1)

    # 从所有样本中随机选取self.k样本作为初始的聚类中心
    def init_random_centroids(self,X):
        n_samples, n_features = np.shape(X)
        centroids = np.zeros((self.k,n_features))
        for i in range(self.k):
            centroid = X[np.random.choice(range(n_samples))]
            centroids[i] = centroid
        return centroids

    # 返回距离该样本最近的一个中心索引[0,self.k)
    def _closest_centroid(self, sample, centroids):
        distances = euclidean_distance(sample, centroids)
        closest_i = np.argmin(distances)
        return closest_i

    # 对所有样本进行归类，归类规则就是将该样本归类到预期最近的中心
    def create_clusters(self, centroids, X):
        clusters = [[] for _ in range(self.k)]
        for sample_i , sample in enumerate(X):
            centroid_i = self._closest_centroid(sample, centroids)
            clusters[centroid_i].append(sample_i)
        return clusters
    
    # 对中心进行更新
    def update_centroids(self, clusters, X):
        n_features = np.shape(X)[1]
        centroids = np.zeros((self.k, n_features))
        for i, cluster in enumerate(clusters):
            centroid = np.mean(X[cluster], axis=0)
            centroids[i] = centroid
        return centroids
    
    # 将所有样本进行归类，其所在的类别的索引就是其标签类别
    def get_cluster_labels(self, clusters,X):
        y_pred = np.zeros(np.shape(X)[0])
        for cluster_i, cluster in enumerate(clusters):
            for sample_i in cluster:
                y_pred[sample_i] = cluster_i
        return y_pred
    
    # 对整个数据集X进行Kmeans聚类，返回其聚类的标签
    def predict(self,X):
        # 从所有样本中随机选取self.k样本作为初始的聚类中心
        centroids = self.init_random_centroids(X)
        # 迭代，直到算法收敛（上一次的聚类中心和这一次的聚类中心几乎重合）或者达到最大迭代次数
        for _ in range(self.max_iterations):
            # 将所有进行归类，归类规则计时将该样归类到预期最近的中心
            clusters = self.create_clusters(centroids,X)
            former_centroids = centroids
            # 计算新的聚类中心
            centroids = self.update_centroids(clusters,X)
            # 如果聚类中心几乎没有任何变化，说明算法已经收敛，退出迭代
            diff = centroids - former_centroids
            if diff.any() < self.varepsilon:
                break
        return self.get_cluster_labels(clusters,X)

def main():
    wine = load_wine()
    scaler = StandardScaler()
    X = scaler.fit_transform(wine.data)
    y = wine.target
    km = Kmeans(k=3)
    y_pred = km.predict(X)

    y[y == 0] = -1
    y[y == 1] = -2
    y[y == 2] = -3

    y_pred[y_pred == 0] = -1
    y_pred[y_pred == 1] = -2
    y_pred[y_pred == 2] = -3

    # 计算吻合度
    acc = accuracy_score(y,y_pred)
    print("聚类的吻合度：{:.2f}".format(acc))

if __name__ == "__main__":
    main()