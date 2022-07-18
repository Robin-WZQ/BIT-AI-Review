# -*- coding: utf-8 -*-
from re import X
import numpy as np  
import matplotlib.mlab as mlab  
import matplotlib.pyplot as plt 
import json 

def plot_loss(data1):
    plt.figure()
    plt.plot(range(len(data1)), data1,color='red')
    plt.legend(['train_loss_cpn'],loc='upper right')
    plt.xlabel('step')
    plt.ylabel('loss')
    #标记极值点
    y1=max(data1)
    x1=data1.index(y1)
    plt.scatter(x1,y1,marker='*',c='k')
    plt.show()

if __name__ == "__main__":
    f = open("record/train_loss_cpynet.txt",'r').readlines()
    train_F = []
    for i in range(len(f)):
        current = f[i].rstrip("\n").rstrip(")").replace("(",'').split(",")
        train_F.append(eval(current[1]))

    plot_loss(train_F)
