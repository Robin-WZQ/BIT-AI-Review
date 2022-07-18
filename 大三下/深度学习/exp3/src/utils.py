'''
绘图函数
'''

# -*- coding: utf-8 -*-
from re import X
import numpy as np  
import matplotlib.mlab as mlab  
import matplotlib.pyplot as plt 
import json 

def label_distribution():
    '''
    标签种类分布绘图函数
    '''
    X=[i for i in range(0,44)]
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
        Y.append(d[i])
    # print(Y)
    
    plt.figure()
    plt.bar(X,Y,0.4,color="green")
    plt.xlabel("labels")
    plt.ylabel("distributions")
    plt.title("Label distribution")


    plt.show()  
    plt.savefig("Label_distribution.jpg")

def plot_acc(data1,data2):
    plt.figure()
    plt.plot(range(len(data1)), data1,color='red')
    plt.plot(range(len(data2)), data2,color='blue')
    plt.legend(['train_acc','val_acc'],loc='upper left')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    #标记极值点
    y1=max(data1)
    x1=data1.index(y1)
    show_max1='['+str(x1)+' '+str( round(y1,4) )+']'
    plt.scatter(x1,y1,marker='*',c='k')
    plt.annotate(show_max1,xy=(x1,y1),xytext=(x1,y1))
    
    y2=max(data2)
    x2=data2.index(y2)
    show_max2='['+str(x2)+' '+str( round(y2,4) )+']'
    plt.scatter(x2,y2,marker='*',c='r')
    plt.annotate(show_max2,xy=(x2,y2),xytext=(x2,y2))
    plt.show()

def plot_loss(data1):
    plt.figure()
    plt.plot(range(len(data1)), data1,color='red')
    plt.legend(['train_loss'],loc='upper right')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    #标记极值点
    y1=max(data1)
    x1=data1.index(y1)
    plt.scatter(x1,y1,marker='*',c='k')
    plt.show()

def plot_F1(data1,data2):
    fig = plt.figure()
    plt.plot(range(len(data1)), data1,color='red')
    plt.plot(range(len(data2)), data2,color='blue')
    plt.legend(['train_F1','val_F1'],loc='down right')
    plt.xlabel('epoch')
    plt.ylabel('F1')
    #标记极值点
    y1=max(data1)
    x1=data1.index(y1)
    show_max1='['+str(x1)+' '+str( round(y1,4) )+']'
    plt.scatter(x1,y1,marker='*',c='k')
    plt.annotate(show_max1,xy=(x1,y1),xytext=(x1,y1))

    y2=max(data2)
    x2=data2.index(y2)
    show_max2='['+str(x2)+' '+str( round(y2,4) )+']'
    plt.scatter(x2,y2,marker='*',c='r')
    plt.annotate(show_max2,xy=(x2,y2),xytext=(x2,y2))
    plt.show()

def plot_F12(data1,data2,data3,data4):
    fig = plt.figure()
    plt.plot(range(len(data1)), data1,color='red')
    plt.plot(range(len(data2)), data2,color='blue')
    plt.plot(range(len(data3)), data3,color='yellow')
    plt.plot(range(len(data4)), data4,color='brown')
    plt.legend(['channel=1','channel=2','channel=3','channel=4'],loc='down left')
    plt.xlabel('channel')
    plt.ylabel('val_acc')
    #标记极值点
    y1=max(data1)
    x1=data1.index(y1)
    show_max1='['+str(x1)+' '+str( round(y1,4) )+']'
    plt.scatter(x1,y1,marker='*',c='k')
    # plt.annotate(show_max1,xy=(x1,y1),xytext=(x1,y1))

    y2=max(data2)
    x2=data2.index(y2)
    show_max2='['+str(x2)+' '+str( round(y2,4) )+']'
    plt.scatter(x2,y2,marker='*',c='r')
    # plt.annotate(show_max2,xy=(x2,y2),xytext=(x2,y2))

    y3=max(data3)
    x3=data3.index(y3)
    show_max3='['+str(x3)+' '+str( round(y3,4) )+']'
    plt.scatter(x3,y3,marker='*',c='r')
    # plt.annotate(show_max3,xy=(x3,y3),xytext=(x3,y3))

    y4=max(data4)
    x4=data4.index(y4)
    show_max4='['+str(x4)+' '+str( round(y4,4) )+']'
    plt.scatter(x4,y4,marker='*',c='r')
    # plt.annotate(show_max4,xy=(x4,y4),xytext=(x4,y4))

    plt.show()


if __name__ == "__main__":
    f1 = open("record/final/acc.txt",'r').readlines()
    f2 = open("record/final/loss.txt",'r').readlines()
    f3 = open("record/final/val_acc.txt",'r').readlines()
    # f4 = open("record/tongdao/val_F1_23.txt",'r').readlines()
    # f5 = open("record/tongdao/val_F1_234.txt",'r').readlines()
    # f6 = open("record/tongdao/val_F1_2345.txt",'r').readlines()
    
    X = []
    train_F1 = []
    val_F1_1 = []
    val_F1_2 = []
    val_F1_3 = []
    val_F1_4 = []
    for i in range(len(f1)):
        current = f1[i].rstrip("\n").rstrip(")").replace("(",'').split(",")
        data1 = f2[i].rstrip("\n").rstrip(")").replace("(",'').split(",")
        current2 = f3[i].rstrip("\n").rstrip(")").replace("(",'').split(",")
        # current4 = f4[i].rstrip("\n").rstrip(")").replace("(",'').split(",")
        # current5 = f5[i].rstrip("\n").rstrip(")").replace("(",'').split(",")
        # current6 = f6[i].rstrip("\n").rstrip(")").replace("(",'').split(",")

        # X.append(eval(current[0]))
        train_F1.append(eval(current[1]))
        val_F1_1.append(eval(current2[1]))
        X.append(eval(data1[1]))
        # val_F1_2.append(eval(current4[1])+0.13)
        # val_F1_3.append(eval(current5[1])+0.13)
        # val_F1_4.append(eval(current6[1])+0.13)

    # plot_F12(val_F1_1,val_F1_2,val_F1_3,val_F1_4)
    plot_loss(X)
    plot_F1(train_F1,val_F1_1)
