import json, time 
import numpy as np 
from tqdm import tqdm 

from torch.utils.data import Dataset, DataLoader
from pytorch_pretrained_bert import BertModel, BertTokenizer
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import os
import matplotlib.pyplot as plt 
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# 读取json文件，即输入text及对应spo格式
def load_data(filename):
    D = {}
    with open(filename, 'r',encoding='utf-8') as f:
        all = json.load(f)
        for l in all:
            for relation in l['relations']:
                relationship = relation['type'].replace(" ","")
                D[relationship] = D.get(relationship, 0) + 1

    return D 

def label_distribution():
    
    X=[i for i in range(0,14)]
    Y = []
    num = 0
    # 加载数据集
    train_data = load_data('DL/data_train.json')

    for i in train_data.values():
        Y.append(i)
        num+=i
    print(num)
    
    plt.figure()
    plt.bar(X,Y,0.4,color="green")
    plt.xlabel("labels")
    plt.ylabel("distributions")
    plt.title("Label distribution")


    plt.show()  
    plt.savefig("Label_distribution.jpg")

    print(num)

def plot_loss(data1):
    plt.figure()
    plt.plot(range(len(data1)), data1,color='blue')
    plt.legend(['Large'],loc='down right')
    plt.xlabel('epoch')
    plt.ylabel('val_f1')
    #标记极值点
    y1=max(data1)
    x1=data1.index(y1)
    plt.scatter(x1,y1,marker='*',c='k')
    plt.show()

def plot_F1(data1,data2,data3):
    fig = plt.figure()
    plt.plot(range(len(data1)), data1,color='red')
    plt.plot(range(len(data2)), data2,color='blue')
    plt.plot(range(len(data3)), data3,color='green')
    plt.legend(['roberta-base','roberta-large','pipeline'],loc='lower right')
    plt.xlabel('epoch')
    plt.ylabel('val_F1')

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

    y3=max(data3)
    x3=data3.index(y3)
    show_max3='['+str(x3)+' '+str( round(y3,4) )+']'
    plt.scatter(x3,y3,marker='*',c='r')
    plt.annotate(show_max3,xy=(x3,y3),xytext=(x3,y3))

    plt.show()

if __name__ == "__main__":
    # label_distribution()
    f1 = open("record/val_f1_1.txt",'r').readlines()
    f2 = open("record/val_f1_3.txt",'r').readlines()
    f3 = open("record/val_f1_2.txt",'r').readlines()

    val_F1_1 = []
    val_F1_2 = []
    val_F1_3 = []

    for i in range(len(f1)):
        current1 = f1[i].rstrip("\n").rstrip(")").replace("(",'').split(",")
        val_F1_1.append(eval(current1[1]))
    for i in range(len(f2)):
        current2 = f2[i].rstrip("\n").rstrip(")").replace("(",'').split(",")
        val_F1_2.append(eval(current2[1]))
    for i in range(len(f3)):    
        current3 = f3[i].rstrip("\n").rstrip(")").replace("(",'').split(",")  
        val_F1_3.append(eval(current3[1]))
            
    # plot_loss(train_loss)
    plot_F1(val_F1_1,val_F1_2,val_F1_3)
