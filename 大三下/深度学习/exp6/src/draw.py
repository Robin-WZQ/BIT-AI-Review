# -*- coding: utf-8 -*-
from re import X
import numpy as np  
import matplotlib.mlab as mlab  
import matplotlib.pyplot as plt 
import json 


def plot_loss(data1,data2,steps):
    plt.figure()
    plt.plot(steps, data1,color='red')
    plt.plot(steps, data2,color='blue')
    plt.legend(['Discrimitor Loss','Generator Loss'],loc='upper right')
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.show()

if __name__ == "__main__":
    f = open("log3.txt",'r').readlines()
    d_l = []
    g_l = []
    steps = []

    for i in range(len(f)):
        current = f[i].rstrip("\n").split(",")
        step = current[0][6:]
        d_loss = current[1][4:]
        g_loss = current[2][3:]

        steps.append(eval(step))
        d_l.append(eval(d_loss))
        g_l.append(eval(g_loss))


    plot_loss(d_l,g_l,steps)
