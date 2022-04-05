'''
@ author: 王中琦
使用mindspore.numpy进行修改，替换部分原有numpy类型改为mindspore的tensor类型
'''
import numpy
import matplotlib.pyplot as plt
import mindspore.numpy as np



def J(theta):
    try:
        return (theta-2.5)**2-1
    except:
        return float('inf')

def dJ(theta):
    return 2*(theta - 2.5)


def gradient_descent(initial_theta, theta_history, n_iter = 1e4, epsilon=1e-8, eta=0.01):
    theta = initial_theta
    theta_history.append(initial_theta)
    i_iter = 0
    while i_iter < n_iter:
        gradient = dJ(theta)
        last_theta = theta
        theta = theta - eta*gradient
        theta_history.append(theta)
        
        if(abs(J(theta) - J(last_theta)) < epsilon):
            break
        
        i_iter+=1

    return theta_history

def plot_theta_history(plot_x,theta_history):
    plot_x = plot_x.asnumpy()
    plt.plot(plot_x,J(plot_x))
    plt.plot(numpy.array(theta_history),J(numpy.array(theta_history)))
    plt.show()

if __name__ == "__main__":
    # 这里使用了mindspore版的numpy类型
    plot_x = np.linspace(-1., 6., 141)

    theta_history = []

    theta_history,gradient_descent(0,theta_history,epsilon=1e-8,eta=0.8)
    plot_theta_history(plot_x,theta_history)