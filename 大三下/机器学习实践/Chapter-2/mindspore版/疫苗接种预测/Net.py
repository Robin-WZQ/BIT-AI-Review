from mindspore import nn
from mindspore.common.initializer import Normal

class LinearNet(nn.Cell):
    def __init__(self):
        super(LinearNet,self).__init__()
        # 定义一个线形层，同时初始化权重和偏置
        self.fc=nn.Dense(1,1,Normal(0.02),Normal(0.02),has_bias=True) 
    
    def construct(self,x):
        x=self.fc(x)
        return x
