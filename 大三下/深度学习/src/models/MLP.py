import torch
import torch.nn as neural_funcs

class HandWritingNumberRecognize_Network(torch.nn.Module): # 继承父类
    def __init__(self):
        super(HandWritingNumberRecognize_Network, self).__init__()
        self.layer = neural_funcs.Sequential( #使用pytorch的容器进行加载模型
            neural_funcs.Linear(784, 600), #线性层
            neural_funcs.Dropout(p=0.5), # dropout防止过拟合
            neural_funcs.ReLU(), # 激活函数
            neural_funcs.Linear(600, 10))
            # 最后的结果一定要变为 10，因为数字的选项是 0 ~ 9

    def forward(self, input_data):
        input_data = input_data.view(input_data.size()[0], -1) #拉平
        input_data = self.layer(input_data)
        return input_data