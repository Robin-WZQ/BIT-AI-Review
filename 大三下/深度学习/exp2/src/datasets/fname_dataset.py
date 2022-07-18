from torch.utils.data import Dataset
import os
from PIL import Image

# 数据加载
class CifarDataset(Dataset):
    def __init__(self,dataset,transform=None):
        # 添加数据集的初始化内容
        self.path = "Dataset" # 文件路径
        self.dataset = dataset # train or val or test
        self.transform = transform # 变换方式(数据增强)

        self.label = [] # 标签定义
        self.fname = [] # 图片的文件路径

        file_name = os.path.join(self.path,self.dataset+"set.txt")
        f = open(file_name,"r")
        listOfLines  =  f.readlines()

        for file in  listOfLines:
            if self.dataset != "test":
                file = file.split(" ")
                self.fname.append(os.path.join(self.path,"image",file[0]))
                self.label.append(eval(file[1].rstrip('\n'))) # 别忘读入不是str类型
            else:
                self.fname.append(os.path.join(self.path,"image",file.rstrip('\n')))
                self.label.append(0)
        
        f.close()

        assert(len(self.fname)==len(self.label)) # 判断读入图片和标签数量是否一致


    def __getitem__(self, index):
        # 添加getitem函数的相关内容
        if self.dataset != "test": #如果是测试集，它没有label所以需要特别处理
            image_name,image_label = self.fname[index],self.label[index]
            image = Image.open(image_name) 
            image = self.transform(image)   # 数据增强
            return image,image_label 
        else:
            image_name = self.fname[index]
            image = Image.open(image_name) 
            image = self.transform(image)   
            return image

    def __len__(self):
        # 添加len函数的相关内容
        return len(self.fname) #返回数据长度