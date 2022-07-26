### Exp1-关键代码

> 主要学习如何导入数据

```Python
class HandWritingNumberRecognize_Dataset(Dataset):
    def __init__(self,opt,dataset,transform=None):
        self.path = opt.file_path #文件路径
        self.dataset = dataset #train or val or test
        self.transform = transform # 变换（这里是totensor）

        self.label = [] # 标签定义
        self.fname = [] # 图片的文件路径

        folder_path = os.path.join(self.path,self.dataset,"images") # 获取图片文件夹
        path_list = os.listdir(folder_path) # 得到文件夹下所有问价
        path_list.sort(key=lambda x:int((x.split('.')[0]).split("_")[1])) # 排序
        
        for fname in path_list:
            self.fname.append(os.path.join(folder_path,fname)) # 导入所有文件路径

        if self.dataset != "test":
            label_file = os.path.join(self.path,self.dataset,"labels_"+self.dataset+".txt")
            with open(label_file) as f:
                self.label = f.readlines() # 读入每行
            for i in range(0, len(self.label)):
                self.label[i] = eval(self.label[i].rstrip('\n')) #去掉换行符，转成数字
            f.close() #关闭文件

    def __getitem__(self, index):
        if self.dataset != "test": #如果是测试集，它没有label所以需要特别处理
            image_name,image_label = self.fname[index],self.label[index]
            image = Image.open(image_name) 
            image = self.transform(image)   #转成tensor
            return image,image_label 
        else:
            image_name = self.fname[index]
            image = Image.open(image_name) 
            image = self.transform(image)   
            return image

    def __len__(self):
        return len(self.fname) #返回数据长度
```

