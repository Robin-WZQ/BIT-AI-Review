import torch


# 定义 test 函数
def test(model,device,batchsize,test_loader):
    # 将预测结果写入result.txt文件中，格式参照实验1
    f = open("predict_labels_1120190892.txt","a")

    model.eval()

    with torch.no_grad():  
        for index, data in enumerate(test_loader, 0):
            data = data.to(device)
            result = model(data)
            for i in range(batchsize):
                try:
                    out = result[i].argmax(dim=0)
                    f.write(str(out.item())+"\n")
                except:
                    pass

    f.close()