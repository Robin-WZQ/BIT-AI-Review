import torch


def alltest(data_loader_test,device,model,batchsize):
    # 测试函数，需要完成的任务有：根据测试数据集中的数据，逐个对其进行预测，生成预测值。
    # 将结果按顺序写入txt文件中，下面一行不必保留

    f = open("predict_labels_1120190892.txt","a")

    model.eval() #模型切换到测试模式，保证dropout和BN层失效

    with torch.no_grad():  # 该函数的意义需在实验报告中写明
        for index, data in enumerate(data_loader_test, 0):
            data = data.to(device) #数据转到相应设备
            result = model(data)
            for i in range(batchsize):
                try: #这里的异常处理是为了阻止数据最后不到batchsize大小（整除不尽，导致 out of index）
                    out = result[i].argmax(dim=0) # 选取10个数里面最大的（选最大概率）
                    f.write(str(out.item())+"\n")
                except:
                    pass

    f.close()