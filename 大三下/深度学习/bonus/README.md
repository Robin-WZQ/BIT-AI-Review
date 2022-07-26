# DL Bonus Exp 

> 因为是开放性的作业，我这里分享一下我们团队实现的代码

## Mission - 任务描述
中文政务领域关系抽取任务，给定一句话，求出该段话内所有的实体级对应关系。

## Environment - 环境

- Linux&Windows

- python = 3.8.3
- pytorch = 1.10.2+cu113
- pytorch_pretrained_bert
- tqdm

## Usage - 使用方法

1. 安装依赖:

```Shell
pip install -r requirements.txt
```

2. 下载预训练模型

Roberta_base: [chinese_roberta_wwm_ext_pytorch.zip - Google 云端硬盘](https://drive.google.com/file/d/1eHM3l4fMo6DsQYGmey7UZGiTmQquHw25/view)

Roberta_Large: [chinese_roberta_wwm_large_ext_pytorch.zip - Google 云端硬盘](https://drive.google.com/file/d/1-2vEZfIFCdM1-vJ3GD6DlSyKT4eVXMKq/view)

3. 保证目录如下所示：

```
| Chinese_roberta_wwm_large_ext_pytorch # 中文预训练模型
----| bert_config.json
----| pytorch_model.bin
----| vocab.txt
| DL
----| data_train.json # 训练集
----| data_dev.json # 开发集
----| data_test_process.json # 测试集
----| schema.json # 关系约束
| record
draw.py # 绘图函数
roberta_base.py # base模型代码
roberta_large.py # Large模型代码
README.md # 说明文件
requirements.txt # 配置文件
```

4. 运行
   
    - roberta_base
    
    ```Shell
    python roberta_base.py
    ```
    
    - roberta_large
    
    ```
    python roberta_large.py
    ```
    
5. 生成文件
- model.pth # 训练完的模型
- dev_pred.json # 开发集预测结果
- RE_pred.json # 测试集预测结果
