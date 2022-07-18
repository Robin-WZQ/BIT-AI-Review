import pandas as pd
import json
import torch
import nltk
import numpy as np

class WikiDataset():
    def __init__(self,dataset,opt):
        '''
            __init__()函数的内容：
            1.读取数据集
            2.拼接问句与表头
        '''
        self.max_len = opt.max_length
        with open('word2index_input.json', 'r') as fp:
            self.word2index_input = json.load(fp)
        with open('word2index_output.json', 'r') as fp:
            self.word2index_output = json.load(fp)

        self.dataset_name = "data/tokenized_" + dataset + ".jsonl"
        self.tabel_name = "data/tokenized_" + dataset +'.tables.jsonl'
        lines = pd.read_json(self.dataset_name, lines=True)
        lines2 = pd.read_json(self.tabel_name, lines=True)

        self.pairs = []
        for idx, row in lines.iterrows():
            tokens_qu = row["tokenized_question"] # NL问句
            tokens_sql = row["tokenized_query"] # SQL问句 

            id = row['table_id'] # 表头ID
            head = lines2.loc[lines2['id'] == id]
            head = head['header'].tolist()[0]
            self.pairs.append([tokens_qu, tokens_sql,head])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pass
        '''
            __getitem__()函数的内容：
                根据index读取输入输出序列，并根据词典转化为对应
            的id序列
        '''
        pair_current = self.pairs[index] #读取当前问句与对应sql，str类型
        input_sequence = [] # 输入ID序列
        output_sequence = [] # 输出ID序列
        input = pair_current[0] # 问句
        input.append('EOS') # EOS_token
        # 注意输出最后不需要EOS，因为输入阶段需要补零之后，所以EOS很重要
        output = pair_current[1] # SQL语句
        head = pair_current[2] # 表头
        for head_name in head:
            for token in nltk.word_tokenize(head_name):
                input.append(token) # 将表头添加至问句后面
            input.append('EOS') 
        for token in input:
            try: # 异常处理解决未登录词的问题
                id = self.word2index_input[token]
            except:
                id = len(self.word2index_input) + 1 # 将未登录词设置为此表长度+1
            input_sequence.append(id) # 转成ID序列
        for token in output:
            try: # 异常处理解决未登录词的问题
                id = self.word2index_output[token]
            except:
                id = len(self.word2index_output) + 1 # 将未登录词设置为此表长度+1
            output_sequence.append(id) # 转成ID序列
        input_len = len(input_sequence) # 输入序列长度
        output_len = len(output_sequence) # 输出序列长度
        if input_len > self.max_len: # 将序列补齐（补零）
            input_sequence = input_sequence[:self.max_len]
        else:
            for i in range(self.max_len - input_len):
                input_sequence.append(0)
        if output_len > self.max_len:
            output_sequence = output_sequence[:self.max_len]
        else:
            for i in range(self.max_len - output_len):
                output_sequence.append(0)
        input_sequence = np.array(input_sequence) # 转成Numpy类型
        output_sequence = np.array(output_sequence) # 同理

        return (input_sequence,output_sequence,input_len,output_len)



        
