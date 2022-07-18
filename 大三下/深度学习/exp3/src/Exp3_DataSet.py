import json

import torch
from torch.utils.data import Dataset

from Exp3_Config import Training_Config


# 训练集和验证集
class TextDataSet(Dataset):
    def __init__(self, filepath,configs):
        with open("word2id.json",'r',encoding='utf-8') as load_f:
            self.lookup_table = list(json.load(load_f))

        with open("data/rel2id.json",'r',encoding='utf-8') as load_f:
            self.rel2id_table = json.load(load_f) 

        lines = open(filepath, 'r', encoding='utf-8').readlines()
        self.original_data = []
        self.classes_for_all_imgs = []
        self.max_len = configs.max_sentence_length
        for line in lines:
            tmp = {}
            line = line.split('\t')
            tmp['head'] = line[0]
            if line[0] == 'subject_placeholder':
                continue
            tmp['tail'] = line[1]
            if line[1] == 'object_placeholder':
                continue
            tmp['relation'] = line[2]
            tmp['text'] = line[3][:-1]
            self.original_data.append(tmp)
            self.classes_for_all_imgs.append( self.rel2id(line[2]))


    def __getitem__(self, index):
        entity1 = self.original_data[index]['head']
        entity2 = self.original_data[index]['tail']
        relation = self.rel2id(self.original_data[index]['relation'])
        sentence = self.original_data[index]['text']

        pos = sentence.find(entity1) # 找到第一个实体所在位置
        pos1,pos2 = self.cal_pos(entity1,entity2,sentence) # 计算位置特征向量
        out_sentence = self.word2id(sentence) # 计算文本特征向量ID
        out_sentence = self.cut(out_sentence,pos) # 文本截断
        entity1,entity2 = self.word2id(entity1),self.word2id(entity2) # 实体特征ID
        entity1 = entity1 + [0]*(self.max_len-len(entity1)) # 实体维度补齐
        entity2 = entity2 + [0]*(self.max_len-len(entity2)) # 实体维度补齐
        pos1 = self.cut(pos1,pos)
        pos2 = self.cut(pos2,pos)
        # 注意需要转换成LongTensor类型
        entity1 = torch.LongTensor(entity1) 
        entity2 = torch.LongTensor(entity2)
        out_sentence = torch.LongTensor(out_sentence)
        pos1 = torch.LongTensor(pos1)
        pos2 = torch.LongTensor(pos2)

        # print(entity1.shape,entity2.shape,relation.shape,out_sentence.shape,pos1.shape,pos2.shape)

        return (entity1,entity2,relation,out_sentence,pos1,pos2)

    def __len__(self):
        return len(self.original_data)
    
    def get_classes_for_all_imgs(self):
        '''
        标签平衡是需要计算
        '''
        return self.classes_for_all_imgs

    def cut(self,sentence,position_entity):
        # 如果该句子大于最大长度
        if len(sentence)>=self.max_len:
            if position_entity+self.max_len > len(sentence): # 如果加上最大长度后超出句子长度
                cut_sentence = sentence[len(sentence)-self.max_len:]
            else:
                cut_sentence = sentence[position_entity:position_entity+self.max_len]
        else:
            cut_sentence = sentence + [0]*(self.max_len-len(sentence))

        return cut_sentence
    
    def cal_pos(self,entity1,entity2,sentence):
        position_entity1 = sentence.find(entity1) # 找到第一个实体所在位置
        position_entity2 = sentence.find(entity2) # 找到第二个实体所在位置

        pos1_list = []
        pos2_list = []
        length = len(sentence)
        for i in range(length):
            pos1 = abs(i - position_entity1)
            pos2 = abs(i - position_entity2)
            pos1_list.append(pos1)
            pos2_list.append(pos2)
        
        return pos1_list,pos2_list

        
    def word2id(self,instr):
        inp = []
        for ch in instr:
            try:
                inp.append(self.lookup_table.index(ch))
            except:
                inp.append(0)

        return inp

    def rel2id(self,rel):
        num = self.rel2id_table[1][rel]
        num = torch.tensor(num)
        return num


# 测试集是没有标签的，因此函数会略有不同
class TestDataSet(Dataset):
    def __init__(self, filepath,configs):
        lines = open(filepath, 'r', encoding='utf-8').readlines()
        self.original_data = []
        self.max_len = configs.max_sentence_length
        for line in lines:
            tmp = {}
            line = line.split('\t')
            tmp['head'] = line[0]
            tmp['tail'] = line[1]
            tmp['text'] = line[2][:-1]
            self.original_data.append(tmp)

        with open("word2id.json",'r',encoding='utf-8') as load_f:
            self.lookup_table = list(json.load(load_f))

        with open("data/rel2id.json",'r',encoding='utf-8') as load_f:
            self.rel2id_table = json.load(load_f) 

    def __getitem__(self, index):
        entity1 = self.original_data[index]['head']
        entity2 = self.original_data[index]['tail']
        sentence = self.original_data[index]['text']

        pos = sentence.find(entity1) # 找到第一个实体所在位置
        pos1,pos2 = self.cal_pos(entity1,entity2,sentence)
        out_sentence = self.word2id(sentence)
        out_sentence = self.cut(out_sentence,pos)
        entity1,entity2 = self.word2id(entity1),self.word2id(entity2)
        entity1 = entity1 + [0]*(self.max_len-len(entity1))
        entity2 = entity2 + [0]*(self.max_len-len(entity2))
        pos1 = self.cut(pos1,pos)
        pos2 = self.cut(pos2,pos)
        entity1 = torch.LongTensor(entity1)
        entity2 = torch.LongTensor(entity2)
        out_sentence = torch.LongTensor(out_sentence)
        pos1 = torch.LongTensor(pos1)
        pos2 = torch.LongTensor(pos2)


        return (entity1,entity2,out_sentence,pos1,pos2)

    def __len__(self):
        return len(self.original_data)

    def cut(self,sentence,position_entity):
        # 如果该句子大于最大长度
        if len(sentence)>=self.max_len:
            if position_entity+self.max_len > len(sentence): # 如果加上最大长度后超出句子长度
                cut_sentence = sentence[len(sentence)-self.max_len:]
            else:
                cut_sentence = sentence[position_entity:position_entity+self.max_len]
        else:
            cut_sentence = sentence + [0]*(self.max_len-len(sentence))

        return cut_sentence
    
    def cal_pos(self,entity1,entity2,sentence):
        position_entity1 = sentence.find(entity1) # 找到第一个实体所在位置
        position_entity2 = sentence.find(entity2) # 找到第二个实体所在位置

        pos1_list = []
        pos2_list = []
        length = len(sentence)
        for i in range(length):
            pos1 = abs(i - position_entity1)
            pos2 = abs(i - position_entity2)
            pos1_list.append(pos1)
            pos2_list.append(pos2)
        
        return pos1_list,pos2_list

        
    def word2id(self,instr):
        inp = []
        for ch in instr:
            try:
                inp.append(self.lookup_table.index(ch))
            except:
                inp.append(0)

        return inp

if __name__ == "__main__":
    config = Training_Config()
    train_dataset = TextDataSet(filepath="./data/data_train.txt",configs=config)
    print("训练集长度为：", len(train_dataset))

