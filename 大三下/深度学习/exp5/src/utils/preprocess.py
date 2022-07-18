'''
@ file_name : pre_process
@ file_function : 制作词表并保存至本地
@ author : 王中琦
@ date : 2022/5/22 
'''
import pandas as pd
from collections import defaultdict
import json
import nltk

dataset_names = ['train','dev','test']
word2index_input = defaultdict(int)
word2index_input['SOS'] = 1
word2index_input['EOS'] = 2
index2word_input = {1: "SOS", 2: "EOS"}

word2index_output = defaultdict(int)
word2index_output['SOS'] = 1
word2index_output['EOS'] = 2
index2word_output = {1: "SOS", 2: "EOS"}

for name in dataset_names:
    dataset_name = "data/tokenized_" + name + ".jsonl"
    lines = pd.read_json(dataset_name, lines=True)
    pairs = []
    for idx, row in lines.iterrows():
        tokens_qu = row["tokenized_question"]
        tokens_sql = row["tokenized_query"]

        for token in tokens_qu:
            if token not in word2index_input:
                word2index_input[token] = len(word2index_input)+1
        
        for token in tokens_sql:
            if token not in word2index_output:
                word2index_output[token] = len(word2index_output)+1

for name in dataset_names:
    dataset_name = 'data/tokenized_' + name + '.tables.jsonl'
    lines = pd.read_json(dataset_name, lines=True)
    for idx, row in lines.iterrows():
        head = row['header']
        for head_name in head:
            for token in nltk.word_tokenize(head_name):
                if token not in word2index_input:
                    word2index_input[token] = len(word2index_input)+1
                if token not in word2index_output:
                    word2index_output[token] = len(word2index_output)+1

for key in word2index_input:
    index2word_input[word2index_input[key]] = key

for key in word2index_output:
    index2word_output[word2index_output[key]] = key

with open('word2index_input.json', 'w') as fp:
    json.dump(word2index_input, fp, indent=4, separators=(',', ':'))
with open('index2word_input.json', 'w') as fp:
    json.dump(index2word_input, fp, indent=4, separators=(',', ':'))
with open('word2index_output.json', 'w') as fp:
    json.dump(word2index_output, fp, indent=4, separators=(',', ':'))
with open('index2word_output.json', 'w') as fp:
    json.dump(index2word_output, fp, indent=4, separators=(',', ':'))