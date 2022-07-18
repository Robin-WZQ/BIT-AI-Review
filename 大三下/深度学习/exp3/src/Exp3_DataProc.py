"""
预处理文件,提前将word->ID存到JSON里
"""
import json

if __name__ == '__main__':
    print("数据预处理开始......")
    filepath = "data/data_train.txt"
    lines = open(filepath, 'r', encoding='utf-8').readlines()
    original_data = []
    d = {}
    for line in lines:
        line = line.split('\t')
        line = list(line[3][:-1])
        for i in range(len(line)):
            original_data.append(line[i])
    word = set(original_data)
    lookup_table = list(word)
    for i in range(len(lookup_table)):
        d[lookup_table[i]] = i
    with open("word2id.json","w",encoding='utf-8') as f:
        json.dump(d,f,ensure_ascii=False,indent = 3)
        print("加载入文件完成...")