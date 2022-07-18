'''
@ file function: 词嵌入向量
@ author: 王中琦
@ date: 2022/4/20
''' 

from gensim.models.word2vec import Word2Vec
from numpy import array


def count_feature(file_name) -> array:
    '''
    文件主函数，计算文本词语特征
    '''
    all_sentence = []

    with open(file_name, 'r', encoding='utf-8') as inp:
        for line in inp.readlines():  # 读入每行文字
            all_words = []
            line = line.split('\t')
            tmp = line[3][:-1]


            # all_words.append(word.split("/")[0])
            # all_words = seg_word(all_words)
            all_words = list(tmp)
            all_sentence.append(all_words)

    
    print("读入数据完毕")

    return all_sentence


def model_train(token_list):
    num_features = 50 # 特征维度
    min_word_count = 1  # 词语最小出现出现频率
    num_workers = 1 # numworker读取器数量
    window_size = 5 # 窗口大小为5
    subsampling = 1e-3 

    model = Word2Vec(
        token_list,
        workers=num_workers,
        vector_size=num_features,
        min_count=min_word_count,
        window=window_size,
        sample=subsampling,
        epochs=10, # 训练轮数
        sg=1
    )

    model.init_sims(replace=True)
    model_name = "my_word2vec_skip" # 保存到该文件中
    model.save(model_name)

    return True


def main():
    file_name = "./data/data_train.txt"
    token_list = count_feature(file_name)

    if(model_train(token_list)):
        print("训练完成")

    model = Word2Vec.load("my_word2vec_skip") # 读取word2vector模型

    # 查看与目标词最相近（通过计算余弦相似度）的词语
    for e in model.wv.most_similar(positive=['病'], topn=10):
        print(e[0], e[1]) # 输出词语及其向量


if __name__ == "__main__":
    main()
