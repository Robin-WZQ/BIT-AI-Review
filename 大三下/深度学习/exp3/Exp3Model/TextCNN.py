import torch
import torch.nn as nn
from gensim.models.word2vec import Word2Vec


class TextCNN_Model(nn.Module):

    def __init__(self, configs):
        super(TextCNN_Model, self).__init__()

        vocab_size = configs.vocab_size # 词表长度
        embedding_dimension = configs.embedding_dimension # 嵌入维度
        label_num = configs.label_num # 标签种类数目
        pos_dimension = configs.pos_embedding_dimension # 位置嵌入维度
        out_dimension = configs.out_dimension # 输出维度
        max_len = configs.max_sentence_length # 最大长度
        kernel_sizes = [2,3,4,5]

        # 词嵌入和dropout
        self.embed = nn.Embedding(vocab_size, embedding_dimension)
        # words_vector = Word2Vec.load("my_word2vec_skip") # 读取word2vector模型
        # self.embed = nn.Embedding.from_pretrained(torch.from_numpy(words_vector.wv.vectors))
        self.embed_pos = nn.Embedding(300, pos_dimension)
        self.dropout = nn.Dropout(configs.dropout)


        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv1d(embedding_dimension + 2 * pos_dimension, out_dimension,
                             kernel_size=kernel_size,padding=int(kernel_size / 2)),
            nn.Tanh(),
            nn.MaxPool1d(max_len)
        ) for kernel_size in kernel_sizes])

        self.fc = nn.Linear(14256, label_num)        

    def forward(self, sentences, entity1, entity2, pos1, pos2):
        sentences = self.embed(sentences)
        entity1 = self.embed(entity1)
        entity2 = self.embed(entity2)
        pos1 = self.embed_pos(pos1) # 此处pos必须为正数，否则会报错
        pos2 = self.embed_pos(pos2)

        Lexical_Feature = torch.concat((entity1,entity2),dim=1)
        Position_Feature = torch.concat((pos1,pos2),dim=2)
        Sentence_Feature_init = torch.concat((sentences,Position_Feature),dim=2)
        conv = [conv(Sentence_Feature_init.permute(0, 2, 1)) for conv in self.convs]
        Sentence_Feature = torch.cat(conv, dim=1)
        self.dropout(Sentence_Feature)
        all_concat = torch.cat([Sentence_Feature.view(Sentence_Feature.size(0), -1), 
                    Lexical_Feature.view(Lexical_Feature.size(0), -1)], dim=1)
        out = self.fc(all_concat)
        
        return out
