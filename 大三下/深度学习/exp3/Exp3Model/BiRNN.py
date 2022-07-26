import torch
import torch.nn as nn

class BiRNN(nn.Module):
    def __init__(self, configs):
        super(BiRNN, self).__init__()
        vocab_size = configs.vocab_size # 词表长度
        embedding_dimension = configs.embedding_dimension # 嵌入维度
        label_num = configs.label_num # 标签种类数目
        max_len = configs.max_sentence_length # 最大长度
        pos_dimension = configs.pos_embedding_dimension # 位置嵌入维度

        self.embed = nn.Embedding(vocab_size, embedding_dimension)
        self.embed_pos = nn.Embedding(300, pos_dimension)

        # bidirectional设为True即得到双向循环神经网络
        self.encoder = nn.RNN(input_size=embedding_dimension + 2*pos_dimension, 
                                hidden_size=64, 
                                num_layers=1,
                                batch_first=True,
                                bidirectional=False)
        self.MaxPooling = nn.MaxPool1d(max_len)
        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.decoder = nn.Linear(64, label_num)

    def forward(self,pos1,pos2,sentences):
        sentences = self.embed(sentences)
        pos1 = self.embed_pos(pos1) # 此处pos必须为正数，否则会报错
        pos2 = self.embed_pos(pos2)
        Position_Feature = torch.concat((pos1,pos2),dim=2)
        Sentence_Feature_init = torch.concat((sentences,Position_Feature),dim=2)
        outputs, _ = self.encoder(Sentence_Feature_init)
        outputs = outputs.permute(0, 2, 1)
        max = self.MaxPooling(outputs)
        outs = self.decoder(max.squeeze(2))

        return outs