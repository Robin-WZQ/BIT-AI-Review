import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTM_Attention(nn.Module):
    def __init__(self, configs):
        super(BiLSTM_Attention, self).__init__()
        vocab_size = configs.vocab_size # 词表长度
        embedding_dimension = configs.embedding_dimension # 嵌入维度
        label_num = configs.label_num # 标签种类数目
        batch_size = configs.batch_size
        hidden_dim = configs.out_dimension

        self.embed = nn.Embedding(vocab_size, embedding_dimension)
        self.encoder = nn.LSTM(input_size=embedding_dimension, 
                                hidden_size=hidden_dim//2, 
                                num_layers=1,
                                batch_first=True,
                                bidirectional=True)

        self.decoder = nn.Linear(hidden_dim, label_num)

        self.dropout_emb = nn.Dropout(p=0.3)
        self.dropout_lstm = nn.Dropout(p=0.4)
        self.dropout_att = nn.Dropout(p=0.5)

        self.att_weight = nn.Parameter(torch.randn(batch_size,1,hidden_dim))

    def attention(self,H):
        H = H.permute(0,2,1)
        M = torch.tanh(H) # 非线性变换 size:(batch_size,hidden_dim,seq_len)
        a = F.softmax(torch.bmm(self.att_weight,M),dim=2) # a.Size : (batch_size,1,seq_len)
        a = torch.transpose(a,1,2) # (batch_size,seq_len,1)
        return torch.bmm(H,a) # (batch_size,hidden_dim,1)

    def forward(self, inputs):
        # inputs的形状是(batch_size,seq_len)
        embeddings = self.embed(inputs)
        embeddings = self.dropout_emb(embeddings)
        # 提取词特征，输出形状为(batch_size,seq_len,embedding_dim)
        # rnn.LSTM只返回最后一层的隐藏层在各时间步的隐藏状态。
        lstm_out, _ = self.encoder(embeddings)  # output, (h, c)
        # lstm_out形状是(batch_size,seq_len, 2 * num_hiddens)
        lstm_out = self.dropout_lstm(lstm_out)
        
        # Attention过程
        att_out = torch.tanh(self.attention(lstm_out))
        att_out = self.dropout_att(att_out)
        # Attention结束
        
        outs = self.decoder(att_out.squeeze(2))
        return outs
