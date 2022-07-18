import torch
import torch.nn as nn
import json

def get_numpy_word_embed(word2ix):
    row = 0
    file = 'data/glove.6B.200d.txt' # Glove词向量
    words_embed = {}
    with open(file, mode='r',encoding='utf-8')as f:
        lines = f.readlines()
        for line in lines:
            # print(line)
            # print(len(line.split()))
            line_list = line.split()
            word = line_list[0]
            embed = line_list[1:]
            embed = [float(num) for num in embed]
            words_embed[word] = embed
            if row > 50000:
                break
            row += 1
    # word2ix = {}
    ix2word = {ix: w for w, ix in word2ix.items()}
    id2emb = {}
    for ix in range(len(word2ix)):
        if ix2word[ix+1] in words_embed:
            id2emb[ix+1] = words_embed[ix2word[ix+1]]
        else:
            id2emb[ix+1] = [0.0] * 200
    data = [id2emb[ix+1] for ix in range(len(word2ix))]
 
    return data

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size,device):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        with open('word2index_input.json', 'r') as fp:
            word2ix = json.load(fp)
        numpy_embed = get_numpy_word_embed(word2ix)
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(numpy_embed))
        self.gru = nn.GRU(input_size= hidden_size, hidden_size =  hidden_size, batch_first=True,bidirectional=False)


    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)

