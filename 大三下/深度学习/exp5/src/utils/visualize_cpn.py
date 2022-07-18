import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import torch
from model.copynet_dbg import CopyEncoder, CopyDecoder
from opts import parse_opts
import json
import nltk
import numpy as np
from torch.autograd import Variable
import warnings
warnings.filterwarnings('ignore')

# ================== 读入词表 ===================
with open('index2word_output.json', 'r') as fp:
    index2word_output = json.load(fp) 
with open('word2index_input.json', 'r') as fp:
    word2index_input = json.load(fp)
# ================== 读入词表 ===================

def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

def decoder_initial(batch_size):
    decoder_in = torch.LongTensor(np.ones(batch_size,dtype=int))*2
    s = None
    w = None
    if torch.cuda.is_available():
        decoder_in = decoder_in.cuda()
    decoder_in = Variable(decoder_in)
    return decoder_in, s, w

def evaluateAndShowAttention(input_sentence,table):
    output_words, attentions = evaluate(input_sentence,table)
    attention_real = torch.zeros((attentions.shape[0],20))
    for i in range(attentions.shape[0]):
        attention_real[i] = attentions[i][:20]
    input = []
    for i in table:
        input.append(i)
        input.append("col")
    print('input =', input_sentence+" "+" ".join(input))
    print('output =', ' '.join(output_words))
    showAttention(input_sentence+" "+" ".join(input), output_words, attention_real)

def evaluate(input_sentence,table):
    '''
    测试函数
    '''
    opt = parse_opts()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    opt.device = device

    input = []
    for token in nltk.word_tokenize(input_sentence):
        input.append(word2index_input[token.lower()])
    input.append(2)
    for col in table:
        for token in nltk.word_tokenize(col):
            try:
                input.append(word2index_input[token])
            except:
                input.append(len(word2index_input)+1)
        input.append(2)
    
    for i in range(100-len(input)):
        input.append(0)
    input_out = np.expand_dims(np.array(input), axis=0)    
    x = torch.LongTensor(input).unsqueeze(0).to(device)

    # ================== 加载模型 ===================
    # 编码器
    encoder = CopyEncoder(len(word2index_input)+2, opt.embed_size, opt.hidden_size).to(device)
    encoder.load_state_dict(torch.load("saved_models/encoder.pth"))    
    # 解码器
    decoder = CopyDecoder(len(word2index_input)+2, opt.embed_size, opt.hidden_size).to(device)
    decoder.load_state_dict(torch.load("saved_models/decoder.pth"))
    # ================== 加载模型 ===================

    encoder.eval()
    decoder.eval() 

    encoded, _ = encoder(x) # [bachsize,seq_len, hidden_size*2]

    decoder_in, s, w = decoder_initial(x.size(0))

    # out_list to store outputs
    out_list=[]
    for j in range(100):
        # 1st state
        if j==0:
            out, s, w = decoder(input_idx=decoder_in, encoded=encoded,
                            encoded_idx=input_out, prev_state=s,
                            weighted=w, order=j)
        # remaining states
        else:
            tmp_out, s, w = decoder(input_idx=decoder_in, encoded=encoded, # w是attention权重
                            encoded_idx=input_out, prev_state=s,
                            weighted=w, order=j)
            out = torch.cat([out,tmp_out],dim=1)  

        decoder_in = out[:,-1].max(1)[1]
        out_list.append(out[:,-1].max(1)[1].cpu().data.numpy()) 

    
    for k in range(1): #这里必须是1
        pre_sequence = []
        for j in range(25):
            if out_list[j][k] == 0:
                pre_sequence.append('<EOS>')
                break
            pre_sequence.append(index2word_output[str(out_list[j][k])])

        attention = w[:j + 1]

    return pre_sequence,attention


evaluateAndShowAttention("What is terrence ross' nationality",["Player","No.","Nationality","Position","Years in Toronto","School/Club Team"])

