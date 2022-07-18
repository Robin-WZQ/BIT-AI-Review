import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
from model.Decoder import AttnDecoderRNN, DecoderRNN
from model.Encoder import EncoderRNN
from opts import parse_opts
import json
import nltk

# ================== 读入词表 ===================
with open('index2word_output.json', 'r') as fp:
    index2word = json.load(fp) # 所有词的词表，不区分输出与输入
with open('word2index_input.json', 'r') as fp:
    word2index = json.load(fp)
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
        input.append(word2index[token.lower()])
    input.append(2)
    for col in table:
        for token in nltk.word_tokenize(col):
            try:
                input.append(word2index[token])
            except:
                input.append(len(word2index)+1)
            input.append(2)
    input = torch.LongTensor(input)

    # ================== 加载模型 ===================
    # 编码器
    encoder = EncoderRNN(len(word2index)+2, opt.hidden_size,opt.device).to(opt.device)
    encoder.load_state_dict(torch.load("saved_models/encoder_seq2seq_9.pt"))    
    # 解码器
    decoder = AttnDecoderRNN(opt.hidden_size, len(word2index)+2, dropout_p=0.1,max_length=opt.max_length,device=opt.device).to(opt.device)
    decoder.load_state_dict(torch.load("saved_models/decoder_seq2seq_9.pt"))
    # ================== 加载模型 ===================

    encoder.eval()
    decoder.eval() 
    with torch.no_grad():
        current_input = input.to(device)
        encoder_hidden = encoder.initHidden()
        decoder_hidden = encoder_hidden
        input_length = current_input.size(0)

        encoder_outputs = torch.zeros(opt.max_length, encoder.hidden_size, device=device)
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                current_input[ei], encoder_hidden)

            encoder_outputs[ei] = encoder_output[0, 0]
        decoder_input = torch.tensor([[opt.SOS_token]], device=device)
        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(opt.max_length, opt.max_length)

        for di in range(9):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == opt.EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(index2word[str(topi.item())])

            decoder_input = topi.squeeze().detach()

        sql_output = decoded_words
        attention = decoder_attentions[:di + 1]

    return sql_output,attention

evaluateAndShowAttention("What is terrence ross' nationality",["Player","No.","Nationality","Position","Years in Toronto","School/Club Team"])
