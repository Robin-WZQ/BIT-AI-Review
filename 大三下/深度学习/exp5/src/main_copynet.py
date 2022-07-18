import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset.dataloader2 import WikiDataset
from model.Decoder import AttnDecoderRNN, DecoderRNN
from model.Encoder import EncoderRNN
from model.copynet_dbg import CopyEncoder, CopyDecoder
from opts import parse_opts
import random
import numpy as np
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
import warnings
warnings.filterwarnings('ignore')

train_l = []
val_acc = []

def decoder_initial(batch_size):
    decoder_in = torch.LongTensor(np.ones(batch_size,dtype=int))*2
    s = None
    w = None
    if torch.cuda.is_available():
        decoder_in = decoder_in.cuda()
    decoder_in = Variable(decoder_in)
    return decoder_in, s, w

def numpy_to_var(x,is_int=True):
    if is_int:
        x = torch.LongTensor(x) # 一定记得转成LongTensor
    else:
        x = torch.Tensor(x)
    return Variable(x)

def order(batch):
    input_out = batch[0].detach().numpy()
    output_out = batch[1].detach().numpy()
    in_len = batch[2].detach().numpy()
    out_len = batch[3].detach().numpy()
    out_rev = out_len.argsort()[::-1]  # 按长度倒序以进行pack_padding_sequence
    return input_out[out_rev], output_out[out_rev], in_len[out_rev], out_len[out_rev]

def train(epoch,encoder,decoder,encoder_optimizer,decoder_optimizer,iterator,opt,device):
    '''
    训练函数
    '''
    encoder.train()
    decoder.train()

    train_l_sum = 0.0
    m=0
    for i, batch in enumerate(iterator):
        # 分别为：输入、输出、输入序列长度、输出序列长度
        input_out,output_out,in_len,out_len = order(batch)

        # 转成Tensor
        x = numpy_to_var(input_out).to(device) # torch.size([b,seq])
        y = numpy_to_var(output_out).to(device)

        encoded, _ = encoder(x) # [bachsize,seq_len, hidden_size*2]

        decoder_in, s, w = decoder_initial(x.size(0))

        # out_list 用来存储预测结果
        out_list=[]
        for j in range(y.size(1)):
            # 初始状态
            if j==0:
                out, s, w = decoder(input_idx=decoder_in, encoded=encoded,
                                encoded_idx=input_out, prev_state=s,
                                weighted=w, order=j)
            # 剩余状态
            else:
                tmp_out, s, w = decoder(input_idx=decoder_in, encoded=encoded, # w是attention权重
                                encoded_idx=input_out, prev_state=s,
                                weighted=w, order=j)
                out = torch.cat([out,tmp_out],dim=1)  

            if epoch % 2 ==1:
                decoder_in = out[:,-1].max(1)[1].squeeze() # 使用模型输出进行训练
            else:
                decoder_in = y[:,j] # 使用GT进行训练
            out_list.append(out[:,-1].max(1)[1].squeeze().cpu().data.numpy()) 

        target = pack_padded_sequence(y,out_len.tolist(), batch_first=True)[0]
        pad_out = pack_padded_sequence(out,out_len.tolist(), batch_first=True)[0]
        # 计算Loss前的必要操作
        pad_out = torch.log(pad_out)
        loss = criterion(pad_out, target)
        loss /=  opt.batch_size  

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        loss.backward()
 
        encoder_optimizer.step()
        decoder_optimizer.step()

        m+=1

        train_l_sum += loss.item()
        train_l.append((epoch*len(iterator)+m*opt.batch_size,train_l_sum / m))

        if m%10==0: # 如果进行10次更新，输出结果
            print('epoch %d [%d/%d], loss %.4f'% (epoch + 1, m,
                    len(iterator),train_l_sum / m))

def evaluate(encoder,decoder,iterator,opt,device):
    '''
    验证函数
    '''
    encoder.eval()
    decoder.eval()
    m=0
    acc = 0
    for i, batch in enumerate(iterator):

        input_out,output_out,in_len,out_len = order(batch)

        x = numpy_to_var(input_out).to(device) # torch.size([b,seq])
        y = numpy_to_var(output_out).to(device)

        encoded, _ = encoder(x) # [bachsize,seq_len, hidden_size*2]

        decoder_in, s, w = decoder_initial(x.size(0))

        # out_list to store outputs
        out_list=[]
        for j in range(y.size(1)):
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

            decoder_in = y[:,j]
            out_list.append(out[:,-1].max(1)[1].squeeze().cpu().data.numpy()) 
        
        gt = y.tolist()
        for k in range(opt.batch_size):
            pre_sequence = []
            gt_sequence = []
            for j in range(out_len[k]):
                pre_sequence.append(out_list[j][k])
                gt_sequence.append(gt[k][j])
            if pre_sequence == gt_sequence:
                acc += 1
        
        m += opt.batch_size

        if m >= 100 :
            break

    print("验证集预测正确的比率为：",acc / m )

def test(iterator,index2word_input,index2word_output,opt,device):
    '''
    测试函数
    '''
    # ================== 加载模型 ===================
    # 编码器
    encoder = CopyEncoder(len(index2word_input)+2, opt.embed_size, opt.hidden_size).to(device)
    encoder.load_state_dict(torch.load("saved_models/encoder_cpn_3.pth"))    
    # 解码器
    decoder = CopyDecoder(len(index2word_input)+2, opt.embed_size, opt.hidden_size).to(device)
    decoder.load_state_dict(torch.load("saved_models/decoder_cpn_3.pth"))
    # ================== 加载模型 ===================

    f = open("predict_labels_1120190892.txt","w",encoding='utf-8')
    encoder.eval()
    decoder.eval()
    for i, batch in enumerate(iterator):

        input_out,output_out,in_len,out_len = order(batch)

        x = numpy_to_var(input_out).to(device) # torch.size([b,seq])
        y = numpy_to_var(output_out).to(device)

        encoded, _ = encoder(x) # [bachsize,seq_len, hidden_size*2]

        decoder_in, s, w = decoder_initial(x.size(0))

        # out_list to store outputs
        out_list=[]
        for j in range(y.size(1)):
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

            decoder_in = y[:,j] 
            out_list.append(out[:,-1].max(1)[1].cpu().data.numpy()) 
        
        for k in range(1): #这里必须是1
            pre_sequence = []
            for j in range(out_len[k]):
                pre_sequence.append(index2word_output[str(out_list[j][k])])

            f.write(' '.join(pre_sequence)+'\n')
    f.close()

def record():
    '''
    保存迭代过程
    '''
    f1 = open("record/train_loss.txt","w")
    f2 = open("record/val_acc.txt",'w')

    for i in range(len(train_l)):
        f1.write(str(train_l[i])+"\n")
    for i in range(len(val_acc)): 
        f2.write(str(val_acc[i])+"\n")

    f1.close()
    f2.close()

if __name__ == '__main__':
    # 配置文件
    opt = parse_opts()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    opt.device = device

    # ================== 读取数据 ===================
    train_set = WikiDataset(dataset='train',opt=opt)
    dev_set = WikiDataset(dataset='dev',opt=opt)
    test_set = WikiDataset(dataset='test',opt=opt)    

    train_loader = DataLoader(train_set,batch_size=1,shuffle=False)
    # print("1/3")
    dev_loader = DataLoader(dev_set,batch_size=opt.batch_size,shuffle=True)
    print("2/3")
    test_loader = DataLoader(test_set,batch_size=1,shuffle=False)
    # ================== 读取数据 ===================

    # ================== 读入词表 ===================
    with open('index2word_input.json', 'r') as fp:
        index2word_input = json.load(fp) # 输入
    with open('index2word_output.json', 'r') as fp:
        index2word_output = json.load(fp) # 输出
    # ================== 读入词表 ===================

    print("数据读入完毕")

    # ================== 设置模型及参数 ===================
    encoder = CopyEncoder(len(index2word_input)+2, opt.embed_size, opt.hidden_size).to(device)
    decoder = CopyDecoder(len(index2word_input)+2, opt.embed_size, opt.hidden_size).to(device)
    # 损失函数
    criterion = nn.NLLLoss()
    # 优化器-1
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=opt.learning_rate)
    # 优化器-2
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=opt.learning_rate)
    # ================== 设置模型及参数 ===================

    print("开始训练！")
    # evaluate(encoder,decoder,dev_loader,opt,device)

    # ================== 训练集用于训练，验证集用于评估 ==================
    for epoch in range(opt.epoch_num):
        train(epoch,encoder,decoder,encoder_optimizer,decoder_optimizer,train_loader,opt,device)
        evaluate(encoder,decoder,dev_loader,opt,device)
        if epoch % 2 == 0:
            # 保存模型参数
            torch.save(encoder.state_dict(),"saved_models/encoder_cpn_{}.pth".format(epoch+1))
            torch.save(decoder.state_dict(),"saved_models/decoder_cpn_{}.pth".format(epoch+1))
    # ================== 训练集用于训练，验证集用于评估 ==================

    # 保存训练过程
    record()

    # 将测试集的预测结果按要求保存在txt文件中
    test(test_loader,index2word_input,index2word_output,opt,device)