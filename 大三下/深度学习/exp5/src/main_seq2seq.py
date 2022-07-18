import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset.dataloader import WikiDataset
from model.Decoder import AttnDecoderRNN, DecoderRNN
from model.Encoder import EncoderRNN
from opts import parse_opts

train_l = []
val_acc = []

def train(epoch,encoder,decoder,encoder_optimizer,decoder_optimizer,iterator,opt,device,scheduler1,scheduler2):
    '''
    训练函数
    '''
    encoder.train()
    decoder.train()

    train_l_sum = 0.0
    m=0
    for i, batch in enumerate(iterator):
        input = batch[0].to(device)
        output = batch[1].to(device)
        for j in range(input.shape[0]):
            loss_all = 0
            encoder_hidden = encoder.initHidden()
            decoder_hidden = encoder_hidden
            current_input = input[j]
            current_output = output[j].unsqueeze(1)
            input_length = current_input.size(0)
            target_length = current_output.size(0)
            if input_length > opt.max_length:
                input_length = opt.max_length
            encoder_outputs = torch.zeros(opt.max_length, encoder.hidden_size, device=device)
            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(current_input[ei], encoder_hidden)
                encoder_outputs[ei] = encoder_output[0, 0]
            decoder_input = torch.tensor([[opt.SOS_token]], device=device)

            decoder_hidden = encoder_hidden

            use_teacher_forcing = True 

            if use_teacher_forcing:
                # Teacher forcing: Feed the target as the next input
                for di in range(target_length):
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, encoder_outputs)
                    loss_all += criterion(decoder_output, current_output[di])
                    decoder_input = current_output[di]  # Teacher forcing

            else:
                # Without teacher forcing: use its own predictions as the next input
                for di in range(target_length):
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, encoder_outputs)
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze().detach()  # detach from history as input

                    loss_all += criterion(decoder_output, current_output[di])
                    if decoder_input.item() == opt.EOS_token:
                        break
                
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            loss_all.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

            m+=1
            train_l_sum+=loss_all.item() / target_length                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
            train_l.append((epoch*len(iterator)+m,train_l_sum / m))

        if m%1000==0: # 如果进行1000次更新，输出结果
            print('epoch %d [%d/%d], loss %.4f'% (epoch + 1, m,
                    len(iterator),train_l_sum / m))
    scheduler1.step()
    scheduler2.step()

def evaluate(epoch,encoder,decoder,iterator,opt,device):
    '''
    验证函数
    '''
    encoder.eval()
    decoder.eval()
    acc = 0.0
    m=0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            input = batch[0].to(device)
            output = batch[1].to(device)

            for j in range(input.shape[0]):
                encoder_hidden = encoder.initHidden()
                decoder_hidden = encoder_hidden
                current_input = input[j]
                current_output = output[j].unsqueeze(1)
                input_length = current_input.size(0)
                target_length = current_output.size(0)
                if input_length > opt.max_length:
                    input_length = opt.max_length
                encoder_outputs = torch.zeros(opt.max_length, encoder.hidden_size, device=device)
                for ei in range(input_length):
                    encoder_output, encoder_hidden = encoder(
                        current_input[ei], encoder_hidden)

                    encoder_outputs[ei] = encoder_output[0, 0]
                decoder_input = torch.tensor([[opt.SOS_token]], device=device)

                decoder_hidden = encoder_hidden
                decoded_words = []

                for di in range(target_length):
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, encoder_outputs)
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze().detach()  

                    if topi.item() == opt.EOS_token:
                        # decoded_words.append('<EOS>')
                        break
                    else:
                        decoded_words.append(str(topi.item()))
                
                decoded_words = ' '.join(decoded_words)
                gt = []
                for token in current_output:
                    if token.item() == opt.EOS_token:
                        break
                    else:
                        gt.append(str(token.item()))
                correct = gt == decoded_words
                acc += correct

            m+=1

            val_acc.append((epoch*len(iterator)+m*input.shape[0],acc / (m * input.shape[0])))
        print("预测正确的比率为：",acc / (m * input.shape[0]))

def test(iterator,index2word_input,index2word_output,opt,device):
    '''
    测试函数
    '''
    # ================== 加载模型 ===================
    # 编码器
    encoder = EncoderRNN(len(index2word_input)+2, opt.hidden_size,opt.device).to(device)
    encoder.load_state_dict(torch.load("saved_models/encoder_seq2seq_7.pt"))    
    # 解码器
    decoder = AttnDecoderRNN(opt.hidden_size, len(index2word_input)+2, dropout_p=0.1,max_length=opt.max_length,device=opt.device).to(device)
    decoder.load_state_dict(torch.load("saved_models/decoder_seq2seq_7.pt"))
    # ================== 加载模型 ===================

    f = open("predict_labels_1120190892.txt","w",encoding='utf-8')
    encoder.eval()
    decoder.eval() 
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            input = batch[0].to(device)
            output = batch[1].to(device)

            for j in range(input.shape[0]):
                encoder_hidden = encoder.initHidden()
                decoder_hidden = encoder_hidden
                current_input = input[j]
                current_output = output[j].unsqueeze(1)
                input_length = current_input.size(0)
                target_length = current_output.size(0)
                encoder_outputs = torch.zeros(opt.max_length, encoder.hidden_size, device=device)
                if input_length > opt.max_length:
                    input_length = opt.max_length
                for ei in range(input_length):
                    encoder_output, encoder_hidden = encoder(
                        current_input[ei], encoder_hidden)

                    encoder_outputs[ei] = encoder_output[0, 0]
                decoder_input = torch.tensor([[opt.SOS_token]], device=device)
                decoder_hidden = encoder_hidden

                decoded_words = []
                decoder_attentions = torch.zeros(opt.max_length, opt.max_length)

                for di in range(target_length):
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, encoder_outputs)
                    decoder_attentions[di] = decoder_attention.data
                    topv, topi = decoder_output.data.topk(1)
                    if topi.item() == opt.EOS_token:
                        # decoded_words.append('<EOS>')
                        break
                    else:
                        decoded_words.append(index2word_output[str(topi.item())])

                    decoder_input = topi.squeeze().detach()

                sql_output = decoded_words
                attention = decoder_attentions[:di + 1]
                f.write(' '.join(sql_output)+'\n')
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
    # train_set = WikiDataset(dataset='train')
    # dev_set = WikiDataset(dataset='dev')
    test_set = WikiDataset(dataset='test')    

    # train_loader = DataLoader(train_set,batch_size=opt.batch_size,shuffle=True)
    print("1/3")
    # dev_loader = DataLoader(dev_set,batch_size=opt.batch_size,shuffle=True)
    print("2/3")
    test_loader = DataLoader(test_set,batch_size=opt.batch_size,shuffle=False)
    # ================== 读取数据 ===================

    # ================== 读入词表 ===================
    with open('index2word_input.json', 'r') as fp:
        index2word_input = json.load(fp) # 输入
    with open('index2word_output.json', 'r') as fp:
        index2word_output = json.load(fp) # 输出
    # ================== 读入词表 ===================

    print("数据读入完毕")

    # ================== 设置模型及参数 ===================
    # 编码器
    encoder = EncoderRNN(len(index2word_input)+2, opt.hidden_size,opt.device).to(device)
    # 解码器
    attn_decoder = AttnDecoderRNN(opt.hidden_size, len(index2word_input)+2, dropout_p=0.1,max_length=opt.max_length,device=opt.device).to(device)
    if opt.pre_trained == True:
        encoder.load_state_dict(torch.load(opt.encoder_path))    
        attn_decoder.load_state_dict(torch.load(opt.decoder_path))
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器-1
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=opt.learning_rate)
    # 优化器-2
    decoder_optimizer = optim.Adam(attn_decoder.parameters(), lr=opt.learning_rate)
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer=encoder_optimizer,
                   milestones=[int(opt.epoch_num * 0.56), int(opt.epoch_num * 0.78)],
                   gamma=0.1, last_epoch=-1)
    scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer=decoder_optimizer,
                   milestones=[int(opt.epoch_num * 0.56), int(opt.epoch_num * 0.78)],
                   gamma=0.1, last_epoch=-1)
    # ================== 设置模型及参数 ===================

    print("开始训练！")

    # ================== 训练集用于训练，验证集用于评估 ==================
    # for epoch in range(opt.epoch_num):
    #     train(epoch,encoder,attn_decoder,encoder_optimizer,decoder_optimizer,train_loader,opt,device,scheduler1,scheduler2)
    #     if epoch % 2 == 0:
    #         evaluate(epoch,encoder,attn_decoder,dev_loader,opt,device)
    #         # 保存模型参数
    #         torch.save(encoder.state_dict(),"saved_models/encoder_seq2seq_{}.pt".format(epoch+1))
    #         torch.save(attn_decoder.state_dict(),"saved_models/decoder_seq2seq_{}.pt".format(epoch+1))
    # ================== 训练集用于训练，验证集用于评估 ==================

    # 保存训练过程
    # record()

    # 将测试集的预测结果按要求保存在txt文件中
    test(test_loader,index2word_input,index2word_output,opt,device)
