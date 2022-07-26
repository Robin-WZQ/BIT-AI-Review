import json, time 
import numpy as np 
from tqdm import tqdm 

from torch.utils.data import Dataset, DataLoader
from pytorch_pretrained_bert import BertModel, BertTokenizer
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import os

train_l = []
val_f1_r = []

def record():
    f1 = open("record/val_f1.txt","w")
    f2 = open("record/loss.txt",'w')

    for i in range(len(train_l)):
        f1.write(str(val_f1_r[i])+"\n")
        f2.write(str(train_l[i])+"\n")

    f1.close()
    f2.close()

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
BERT_PATH = "chinese_roberta_wwm_ext_pytorch/"
maxlen = 256 

# 读取json文件，即输入text及对应spo格式
def load_data(filename):
    D = []
    with open(filename, 'r',encoding='utf-8') as f:
        all = json.load(f)
        for l in all:
            d = {'text': l['text'], 'spo_list': []}
            for relation in l['relations']:
                relationship = relation['type']
                for i in range(len(l['entities'])):
                    if relation['head_id'] == l['entities'][i]['id']:
                        head_entity = l['entities'][i]['value']
                        break
                for i in range(len(l['entities'])):
                    if relation['tail_id'] == l['entities'][i]['id']:
                        tail_entity = l['entities'][i]['value']
                        break
                d['spo_list'].append(
                    (head_entity, relationship, tail_entity)
                )

            D.append(d)
    return D 

# 加载数据集
train_data = load_data('DL/data_train.json')
valid_data = load_data('DL/data_dev.json')  

def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1

train_data_new = []   # 创建新的训练集，把结束位置超过250的文本去除，可见并没有去除多少
for data in tqdm(train_data):
    flag = 1
    for s, p, o in data['spo_list']:
        s_begin = search(s, data['text'])
        o_begin = search(o, data['text'])
        if s_begin == -1 or o_begin == -1 or s_begin + len(s) > 250 or o_begin + len(o) > 250:
            flag = 0
            break 
    if flag == 1:
        train_data_new.append(data)
print(len(train_data))
print(len(train_data_new))

# 读取schema
with open('DL/schema.json', encoding='utf-8') as f:
    id2predicate, predicate2id, n = {}, {}, 0
    predicate2type = {}
    for l in f:
        l = json.loads(l)
        predicate2type[l['predicate']] = (l['subject_type'], l['object_type'])
        key = l['predicate']
        if key not in predicate2id:
            id2predicate[n] = key
            predicate2id[key] = n
            n += 1
print(len(predicate2id))

import unicodedata
class OurTokenizer(BertTokenizer):
    def tokenize(self, text):
        R = []
        for c in text:
            if c in self.vocab:
                R.append(c)
            elif self._is_whitespace(c):
                R.append('[unused1]')
            else:
                R.append('[UNK]')
        return R 

    def _is_whitespace(self, char):
        if char == " " or char == "\t" or char == "\n" or char == "\r":
            return True
        cat = unicodedata.category(char)
        if cat == "Zs":
            return True
        return False

# 初始化分词器
tokenizer = OurTokenizer(vocab_file=BERT_PATH + "vocab.txt")

class TorchDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, i):
        t = self.data[i]
        x = tokenizer.tokenize(t['text'])
        x = ["[CLS]"] + x + ["[SEP]"]
        token_ids = tokenizer.convert_tokens_to_ids(x)
        seg_ids = [0] * len(token_ids) 
        assert len(token_ids) == len(t['text'])+2
        spoes = {}
        for s, p, o in t['spo_list']:
            s = tokenizer.tokenize(s)
            s = tokenizer.convert_tokens_to_ids(s)
            p = predicate2id[p.replace(" ","")] # 关系id
            o = tokenizer.tokenize(o)
            o = tokenizer.convert_tokens_to_ids(o)
            s_idx = search(s, token_ids) # subject起始位置
            o_idx = search(o, token_ids) # object起始位置
    
            if s_idx != -1 and o_idx != -1:
                s = (s_idx, s_idx + len(s) - 1)
                o = (o_idx, o_idx + len(o) - 1, p)  # 同时预测o和p
                if s not in spoes:
                    spoes[s] = [] # 可以一个subject多个object
                spoes[s].append(o)
        
        if spoes:
            sub_labels = np.zeros((len(token_ids), 2))
            for s in spoes:
                sub_labels[s[0], 0] = 1 
                sub_labels[s[1], 1] = 1
            # 随机选一个subject
            start, end = np.array(list(spoes.keys())).T
            start = np.random.choice(start)
            end = sorted(end[end >= start])[0]
            sub_ids = (start, end)
            obj_labels = np.zeros((len(token_ids), len(predicate2id), 2))
            for o in spoes.get(sub_ids, []):
                obj_labels[o[0], o[2], 0] = 1 
                obj_labels[o[1], o[2], 1] = 1 
        
        token_ids = self.sequence_padding(token_ids, maxlen=maxlen)
        seg_ids = self.sequence_padding(seg_ids, maxlen=maxlen)
        sub_labels = self.sequence_padding(sub_labels, maxlen=maxlen, padding=np.zeros(2))
        sub_ids = np.array(sub_ids)
        obj_labels = self.sequence_padding(obj_labels, maxlen=maxlen,
                                           padding=np.zeros((len(predicate2id), 2)))
        
        return (torch.LongTensor(token_ids), torch.LongTensor(seg_ids), torch.LongTensor(sub_ids),  
               torch.LongTensor(sub_labels), torch.LongTensor(obj_labels) )
 
    def __len__(self):
        data_len = len(self.data)
        return data_len
    
    def sequence_padding(self, x, maxlen, padding=0):
        output = np.concatenate([x, [padding]*(maxlen-len(x))]) if len(x)<maxlen else np.array(x[:maxlen])
        return output 

train_dataset = TorchDataset(train_data_new)
train_loader = DataLoader(dataset=train_dataset, batch_size=25, shuffle=True)
## debug
# for i, x in enumerate(train_loader):
#     print([_.shape for _ in x])
#     if i == 10:
#         break

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)   # [bs, maxlen, 1]
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class REModel(nn.Module):
    def __init__(self):
        super(REModel, self).__init__()
        self.bert = BertModel.from_pretrained(BERT_PATH)
        for param in self.bert.parameters():
            param.requires_grad = True
        
        self.linear = nn.Linear(768, 768)
        self.relu = nn.ReLU()
        self.sub_output = nn.Linear(768, 2)
        self.obj_output = nn.Linear(768, len(predicate2id)*2)
        
        self.sub_pos_emb = nn.Embedding(256, 768)   # subject位置embedding
        self.layernorm = BertLayerNorm(768, eps=1e-12)
    
    def forward(self, token_ids, seg_ids, sub_ids=None):
        out, _ = self.bert(token_ids, token_type_ids=seg_ids, 
                           output_all_encoded_layers=False)   # [batch_size, maxlen, size]
        sub_preds = self.sub_output(out)   # [batch_size, maxlen, 2]
        sub_preds = torch.sigmoid(sub_preds) 
        # sub_preds = sub_preds ** 2 

        if sub_ids is None:
            return sub_preds

        # 融入subject特征信息
        sub_pos_start = self.sub_pos_emb(sub_ids[:, :1])
        sub_pos_end = self.sub_pos_emb(sub_ids[:, 1:])   # [batch_size, 1, size]

        sub_id1 = sub_ids[:, :1].unsqueeze(-1).repeat(1, 1, out.shape[-1])     # subject开始的位置id
        sub_id2 = sub_ids[:, 1:].unsqueeze(-1).repeat(1, 1, out.shape[-1])     # [batch_size, 1, size]
        sub_start = torch.gather(out, 1, sub_id1)
        sub_end = torch.gather(out, 1, sub_id2)   # [batch_size, 1, size]
        
        sub_start = sub_pos_start + sub_start
        sub_end = sub_pos_end + sub_end
        out1 = out + sub_start + sub_end
        out1 = self.layernorm(out1)
        out1 = F.dropout(out1, p=0.5, training=self.training)
        
        output = self.relu(self.linear(out1))  
        output = F.dropout(output, p=0.4, training=self.training)
        output = self.obj_output(output)  # [batch_size, maxlen, 2*plen]
        output = torch.sigmoid(output)
        # output = output ** 2
        
        obj_preds = output.view(-1, output.shape[1], len(predicate2id), 2)
        return sub_preds, obj_preds

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = REModel().to(DEVICE)
print(DEVICE)
optimizer = torch.optim.AdamW(net.parameters(), lr=1e-5)

def extract_spoes(text, model, device):
    """抽取三元组"""
    if len(text) > 254:
        text = text[:254]
    tokens = tokenizer.tokenize(text)
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    assert len(token_ids) == len(text) + 2 
    seg_ids = [0] * len(token_ids) 
    
    sub_preds = model(torch.LongTensor([token_ids]).to(device), 
                      torch.LongTensor([seg_ids]).to(device))
    sub_preds = sub_preds.detach().cpu().numpy()  # [1, maxlen, 2]
    # print(sub_preds[0,])
    start = np.where(sub_preds[0, :, 0] > 0.2)[0]
    end = np.where(sub_preds[0, :, 1] > 0.2)[0]
    # print(start, end)
    tmp_print = []
    subjects = []
    for i in start: 
        j = end[end>=i]
        if len(j) > 0:
            j = j[0]
            subjects.append((i, j))
            tmp_print.append(text[i-1: j])

    if subjects:
        spoes = []
        token_ids = np.repeat([token_ids], len(subjects), 0)   # [len_subjects, seqlen]
        seg_ids = np.repeat([seg_ids], len(subjects), 0)
        subjects = np.array(subjects)   # [len_subjects, 2]
        # 传入subject 抽取object和predicate
        _, object_preds = model(torch.LongTensor(token_ids).to(device), 
                            torch.LongTensor(seg_ids).to(device), 
                            torch.LongTensor(subjects).to(device))
        object_preds = object_preds.detach().cpu().numpy()
#         print(object_preds.shape)
        for sub, obj_pred in zip(subjects, object_preds):
            # obj_pred [maxlen, 55, 2]
            start = np.where(obj_pred[:, :, 0] > 0.2)
            end = np.where(obj_pred[:, :, 1] > 0.2)
            for _start, predicate1 in zip(*start):
                for _end, predicate2 in zip(*end):
                    if _start <= _end and predicate1 == predicate2:
                        spoes.append(
                            ((sub[0]-1, sub[1]-1), predicate1, (_start-1, _end-1))
                        )
                        break
        return [(text[s[0]:s[1]+1], id2predicate[p], text[o[0]:o[1]+1]) 
                for s, p, o in spoes]
    else:
        return []

def evaluate(data, model, device):
    """评估函数，计算f1、precision、recall
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    f = open('DL/dev_pred.json', 'w', encoding='utf-8')
    pbar = tqdm()
    for d in data:
        R = extract_spoes(d['text'], model, device)
        T = d['spo_list']
#         print(R, T)
        R = set(R)
        T = set(T)
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        pbar.update()
        pbar.set_description(
            'f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall)
        )
        s = json.dumps({
            'text': d['text'],
            'spo_list': list(T),
            'spo_list_pred': list(R),
            'new': list(R - T),
            'lack': list(T - R),
        }, ensure_ascii=False, indent=4)
        f.write(s + '\n')
    pbar.close()
    f.close()

    return f1, precision, recall

def train(model, train_loader, optimizer, epoches, device):
    f1_max = 0.5556
    for _ in range(epoches):
        print('epoch: ', _ + 1)
        start = time.time()
        train_loss_sum = 0.0
        for batch_idx, x in tqdm(enumerate(train_loader)):
            token_ids, seg_ids, sub_ids = x[0].to(device), x[1].to(device), x[2].to(device)
            mask = (token_ids > 0).float()
            mask = mask.to(device)   # zero-mask
            sub_labels, obj_labels = x[3].float().to(device), x[4].float().to(device)
            sub_preds, obj_preds = model(token_ids, seg_ids, sub_ids)
            # (batch_size, maxlen, 2),  (batch_size, maxlen, 44, 2)

            # 计算loss
            loss_sub = F.binary_cross_entropy(sub_preds, sub_labels, reduction='none')  #[bs, ml, 2]
            loss_sub = torch.mean(loss_sub, 2)  # (batch_size, maxlen)
            loss_sub = torch.sum(loss_sub * mask) / torch.sum(mask)
            loss_obj = F.binary_cross_entropy(obj_preds, obj_labels, reduction='none')  # [bs, ml, 44, 2]
            loss_obj = torch.sum(torch.mean(loss_obj, 3), 2)   # (bs, maxlen)
            loss_obj = torch.sum(loss_obj * mask) / torch.sum(mask)
            loss = loss_sub + loss_obj
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.cpu().item()
            if (batch_idx + 1) % 300 == 0:
                print('loss: ', train_loss_sum / (batch_idx+1), 'time: ', time.time() - start) 
                train_l.append((_ + 1,train_loss_sum / (batch_idx+1)))
        
        
        with torch.no_grad():
            val_f1, pre, rec = evaluate(valid_data, net, DEVICE)
            print("f1, pre, rec: ", val_f1, pre, rec)
            if val_f1>f1_max:
                torch.save(net.state_dict(), "DL/model.pth")
                f1_max = val_f1
        
        val_f1_r.append((_ + 1,val_f1))

# 如果要继续训练就不注释这行
# net.load_state_dict(torch.load("DL/model.pth"))
train(net, train_loader, optimizer, 30, DEVICE)

def combine_spoes(spoes):
    """ 
    """
    new_spoes = {}
    for s, p, o in spoes:
        p1 = p
        if (s, p1) in new_spoes:
            new_spoes[(s, p1)] = o
        else:
            new_spoes[(s, p1)] = o

    return [(k[0], k[1], v) for k, v in new_spoes.items()]

def predict_to_file(in_file, out_file):
    """预测结果到文件，方便提交
    """
    fw = open(out_file, 'w', encoding='utf-8')
    with open(in_file, encoding='utf-8') as fr:
        all = json.load(fr)
        for l in all:
            spoes = combine_spoes(extract_spoes(l['text'], net, DEVICE))
            spoes = [{
                'type': spo[1],
                'head': spo[0],
                'tail': spo[2],
            }
                     for spo in spoes]
            l['relations'] = spoes
            s = json.dumps(l, ensure_ascii=False)
            fw.write(s + ',\n')
    fw.close()

predict_to_file('DL/data_test_process.json', 'DL/RE_pred.json')
record()