'''
@ file_name : post_process
@ file_function : 将输出SQL处理成evaluate.py需要的输入格式
@ author : 王中琦
@ date : 2022/5/22 
'''
import encodings
import jsonlines
import pandas as pd

tables = pd.read_json("data/tokenized_test.tables.jsonl", lines=True)
queries = pd.read_json("data/tokenized_test.jsonl", lines=True)
with open("version1.0.txt",encoding='utf-8') as f:
    lines = f.readlines()
out = {}

agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
cond_ops = ['EQL', 'GT', 'LT']

with jsonlines.open("./pred.test.jsonl", 'w') as f:
    for index,row in queries.iterrows():
        error = 0
        current = {"agg":0,"sel":0,"conds":[]}
        table_id = row['table_id']
        header = tables.loc[tables['id'] == table_id]
        headers = header['header'].tolist()[0]    
        my_answer = lines[index].replace("\n","")
        position_1 = my_answer.find("(")
        position_2 = my_answer.find(")")
        position_3 = my_answer.find("FROM")
        # ======================== 算法1 ========================
        if position_1 != -1 and position_2 != -1 and position_2 < position_3 and position_1 < position_2:
            current_op = my_answer[6:position_1]
            k=0
            for op in agg_ops:
                if op in current_op:
                    current['agg'] = k
                    flag =1 
                    break
                k+=1
            flag = 0
            current_head = my_answer[position_1:position_2]
            for i in range(len(headers)):
                if headers[i].lower() == current_head.lower():
                    current['sel'] = i
                    flag =1 
                    break
            if flag == 0:
                error = 1
        else:
            current['agg'] = 0
            current_head = my_answer[7:position_3-1]
            flag = 0
            for i in range(len(headers)):
                if headers[i].lower() == current_head.lower():
                    current['sel'] = i
                    flag =1 
                    break
            if flag == 0:
                error = 1
        # ======================== 算法1 ========================

        # ======================== 算法2 ========================
        conds = []
        position_4 = my_answer.find("WHERE")
        position_5 = my_answer.find("AND")
        if position_4+5 != len(my_answer):
            line_split = my_answer.split(" ")
            startpos = line_split.index('WHERE') + 1
            and_indices = [i for i, x in enumerate(line_split) if x == "AND"]
            num = my_answer.count("AND")
            and_indices.append(len(line_split))
            for endpos in and_indices:
                if endpos < startpos:
                    continue
                current_str = " ".join(my_answer[startpos:endpos])
                for j in range(len(cond_ops)):
                    position_cond = current_str.find(cond_ops[j])
                    if position_cond != -1:
                        conds.append(j)
                        break
                current_str_1 = current_str[:position_cond-1]
                if position_cond != 0:
                    current_str_2 = current_str[position_cond+4:]
                else:
                    current_str_2 = current_str[position_cond+5:]
                flag=0
                for j in range(len(headers)):
                    if headers[j].lower() == current_str_1.lower():
                        conds.insert(0,j)
                        flag=1
                        break
                if flag==0:
                    error = 1
                conds.append(current_str_2)
                if len(conds) < 3:
                    conds.insert(0,0)
                if len(conds) < 3:
                    conds.insert(0,0)
                current['conds']= [conds]
                startpos = endpos + 1
                # current_str = my_answer[position_5+4:]
        else:
            current['conds'] = []

        # if error == 1:
            # f.write({"query": "null","error":"Not Found"})
        # else:
        f.write({"query":current,"error":""})            
        # if index == 30:
        #     break
