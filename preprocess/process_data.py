# -*- coding: utf-8 -*-
"""
把分词数据处理为需要的格式：

例如
今天 天气 很好
今天天气很好<sep>1\t3\n
"""

import numpy as np
import os
import re

def process_string(string):
    '''把string对应处理为需要格式
    
    Args:
        string: 一个句子字符串
    '''
    lst = string.split()  #eg [今天，天气，很好]
    len_lst = [len(item) for item in lst]        
    accumulated_len_lst = []
    for j,l in enumerate(len_lst):
        if j==0:
            accumulated_len_lst.append(l)
        else:
            accumulated_len_lst.append(accumulated_len_lst[-1]+l)
    sep_id_lst = np.array( np.array(accumulated_len_lst[:-1])-1, dtype='str')
    
    final_string = ''.join(lst)
    final_sep_id_string = '\t'.join(sep_id_lst)
    return final_string+'<sep>'+final_sep_id_string



def read_path_file(file_path='测试集\输出'):
    '''读入数据，输出a list of string sentence
    '''
    string_lst = []
    
    for f_name in os.listdir(file_path):
        print('读入文件', f_name)
        
        f = open(os.path.join(file_path,f_name),'r',encoding='utf-8')
        for line in f:
            line = line.strip()
            
            if line.startswith('<')  and line.endswith('>'):  #数据中  < 病历 > 这类形式的就不作为需要切分的例子了
                continue
            
            # 把文本中“1 、 给予 抗凝 ， 抗血小板 ， 扩冠 ， 调脂 ， 改善心肌供血 ， 对症治疗 。”这种开头数字给去掉
            new_line = re.sub('^\d\s*[,，、.]*','',line).strip()
            if new_line=='\x1a': #检查发现有个这种不合理字符，不读入
                continue
            
            string_lst.append(new_line)
    return string_lst



if __name__=='__main__':
#    string_lst = read_path_file(file_path='测试集\输出')
#    out_lst = [process_string(string) for string in string_lst]
#   
#    with open('test.txt','w',encoding='utf-8') as f:
#        for line in out_lst:
#            f.write(line+'\n')
            
    string_lst = read_path_file(file_path='训练集\输出')
    out_lst = [process_string(string) for string in string_lst]
    from sklearn.model_selection import train_test_split
    train, dev, _, _ = train_test_split(out_lst, out_lst, test_size=0.2)
    
   
    with open('train.txt','w',encoding='utf-8') as f_train:
        for line in train:
            f_train.write(line+'\n')         
    
    with open('dev.txt','w',encoding='utf-8') as f_dev:
        for line in dev:
            f_dev.write(line+'\n')         