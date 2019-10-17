# -*- coding: UTF-8 -*-
"""
@author: liangjunjun
@contact: liangjunjun@pku.edu.cn
@version: v0.0.1
@file: extractor.py
@time: 2019/9/21 22:03
@desc: 封装后的分词的类
"""
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.common.util import import_submodules

import_submodules('textsegment.library')

class TextSegmenter(object):
    def __init__(self, model_path, output_path='./result.csv', cuda_device=0):
        archive = load_archive(model_path, cuda_device=cuda_device)
        self.predictor = Predictor.from_archive(archive, 'bert_sl_predictor')
        self.output_path = output_path

    def get_seg_index(self, text):
        return self.predictor.predict_json({'text':text})

    def seg_text(self, text):
        """
        对给定的字符串进行分词
        :param text:
        :return:
        """
        seg = []
        # 因为bert只能处理不超过500个字的文本，所以需要进行切割
        sen_len = 0
        seg_index = []
        for i in range(len(text)):
            sen_len += 1
            if sen_len >400 and text[i] in ['!','！','。','?',';','；','？']:
                _seg_index = self.get_seg_index(text[i-sen_len+1:i])
                seg_index.extend(_seg_index)
                sen_len = 0
        _seg_index = self.get_seg_index(text[i-sen_len+1:i+1])
        seg_index.extend(_seg_index)
        print(seg_index)
        return seg_index

    def seg_txt(self, file_path,output_path):
        """
        读入带分词的文本，将分词后的结果写入output_path
        :param file_path:
        :param output_path:
        :return:
        """
        with open(file_path) as fin, open(output_path, 'w') as fout:
            text_list = fin.readlines()
            for text in text_list:
                text = text.split('<sep>')[0]
                if not text:
                    continue
                print(text)
                segged_text = ''
                seg_index = self.seg_text(text)
                for i, label in enumerate(seg_index):
                    segged_text += text[i]
                    if label == 1:
                        segged_text += ' '
                fout.write(segged_text)
                fout.write('\n')

if __name__ == "__main__":
    segmenter = TextSegmenter(
        model_path='/home/liangjunjun/textsegment/model/bert_seg',
        cuda_device=2)
    # segmenter.seg_txt('../data/test.txt','./output.txt')
    text = '查体:体温:36.4°C,脉搏82次/分,呼吸20次/分,BP:120/85mmHg,神清语明,发育正常,自主体位,查体合作.全身皮肤黏膜无黄染,浅表淋巴结未触及肿大,头颅无畸形,颜面略有浮肿,口唇略发绀,颈软,气管居中,颈静脉无怒张,双肺呼吸音粗,双肺闻诊可疑干湿啰音,心律不齐,心率100次/分,第一心音强弱不等,心界向左右扩大,心音低钝,腹部膨隆,腹软,肝区无压痛,无移动性浊音,双下肢浮肿,生理反射存在,病理反射未引出'
    # text = '哈'
    seg_index = segmenter.seg_text(text)
    segged_text = ''
    for i,label in enumerate(seg_index):
        segged_text += text[i]
        if label == 1:
            segged_text += ' '
    print(segged_text)