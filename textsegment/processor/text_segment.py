# -*- coding: UTF-8 -*-
"""
@author: liangjunjun
@contact: liangjunjun@pku.edu.cn
@version: v0.0.1
@file: extractor.py
@time: 2019/9/21 22:03
@desc: 
"""
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor


class TextSegmenter(object):
    def __init__(self, model_path, output_path='./result.csv', cuda_device=0):
        archive = load_archive(model_path, cuda_device=cuda_device)
        self.predictor = Predictor.from_archive(archive, 'bert_sl_predictor')
        self.output_path = output_path

    def get_seg_index(self, text):
        return self.predictor.predict_json({'text':text})

    def seg_text(self, text):
        seg = []
        # 因为bert只能处理不超过500个字的文本，所以需要进行切割
        sen_len = 0
        seg_index = []
        for i in range(len(text)):
            sen_len += 1
            if sen_len >400 and text[i] in ['!','！','。','?',';','；','？']:
                _seg_index = self.get_seg_index(text[i:i+sen_len])
                _seg_index = [index + i for index in _seg_index]
                seg_index.extend(seg_index)
        _seg_index = self.get_seg_index(text[i:i + sen_len])
        _seg_index = [index + i for index in _seg_index]
        seg_index.extend(seg_index)
        return seg_index

    def seg_txt(self, file_path,output_path):
        with open(file_path) as fin, open(output_path, 'w') as fout:
            text_list = fin.readlines()
            for text in text_list:
                segged_text = ''
                seg_index = self.seg_text()
                for i in len(text):
                    if i in seg_index:
                        segged_text += ' '
                    segged_text += text[i]
                fout.write(segged_text)
                fout.write('\n')

if __name__ == "__main__":
    segmenter = TextSegmenter(
        model_path='/home/liangjunjun/textsegment/model/model.tar.gz',
        cuda_device=2)
    segmenter.seg_txt('../data/dev.txt','./output.txt')
