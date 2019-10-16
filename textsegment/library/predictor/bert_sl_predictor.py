# -*- coding: UTF-8 -*-
"""
@author: liangjunjun
@contact: liangjunjun@pku.edu.cn
@version: v0.0.1
@file: bert_sl_predictor.py
@time: 2019/9/21 21:11
@desc:
"""
import logging
from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
logger = logging.getLogger(__name__)

@Predictor.register('bert_sl_predictor')
class BertSLPredictor(Predictor):
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        text = json_dict['text']
        return self._dataset_reader.text_to_instance(text)

    @overrides
    def predict_json(self, input: JsonDict) -> JsonDict:
        input_instance = self._json_to_instance(input)
        out_dict = self.predict_instance(input_instance)
        result = out_dict['pred']
        return result
