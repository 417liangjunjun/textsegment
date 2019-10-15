# -*- coding: UTF-8 -*-
"""
@author: liangjunjun
@contact: liangjunjun@pku.edu.cn
@version: v0.0.1
@file: bert_sl_tagger.py
@time: 2019/9/8 13:57
@desc: 
"""
import logging
from typing import Dict, List, Optional, Any
import torch
from torch.nn.modules.linear import Linear
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics.f1_measure import F1Measure
from overrides import overrides
import allennlp.nn.util as util
from textsegment.library.training.metrics.multi_f1 import MultiF1Measure
logger = logging.getLogger(__name__)


@Model.register("bert_sl_tagger")
class BertSlTaggerModel(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 ) -> None:
        super(BertSlTaggerModel, self).__init__(vocab, regularizer)
        self._text_field_embedder = text_field_embedder
        self.num_class = 2
        self._labels_predictor = Linear(self._text_field_embedder.get_output_dim(), self.num_class)
        self._text_field_embedder = text_field_embedder
        self.loss = torch.nn.CrossEntropyLoss()
        self.metrics = {
            'f1': MultiF1Measure(1)
        }
        initializer(self)

    def forward(self,
                text: Dict[str, torch.LongTensor],
                metadata: List[Dict[str, Any]] = None,
                labels: torch.IntTensor = None) -> Dict[str, torch.Tensor]:
        embeded_text = self._text_field_embedder(text)
        logits = self._labels_predictor(embeded_text)
        mask = util.get_text_field_mask(text)
        predicted_probability = torch.sigmoid(logits.cpu())
        if labels is not None:
            # print(metadata)
            loss = util.sequence_cross_entropy_with_logits(logits, labels, mask)
            for metric in self.metrics.values():
                metric(logits, labels, mask)
        output_dict = {"logits": logits , "mask": mask, "tags": labels,
                       "probabilities": predicted_probability,"loss":loss}
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """获取评价指标"""
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}


