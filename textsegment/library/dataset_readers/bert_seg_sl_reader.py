# -*- coding: UTF-8 -*-
"""
@author: liangjunjun
@contact: liangjunjun@pku.edu.cn
@version: v0.0.1
@file: bert_seg_sl_reader.py
@time: 2019/9/8 15:36
@desc: 
"""
import json
import logging
from typing import Dict, List, Tuple
from overrides import overrides
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("bert_sl_tagger")
class BertSlTaggerReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 text_length_limit: int = None) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.text_length_limit = text_length_limit

    @overrides
    def _read(self, file_path: str):
        """读取数据"""
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path) as dataset_file:
            for line in dataset_file:
                test,label = line.split('<sep>')
                if not test:
                    continue
                test = test[:500]
                label = label.strip().split('\t')
                label = [int(_label) for _label in label if _label and int(_label)<len(test)-1]
                instance = self.text_to_instance(test, label)
                yield instance

    @overrides
    def text_to_instance(self, text_token_strs: List[str],
                         answers: List[int] = None) -> Instance:
        """数据格式转换"""
        text_tokens = [Token(text) for text in text_token_strs]
        if self.text_length_limit and len(text_tokens) > self.text_length_limit:
            logger.warning(F"passage's length of {len(text_tokens)} is limited to {self.text_length_limit}")
            text_tokens = text_tokens[: self.text_length_limit]
        metadata = {'orig_text_tokens': text_token_strs,
                    'text_tokens': [token.text for token in text_tokens],
                    'original_answer': answers}

        fields: Dict[str, Field] = {'text': TextField(text_tokens, self._token_indexers),
                                    'metadata': MetadataField(metadata)
                                    }

        if answers is not None:
            labels = [0] * len(text_tokens)
            for index in answers:
                labels[index] = 1
            fields['labels'] = SequenceLabelField(labels, fields['text'])

        return Instance(fields)
