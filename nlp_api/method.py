# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   method.py
@Project :   PyCharm
@License :   (C)Copyright  2019-

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/4/17 9:24    yuwc      0.1          方法实现
"""

from .traintools.bert_ner import BertWorker
from .traintools.bert_vec import BertVector


ner = BertWorker()
vec = BertVector()


def check_param_exist(name, dict={}):
    if name not in dict:
        return False
    name = dict[name]
    if name is None or len(name) == 0:
        return False
    return True


# 获取的字典的键值
def get_value(data,k):
    if k in data:
        return data[k]
    return ''


# 单例模式
class Singleton(object):
    _instance = None

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls._instance = super(Singleton, cls).__new__(cls)
        return cls._instance


def bert_ner(data):
    """
    基于bert biLSTM CRF的 NER
    :param data:
    :return:
    """
    content = data['content']
    re_dict = {}
    r = ner.predict(content)
    re_dict['result'] = r
    return re_dict


def bert_vec(data):
    content = data['content']
    result = dict()
    result['result'] = vec.calc_vec(content)
    return result
