#!/usr/bin/env python
# -*-coding: utf-8 -*-
'''
@File    :   ft_classify.py
@Project :   TextSum
@License :   (C)Copyright  2019- 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/5/1511:30    whz      0.1          
'''
import fastText.FastText as ft
import fastText

train_path = ""
model_path = ""
test_path = ""

def train():
    classifier = ft.train_supervised(train_path)
    model = classifier.save_model(model_path)
    test = classifier.test(test_path)
    print("准确率:", test.precision)
    print("回归率:", test.recall)
    classifier.get_labels()

def predict():
    # 使用模型,以测试集中第一个文档为例
    classifier = fastText.load_model(model_path)
    f = open("fasttext_test.txt")
    line = f.readlines()[0]
    f.close()
    result = classifier.predict([line])
    print(result)


