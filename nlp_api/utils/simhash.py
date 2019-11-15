#!/usr/bin/env python
# -*-coding: utf-8 -*-
'''
@File    :   simhash.py
@Project :   TextSum
@License :   (C)Copyright  2019- 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/5/1419:20    whz      0.1          
'''
# -*- coding: utf-8 -*-
import jieba
import jieba.analyse
import numpy as np


class simhash:
    def __init__(self, content):
        self.simhash = self.simhash(content)

    def __str__(self):
        return str(self.simhash)

    def simhash(self, content):
        seg = jieba.cut(content)
        jieba.analyse.set_stop_words('stopword.txt')
        keyWord = jieba.analyse.extract_tags(
            '|'.join(seg), topK=20, withWeight=True, allowPOS=())  # 在这里对jieba的tfidf.py进行了修改
        # 将tags = sorted(freq.items(), key=itemgetter(1), reverse=True)修改成tags = sorted(freq.items(), key=itemgetter(1,0), reverse=True)
        # 即先按照权重排序，再按照词排序
        keyList = []
        # print(keyWord)
        for feature, weight in keyWord:
            weight = int(weight * 20)
            feature = self.string_hash(feature)
            temp = []
            for i in feature:
                if (i == '1'):
                    temp.append(weight)
                else:
                    temp.append(-weight)
            # print(temp)
            keyList.append(temp)
        list1 = np.sum(np.array(keyList), axis=0)
        print(list1)
        if (keyList == []):  # 编码读不出来
            return '00'
        simhash = ''
        for i in list1:
            if (i > 0):
                simhash = simhash + '1'
            else:
                simhash = simhash + '0'
        return simhash

    def string_hash(self, source):
        if source == "":
            return 0
        else:
            x = ord(source[0]) << 7
            m = 1000003
            mask = 2 ** 128 - 1
            for c in source:
                x = ((x * m) ^ ord(c)) & mask
            x ^= len(source)
            if x == -1:
                x = -2
            x = bin(x).replace('0b', '').zfill(64)[-64:]
            print(source, x)

            return str(x)

        '''
        以下是使用系统自带hash生成，虽然每次相同的会生成的一样，
        不过，对于不同的汉子产生的二进制，在计算海明码的距离会不一样，
        即每次产生的海明距离不一致
        所以不建议使用。
        '''
        # x=str(bin(hash(source)).replace('0b','').replace('-','').zfill(64)[-64:])
        # print(source,x,len(x))
        # return x

    def hammingDis(self, com):
        t1 = '0b' + self.simhash
        t2 = '0b' + com.simhash
        n = int(t1, 2) ^ int(t2, 2)
        i = 0
        while n:
            n &= (n - 1)
            i += 1
        return i

