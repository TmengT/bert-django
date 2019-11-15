#!/usr/bin/env python
# -*-coding: utf-8 -*-
'''
@File    :   test.py    
@Project :   PyCharm
@License :   (C)Copyright  2019- 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/4/23 15:01    yuwc      0.1         
'''
from jpype import *
import jpype
'''
jvmPath = jpype.getDefaultJVMPath()
print(jvmPath)
jpype.startJVM(jvmPath)
jpype.java.lang.System.out.println("hello world!")
java.lang.System.out.println("hello world")
jpype.shutdownJVM()

startJVM(getDefaultJVMPath(),
         "-Djava.class.path=D:/soft/anaconda/envs/tf/Lib/site-packages/pyhanlp/static/hanlp-1.7.2.jar;D:/soft/anaconda/envs/tf/Lib/site-packages/pyhanlp/static/",
         "-Xms1g",
         "-Xmx1g")
NLPTokenizer = JClass('com.hankcs.hanlp.tokenizer.NLPTokenizer')
print(NLPTokenizer.segment('中国科学院计算技术研究所的教授正在教授自然语言处理课程'))
 '''
def main():
    startJVM(getDefaultJVMPath(),
             "-Djava.class.path=D:/work/python-space/nlp_api/nlp_api/resource/hanlp-1.7.2jar;D:/work/python-space/nlp_api/nlp_api/resource/",
             "-Djava.class.path=D:/soft/anaconda/envs/tf/Lib/site-packages/pyhanlp/static/hanlp-1.7.2.jar;D:/soft/anaconda/envs/tf/Lib/site-packages/pyhanlp/static/",
             "-Xms1g",
             "-Xmx1g")
    print("=" * 30 + "HanLP分词" + "=" * 30)
    HanLP = JClass('com.hankcs.hanlp.HanLP')
    CRFseg = JClass('com.hankcs.hanlp.model.crf.CRFLexicalAnalyzer')
    crf = CRFseg()
    print(crf.analyze('小米，欢迎在Python中调用HanLP的API').toString())
    # 中文分词

   # print(HanLP.segment('你好，欢迎在Python中调用HanLP的API'))
    print("-" * 70)



if __name__ == '__main__':
    main()