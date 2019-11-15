#!/usr/bin/env python
# -*-coding: utf-8 -*-
'''
@File    :   word_cloud.py
@Project :   TextSum
@License :   (C)Copyright  2019- 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/5/15 14:18    whz      0.1
'''
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from scipy.misc import imread
#import imageio
import matplotlib.pyplot as plt
import collections
from . import fileprocessing as fp
from . import segment  as seg


__f_path = fp.get_full_path('nlp_api/data/wordcloud/seg_result/')
__imagePath = fp.get_full_path('nlp_api/data/wordcloud/bkg/')
__savePath = fp.get_full_path('static/')
__ttfPath = fp.get_full_path('nlp_api/data/wordcloud/fonts/simhei.ttf')
__stopWordPath = fp.get_full_path('nlp_api/data/wordcloud/stop_words.txt')

def word_cloud(seg_file,image_name='nan.jpg',):

    #读取背景图片
    alice_coloring = imread(__imagePath+image_name)

    #读取分词后的文件
    if not fp.file_is_exist(seg_file):
        raise RuntimeError("文件不存在!")
    cloud_name = seg_file.split('/')[-1].split('.')[0]

    words = seg.segment_file(seg_file,token='standard')
    text = fp.readfile(seg_file)
    stop_words = seg.load_stopWords(__stopWordPath)

    word_counts = {}
    for w in words:
        if len(w) == 1 and w in stop_words:
            continue
        if w not in word_counts:
            word_counts[w] = 1
        else:
            word_counts[w] += 1
    #     word_list.append(words)
    # word_dict = collections.Counter(word_list)

    wc = WordCloud(font_path=__ttfPath,
                   background_color="black",  # 背景颜色 white black
                   max_words=100,  # 词云显示的最大词数
                   mask=alice_coloring,  # 设置背景图片
                   max_font_size=200,  # 字体最大值
                   random_state=42,
                   width=alice_coloring.shape[1], height=alice_coloring.shape[0], margin=2,
                   # 设置图片默认的大小,但是如果使用背景图片的话,那么保存的图片大小将会按照其
                   )
    # wc.generate(str(text))
    wc.generate_from_frequencies(word_counts)
    # show
    plt.imshow(wc)
    plt.axis("off")
    plt.figure()
    #plt.show()
    c_path = __savePath + cloud_name+'.png'
    wc.to_file(c_path)
    return '/static/'+cloud_name+'.png'

def main():

    word_cloud('res.txt','cloud.jpg')

if __name__ == '__main__':
    main()