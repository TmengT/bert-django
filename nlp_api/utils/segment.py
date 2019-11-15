# -*-coding: utf-8 -*-
#分词统一接口
##
from . import config_parser as cp
import jieba
from . import fileprocessing as fp
from pyhanlp import *
import os
import io
import math
import re


def load_stopWords():
    '''
    加载停用词
    :param path:
    :return:
    '''
    stopwords = []
    for  w in fp.readfile(cp.get_value("path","stop_path")):
        stopwords.append(w.rstrip('\n'))

    return stopwords


def cut_content_jieba(content,flag_pos=False):
    '''
    按字词word进行分割
    :param content: str
    :return:
    '''
    lines_cut = jieba.cut(content)
    return lines_cut

def cut_content_standard(content,flag_pos=False):
    '''
    标准分词
    :param content:
    :return:
    '''
    line_cut = []
    seg = HanLP.newSegment()
    for w in seg.seg(content):
        w = str(w.toString())
        if not flag_pos:
            w = w.split('/')[0]
        line_cut.append(w)
    return line_cut

def cut_content_crf(content,flag_pos=False):
    '''
    crf 分词
    :param content:
    :return:
    '''
    line_cut = []
    seg = HanLP.newSegment('crf')
    for w in seg.seg(content):
        w = str(w.toString())
        if not flag_pos:
            w = w.split('/')[0]
        line_cut.append(w)

def delete_stopwords(lines_list,stopwords=[]):
    '''刪除停用詞'''
    sentence_segment=[]
    for word in lines_list:
        if word not in stopwords:
            sentence_segment.append(word)
    return sentence_segment

def padding_sentence(sentence, padding_token, padding_sentence_length):
    '''
    padding句子长度
    :param sentence: type->list[str]
    :param padding_token:
    :param padding_sentence_length:
    :return:
    '''
    if len(sentence) > padding_sentence_length:
        sentence = sentence[:padding_sentence_length]
    else:
        sentence.extend([padding_token] * (padding_sentence_length - len(sentence)))
    return sentence

def padding_sentences(sentences_list, padding_token, padding_sentence_length):
    '''
    padding句子长度
    :param sentences_list: type->list[list[str]]
    :param padding_token:  设置padding的内容
    :param padding_sentence_length: padding的长度
    :return:
    '''
    for i, sentence in enumerate(sentences_list):
        sentence=padding_sentence(sentence, padding_token, padding_sentence_length)
        sentences_list[i]=sentence
    return sentences_list

def segment_content_word(content,token='1',flag_stop = True,flag_pos = False,stopwords=[]):
    '''
    1 jieba 分词， 2 标准分词， 3 crf分词
    :param content:
    :param stopwords:
    :param token:
    :return:
    '''
    seg_content = []
    if token == '1':
        seg_content=cut_content_jieba(content,flag_pos)
    elif token == '2' :
        seg_content = cut_content_standard(content,flag_pos)
    else:
        seg_content = cut_content_crf(content,flag_pos)

    if flag_stop :
        seg_content = delete_stopwords(seg_content,stopwords)

    return seg_content
def clean_str(string):
    string = re.sub(r"[^\u4e00-\u9fff]", " ", string)
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    # string = re.sub(r"\'s", " \'s", string)
    # string = re.sub(r"\'ve", " \'ve", string)
    # string = re.sub(r"n\'t", " n\'t", string)
    # string = re.sub(r"\'re", " \'re", string)
    # string = re.sub(r"\'d", " \'d", string)
    # string = re.sub(r"\'ll", " \'ll", string)
    # string = re.sub(r",", " , ", string)
    # string = re.sub(r"!", " ! ", string)
    # string = re.sub(r"\(", " \( ", string)
    # string = re.sub(r"\)", " \) ", string)
    # string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    # return string.strip().lower()
    return string.strip()

def seperate_line(line):
    return ''.join([word + ' ' for word in line])

def cut_content_char(content):
    '''
    按字符char进行分割
    :param content: str
    :return:
    '''
    lines_cut = clean_str(seperate_line(content))
    return lines_cut

def segment_content_word(content,stopwords=[]):
    lines_cut_list=cut_content_jieba(content)
    segment_list=delete_stopwords(lines_cut_list,stopwords)
    return segment_list

def segment_content_char(content,stopwords=[]):
    lines_cut_str=cut_content_char(content)
    lines_cut_list = lines_cut_str.split(' ')
    segment_list=delete_stopwords(lines_cut_list,stopwords)
    return segment_list
def read_file_content(file,mode='rb'):
    '''
    读取文件内容，并去除去除头尾字符、空白符(包括\n、\r、\t、' '，即：换行、回车、制表符、空格)
    :param file:
    :param mode:
    :return: str
    '''
    with open(file, mode=mode,encoding='UTF-8') as f:
        lines = f.readlines()
        contents=[]

        for line in lines:
            line=line.strip()
            if line.rstrip()!='':
                contents.append(line)
        contents='\n'.join(contents)
    return contents

def read_files_list_content(files_list,mode='r'):
    '''
    读取文件列表内容，并去除去除头尾字符、空白符(包括\n、\r、\t、' '，即：换行、回车、制表符、空格)
    :param files_list: 文件列表
    :param mode:
    :return: list[str]
    '''
    content_list=[]
    for i , file in enumerate(files_list):
        content=read_file_content(file,mode=mode)
        content_list.append(content)
    return content_list
def segment_file(file, stopwords=[], segment_type='word'):
    '''
    字词分割
    :param file:
    :param stopwords:
    :param segment_type: word or char，选择分割类型，按照字符char，还是字词word分割
    :return:
    '''
    content = read_file_content(file, mode='r')
    if segment_type=='word' or segment_type is None:
        segment_content = segment_content_word(content, stopwords)
    elif segment_type=='char':
        segment_content = segment_content_char(content, stopwords)
    return segment_content

def segment_files_list(files_list,stopwords=[],segment_type='word'):
    '''
    字词分割
    :param files_list:
    :param stopwords:
    :param segment_type: word or char，选择分割类型，按照字符char，还是字词word分割
    :return:
    '''
    content_list=[]
    for i, file in enumerate(files_list):
        segment_content=segment_file(file,stopwords,segment_type)
        content_list.append(segment_content)
    return content_list

def sentence_parse(content,Translator = True):
    '''
    句法分析
    :param content:
    :return:
    '''
    r_list = []
    seg = HanLP.newSegment("crf").enablePartOfSpeechTagging(True);
    dp = DependencyParser(seg).enableDeprelTranslator(Translator)
    for res in dp.parse(content).getWordArray():
        r_list.append(" ".join(res.toString().split('\t')))
    return r_list

if __name__=='__main__':
    print(segment_content_word("风流倜傥",token='1'))

    # 多线程分词
    # jieba.enable_parallel()
    # 加载自定义词典
    # user_path = '../data/user_dict.txt'
    # jieba.load_userdict(user_path)

    sentence_parse("手机非常不错，玩游戏一点压力都没有，颜值非常高，苏宁的服务也非常到位，值得购买的体验！")


