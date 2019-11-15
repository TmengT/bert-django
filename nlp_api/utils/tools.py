#!/usr/bin/env python
# -*-coding: utf-8 -*-
'''
@File    :   tools.py
@Project :   personlabel
@License :   (C)Copyright  2019- 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/6/1310:17    yuwc      0.1
'''

from pyhanlp import HanLP
from multiprocessing import cpu_count
from .mysql_db import MysqlDb
from .urldeal import post_req
import datetime
import time
import re
from . import config_parser as cp


#词性编码表 other  数字代表地点， 小写字母代表的机构单位，大字母N代表的职位，B代表职级 m代表时间
pos_code = eval(cp.get_value("dict","pos_code"))
def special_level(sp_list,text,level):
    '''
    特殊的在职状态  挂职、借调
    :param sp_list:
    :param text:
    :param level:
    :return:
    '''
    for sp in sp_list:
        if sp in text:
            level+= ","+sp
    return level.lstrip(",")

def re_search(pattern, string):
    '''
    正则匹配是否能够符合规则
    :param pattern:
    :param string:
    :return:
    '''
    res = re.search(pattern, string)
    if res is None :
        return False
    else:
        return True


def hanlp_segment(data="", flag_remote = False):
    '''
    # hanlp分词
    :param data:
    :param flag_remote:
    :return:
    '''

    if data is None or len(data) == 0:
        return ""

    seg = ""
    if flag_remote:
        data = {}
        data["method"] = "token"
        data["type"] = "2"
        data["content"] = data
        seg = post_req(data).strip('[').strip(']')

    else:
        seg = HanLP.segment(data).toString().strip('[').strip(']')

    result = seg.split(',')
    return result

def find_org(dest,src):
    mat =re.search(dest,src)
    line = ""
    if mat is not None:
        line = mat.group(1)
    return  line

def find_all_pos(dest,src):
    '''
    匹配出所有 dest在src中的位置
    :param dest:
    :param src:
    :return:
    '''
    return [i.start() for i in re.finditer(dest, src)]

def find_all_p(dest,src):
    '''
    匹配出所有 dest在src中的位置
    :param dest:
    :param src:
    :return:
    '''
    return [(i.start(),i.end()) for i in re.finditer(dest, src)]

def find_somestr(dest,src):
    '''
    发现是否包含某个子串
    :param dest:
    :param src:
    :return:
    '''
    res = re.findall(dest,src)
    if len(res)>0:
        return True
    else:
        return False

def find_allstr(dest=[],src=""):
    '''
    必须包含所有子串
    :param dest:
    :param src:
    :return:
    '''
    flag = True
    for d in dest:
        res = re.findall(r"(%s)" % (d),src)
        if len(res)>0:
            flag = True
        else:
            flag = False
            return flag
    return flag

#把datetime转成字符串
def datetime_toString(dt):
    return dt.strftime("%Y-%m-%d-%H")

#把字符串转成datetime
def string_toDatetime(string):
    return datetime.datetime.strptime(string, "%Y%m")

#把字符串转成时间戳形式
def string_toTimestamp(strTime):
    return time.mktime(string_toDatetime(strTime).timetuple())

#把时间戳转成字符串形式
def timestamp_toString(stamp):
    return time.strftime("%Y-%m-%d-%H", time.localtime(stamp))

#把datetime类型转外时间戳形式
def datetime_toTimestamp(dateTim):
    return time.mktime(dateTim.timetuple())

db1 = cp.get_dict("MYSQLDB1")
db2 = cp.get_dict("MYSQLDB2")
mysqldbnew = MysqlDb(db1)
mysqldblocal = MysqlDb(db2)
#mysqldblocal = MysqlDb(db1["user"],db1["password"],db1["host"],db1["port"],db1["db"],mutilThread=False)
commit_batch_max_num = 2000

def get_cpu_count():
    return cpu_count()

def replaceAll(line = "",src=""):
    '''
    替换所有字符
    :param line:
    :param src:
    :return:
    '''
    for s in line.split("|"):
        src = src.replace("s","")
    return src
if __name__ == "__main__":

    # l_list =[]
    # for line in fp.readfile("D:\\county1.txt"):
    #     if len(line)> 3:
    #         if line.find("自治县")>0 or line.find("县")>0:
    #             line = line.replace("自治县","").replace("县","")
    #         elif line.find("矿区")>0 or line.find("区")>0 or line.find("市")>0:
    #             line = line.replace("矿区","").replace("区","").replace("市","")
    #         if len(line) <3:
    #             continue
    #         l_list.append(line)
    #
    # l_list = list(set(l_list))
    # fp.writefile("D:\\county2.txt",l_list)


    ll = '34324234http:\\\/\\\/slide.ent.sina.com.cn\\\/star\\\/slide_4_704_317986.htmlsdasdasdhttp:\\\/\\\/slide.ent.sina.com.cn\\\/star\\\/slide_4_704_317986.htmlsdfs'
    a = find_allstr(r"http([\w]*)html",ll)

    print(a)
