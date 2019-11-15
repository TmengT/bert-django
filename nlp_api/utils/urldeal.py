#!/usr/bin/env python
# -*-coding: utf-8 -*-
'''
@File    :   urldeal.py
@Project :   TextSum
@License :   (C)Copyright  2019- 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/5/69:16      yuwc      0.1          请求数据接口的
'''

from urllib import parse,request
import json


base_url = "http://172.31.7.169:8888/nlp"

#GET请求数据
def get_req(data={}):

    data = parse.urlencode(data)
    #print(data)
    # for key in data:
    #     url = url+key+'='+data[key]+'&'
    # url= url.rstrip('&')
    #url = parse.urlencode(url)
    header_dict = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko'}
    req = request.Request(url='%s%s%s' % (base_url,'?',data),headers=header_dict)
    res = request.urlopen(req)
    res = res.read()
    #print(res)
    print(res.decode(encoding='utf-8'))
    return res.decode(encoding='utf-8')

#POST 请求数据
def post_req(data={}):
    data = json.dumps(data).encode(encoding='utf-8')
    #data = parse.urlencode(data).encode(encoding='utf-8')
    header_dict = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko',
                   "Content-Type": "application/json"}
    req = request.Request(url=base_url, data=data, headers=header_dict)
    res = request.urlopen(req)
    res = res.read()
    #print(res.decode(encoding='utf-8'))
    res = json.loads(res.decode(encoding='utf-8'))
    if 'result' in res:
        return res['result']
    else:
        return ""

def local_segmemt():
    pass

def main():
    #测试get请求
    data ={}
    data["method"] ="token"
    data["type"] = "2"
    data["content"] = "天真的你"
    get_req(data)
    post_req(data)


if __name__ =='__main__':
    main()