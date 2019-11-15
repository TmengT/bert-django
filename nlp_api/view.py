#!/usr/bin/env python
# -*-coding: utf-8 -*-
'''
@File    :   view.py
@Project :   PyCharm
@License :   (C)Copyright  2019-

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/4/17 9:24    yuwc      0.1          views视图
'''

from django.http  import HttpResponse,QueryDict
from django.shortcuts import render
from django.urls import reverse
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.request import Request
from rest_framework.views import APIView
from .method import Segment,Semantic,smart_accumulate,bert_ner,bert_vec
import time
from pyhanlp import *
from .utils import fileprocessing as fp

print("loading  model begin !")
segment = Segment()
semantic = Semantic()
HanLP.segment('nihao!')
print("loading   model end !")


@api_view(['GET','POST','DELETE','PUT'])
def hello(request):
    if request.method == 'GET':
        return HttpResponse("GET Hello  django!")
    elif request.method =='POST':
        print(request.data)
        return Response(request.data)
    elif request.method == 'DELETE':
        return HttpResponse("DELETE Hello django!")
    elif request.method == 'PUT':
        return  HttpResponse("PUT Hello django!")

    return HttpResponse("Hello  django!")
    #return HttpResponse(status=404)
    #return Response(status=status.HTTP_400_BAD_REQUEST)


def hello_path(request):
    return HttpResponse('Hello path')


def hello_html(request):
    context = {}
    context['hello'] = 'hello html!'
    return render(request,"hello.html",context)


def redict_path(request):
    return HttpResponse(reverse("nlp"))


#支持可以跨域访问
"""
@csrf_exempt
def  segment(request):
    if request.method == 'GET':
        pass
    elif request.method == 'POST':
        pass
    return JsonResponse(status=400)
"""


@api_view(['GET','POST'])
def segment_content(request):
    return Response('content error !', status=status.HTTP_400_BAD_REQUEST)\


def cost_time(func):
    # 定义一个 计时装饰器
    def wrapper(*args, **kwargs):
        print("Before %s called"%(func.__name__))
        begin = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        print("After %s called,cost time : %s "%(func.__name__,str(end-begin)))
        result={}
        if (type(res).__name__ == 'dict') and 'result' in res:
            result = res
        else:
            result['result'] = res
        result['cost_time'] = str(end-begin)
        return result
    return wrapper


@api_view(['GET','POST'])
def upload_file(request):
    # 请求方法为POST时，进行处理
    if request.method == "POST":
        # 获取上传的文件，如果没有文件，则默认为None
        file = request.FILES.get("myfile", None)
        if file is None:
            return HttpResponse("没有需要上传的文件")
        else:
            # 打开特定的文件进行二进制的写操作
            f_path = fp.get_full_path('nlp_api/data/upload/' + file.name)
            with open(f_path, 'wb+') as f:
                # 分块写入文件
                for chunk in file.chunks():
                    f.write(chunk)
            data = {}
            data['fname'] = f_path
            img_ret = word_cloud(data)
            return render(request, 'upload.html', {'pic': img_ret['result'],'data':'OK'})
    else:
        return render(request, "upload.html")


# APIView 接口类
class Nlp_API(APIView):

    """
    统一接口入口
    """
    def get(self, request, *args, **kwargs):
        # request.setCharacterEncoding("utf-8")
        # request.setContentType("text/html;charset=utf-8")
        data = get_parameter_dic(request)
        if data is None:
            return Response("Please input params ，for example :" \
                            "http://ip:port/nlp?method=semantic&w1=赵本山")
        print(data)
        return Response(method_router(data,*args,**kwargs))

    def post(self, request, *args, **kwargs):
        data = get_parameter_dic(request)
        if data == {}:
            return Response("Please input params ，for example :" \
                            "http://ip:port/nlp?method=semantic&w1=赵本山")
        print(data)
        return Response(method_router(data,*args,**kwargs))


# 获取请求的参数
def get_parameter_dic(request):
    if isinstance(request,Request) is False:
        return {}
    query_params = request.query_params
    if isinstance(query_params,QueryDict):
        query_params = query_params.dict()
    query_data = request.data
    if isinstance(query_data,QueryDict):
        query_data = query_data.dict()

    if query_params != {}:
        return query_params

    if query_data != {}:
        return query_data


# 算法路由
def method_router(data,*args,**kwargs):
    # 获取文件后缀
    method = data["method"]
    switch = {
        "bertNER":lambda x: bert_ner_func(x),  # 基于bert bi-LSTM CRF的 NER
        "bertvec":lambda x: bert_vec_func(x),
    }

    result = {}
    try:
        result = switch[method](data)
    except Exception as e:
        print(e)
        pass
    if result is None:
        return "Result is nothing ."
    return result


# 基于bert bi-LSTM CRF 的实体识别
@cost_time
def bert_ner_func(data):
    return bert_ner(data)


@cost_time
def bert_vec_func(data):
    return bert_vec(data)
