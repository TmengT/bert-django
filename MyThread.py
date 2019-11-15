#!/usr/bin/env python
# -*-coding: utf-8 -*-
"""
@File    :   thread.py
@Project :   nlp_api
@License :   (C)Copyright  2019- 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019-11-0815:12    yuwc      0.1          
"""

import threading
from multiprocessing import cpu_count


class MyThread(threading.Thread):
    def __init__(self, thread_id, name, data=[],process=None):
        threading.Thread.__init__(self)
        self.threadID = thread_id
        self.name = name
        self.data = data
        self.process = process

    def run(self):
        print("开启线程：" + self.name)
        self.process(self.name,self.data)
        #process_data(self.name,self.data)
        print("退出线程：" + self.name)


def process_data(thread_name, data=[]):
    for e in data :
        print(thread_name + '\t' + str(e)+ '\n')


if __name__ == '__main__':
    cpu_num = cpu_count()
    a = range(1000)
    thread_num = cpu_num + 1
    batch_size = int(len(a)/thread_num)
    threads = []
    for i in range(thread_num):
        if i+1 < thread_num:
            thread = MyThread(i,'thead'+str(i),a[i*batch_size:(i+1)*batch_size],process_data)
            thread.start()
            threads.append(thread)
        else:
            thread = MyThread(i,'thead'+str(i),a[i*batch_size:],process_data)
            thread.start()
            threads.append(thread)
    for t in threads:
        t.join()
    print("process  is  done !")
