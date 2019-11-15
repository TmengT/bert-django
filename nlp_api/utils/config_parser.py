#!/usr/bin/env python
# -*-coding: utf-8 -*-
'''
@File    :   config_parser.py
@Project :   personlabel
@License :   (C)Copyright  2019- 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/6/2815:29    yuwc      0.1
'''

import configparser as cp
import os

#root_dir = os.path.dirname(os.path.abspath('.'))
#root_dir = os.path.dirname(os.path.dirname(os.path.abspath('.')))

root_dir = os.path.abspath('.')
cf = cp.ConfigParser()
c_path = root_dir+"/config.ini"
cf.read(c_path)


def get_all_section():
    return cf.sections()

def get_dict(op=""):
    return dict(cf.items(op))

def get_value(op="",name=""):
    return cf.get(op,name)


if __name__ == "__main__":
    dict = get_dict("MYSQLDB1")
    print(dict)