#!/usr/bin/env python
# -*-coding: utf-8 -*-
'''
@File    :   logger.py
@Project :
@License :   (C)Copyright  2019- 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/7/314:04    yuwc      0.1           日志格式输出
'''
import logging
import os
import utils.config_parser as cp

LOG_DIR = cp.root_dir+"/"+cp.get_value("log_info","log_dir")
LOG_NAME = cp.get_value("log_info","log_name")
is_exists = os.path.exists(LOG_DIR)
if not is_exists:
    os.mkdir(LOG_DIR)


f_level = cp.get_value("log_info","file_level")
c_level = cp.get_value("log_info","console_level")


file_handler = logging.FileHandler(LOG_DIR+"/"+LOG_NAME)
file_handler.setLevel(f_level)
console_handler = logging.StreamHandler()
console_handler.setLevel(c_level)
# 获取日志信息
fmt = '%(asctime)s - %(funcName)s - %(lineno)s - %(levelname)s - %(message)s'
formatter = logging.Formatter(fmt)
#logging.basicConfig(format=fmt, level=logging.INFO,filename=LOG_FILE)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger = logging.getLogger('updateSecurity')
logger.setLevel(f_level)
logger.addHandler(file_handler)    #添加handler
logger.addHandler(console_handler)

