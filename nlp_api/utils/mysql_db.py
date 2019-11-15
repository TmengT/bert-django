#!/usr/bin/env python
# -*-coding: utf-8 -*-
'''
@File    :   mysql_db.py
@Project :   TextSum
@License :   (C)Copyright  2019- 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/5/249:52    whz      0.1          
'''
from sqlalchemy import create_engine
#from sqlalchemy.pool import SingletonThreadPool
import pandas as pd
#import MySQLdb


class  MysqlDb:
    def __init__(self,dict={},mutilThread = False):
        '''
        初始化的数据连接
        :param username:
        :param password:
        :param host:
        :param port:
        :param dbname:
        '''
        self._db_info = dict
        #print(self._db_info)
        if not mutilThread :
            self.engine = create_engine(
                'mysql+mysqlconnector://%(user)s:%(password)s@%(host)s:%(port)s/%(db)s?charset=utf8' % self._db_info,
                encoding='utf-8')#poolclass=SingletonThreadPool)
        else :
            self.engine = create_engine(
                'mysql+mysqlconnector://%(user)s:%(password)s@%(host)s:%(port)s/%(db)s?charset=utf8' % self._db_info,
                encoding='utf-8', max_overflow=2, #超过连接池大小之后，允许最大扩展连接数；
                pool_size=8,   #连接池大小
                pool_timeout=30,#连接池如果没有连接了，最长等待时间
                pool_recycle=-1,#多久之后对连接池中连接进行一次回收
                 )
        # # 打开数据库连接
        # self.db = MySQLdb.connect(self._db_info["host"],self._db_info["user"],self._db_info["password"],self._db_info["db"])
        #
        # # 使用cursor()方法获取操作游标
        # self.cursor = self.db.cursor()

    #def __del__(self):

        # if self.engine is not None:
        #     self.engine.dispose()
        # if self.db is not None:
        #     self.db.close()

    def read(self,sql):
        '''
        读取表数据
        :param sql:
        :return:
        '''
        df =pd.read_sql_query(sql,self.engine)
        # print(df.head(1))
        # print(df.index)
        # #print(df.lookup('9','9'))
        # print(df.iterrows())
        return df

    def insert(self,tablename='',data={}):
        '''
        插入表数据
        :param sql:
        :param tablename:
        :param data: {'id': [1, 2, 3, 4], 'name': ['zhangsan', 'lisi', 'wangwu', 'zhuliu']}
        :return:
        '''
        df = pd.DataFrame(data)
        # 将新建的DataFrame储存为MySQL中的数据表，储存index列
        df.to_sql(tablename, self.engine, if_exists='append',index=False)

    # def update(self,sql=''):
    #     # SQL 更新语句
    #     #sql = "UPDATE EMPLOYEE SET AGE = AGE + 1 WHERE SEX = '%c'" % ('M')
    #     try:
    #         # 执行SQL语句
    #         self.cursor.execute(sql)
    #         # 提交到数据库执行
    #         self.db.commit()
    #     except:
    #         # 发生错误时回滚
    #         self.db.rollback()



