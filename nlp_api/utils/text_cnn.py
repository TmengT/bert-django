#!/usr/bin/env python
# -*-coding: utf-8 -*-
'''
@File    :   text_cnn.py    
@Contact :   
@License :   (C)Copyright  2019- 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019-03-15 9:22         0.1         None
'''
import tensorflow as tf
import numpy as np


class TextCNN(object):
    '''
    A CNN for text classification
    Uses and embedding layer, followed by a convolutional, max-pooling and softmax layer.
    '''

    def __init__(
            self, sequence_length, num_classes,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        # Placeholders for input, output, dropout
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        # self.embedded_chars = [None(batch_size), sequence_size, embedding_size]
        # self.embedded_chars = [None(batch_size), sequence_size, embedding_size, 1(num_channels)]
        self.embedded_chars = self.input_x
        #增加tensor的维度，-1 表示在最后增加一维
        self.embedded_chars_expended = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):  # "filter_sizes", "3,4,5,6",

            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution layer 第二位 与输入的第三位  保持一致，文本只需要朝一个方向卷积
                filter_shape = [filter_size, embedding_size, 1, num_filters]   # = 200
                #正态分布均值mean为0  标准差stddev是0.1   卷积核 W
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                #创建1-D 0.1 张量
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                #卷积函数
                conv = tf.nn.conv2d(self.embedded_chars_expended,
                                    W,
                                    strides=[1, 1, 1, 1],
                                    padding="VALID",
                                    name="conv")
                # Apply nonlinearity
                #加上b 0.1的 张量之后，计算 卷积张量中 小于0的数，全部置零。
                # relu 激活函数
                # bias_add  张量conv的每行 与 b 进行相加
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        # self.h_pool = tf.concat(3, pooled_outputs)
        #调整张量的 形状，-1 表示自动计算该维度的值，shape中只能有一位是 -1
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout 防止过拟合  随机过滤一些 神经元不参与计算
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnomalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())

            b = tf.Variable(tf.constant(0.1, shape=[num_classes], name="b"))
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            #h_drop*W+b
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
