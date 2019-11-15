#!/usr/bin/env python
# -*-coding: utf-8 -*-
"""
@File    :   bert_vec.py
@Project :   TextSum
@License :   (C)Copyright  2019- 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019-11-059:28    yuwc      0.1          
"""

import os
import json
import contextlib
from enum import Enum
from .. utils import fileprocessing as fp
from .. utils.logger import logger as lg
from .. utils.bert import modeling
from .. utils.bert.tokenization import FullTokenizer
from .. utils.bert.extract_features import convert_lst_to_features


class PoolingStrategy(Enum):
    NONE = 0
    REDUCE_MAX = 1
    REDUCE_MEAN = 2
    REDUCE_MEAN_MAX = 3
    FIRST_TOKEN = 4  # corresponds to [CLS] for single sequences
    LAST_TOKEN = 5  # corresponds to [SEP] for single sequences
    CLS_TOKEN = 4  # corresponds to the first token for single seq.
    SEP_TOKEN = 5  # corresponds to the last token for single seq.

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return PoolingStrategy[s]
        except KeyError:
            raise ValueError()


def import_tf(device_id=-1, verbose=False, use_fp16=False):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' if device_id < 0 else str(device_id)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' if verbose else '3'
    os.environ['TF_FP16_MATMUL_USE_FP32_COMPUTE'] = '0' if use_fp16 else '1'
    os.environ['TF_FP16_CONV_USE_FP32_COMPUTE'] = '0' if use_fp16 else '1'
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.DEBUG if verbose else tf.logging.ERROR)
    return tf


def optimize_bert_graph(model_pb_dir='',bert_model_dir='',config_name='',max_seq_len=128):
    try:
        if not os.path.exists(model_pb_dir):
            os.mkdir(model_pb_dir)
        pb_file = os.path.join(model_pb_dir, 'bert_model.pb')
        if os.path.exists(pb_file):
            return pb_file
        # we don't need GPU for optimizing the graph
        tf = import_tf(verbose=False)
        from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference

        config = tf.ConfigProto(device_count={'GPU': 0}, allow_soft_placement=True)

        config_fp = os.path.join(bert_model_dir, config_name)
        init_checkpoint = os.path.join(bert_model_dir, 'bert_model.ckpt')
        lg.info('model config: %s' % config_fp)
        lg.info('checkpoint%s: %s' % (
            ' (override by the fine-tuned model)' if bert_model_dir else '', init_checkpoint))
        with tf.gfile.GFile(config_fp, 'r') as f:
            bert_config = modeling.BertConfig.from_dict(json.load(f))

        lg.info('build graph...')
        # input placeholders, not sure if they are friendly to XLA
        input_ids = tf.placeholder(tf.int32, (None, max_seq_len), 'input_ids')
        input_mask = tf.placeholder(tf.int32, (None, max_seq_len), 'input_mask')
        input_type_ids = tf.placeholder(tf.int32, (None, max_seq_len), 'input_type_ids')

        jit_scope = tf.contrib.compiler.jit.experimental_jit_scope if False else contextlib.suppress

        with jit_scope():
            input_tensors = [input_ids, input_mask, input_type_ids]

            model = modeling.BertModel(
                config=bert_config,
                is_training=False,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=input_type_ids,
                use_one_hot_embeddings=False)

            tvars = tf.trainable_variables()

            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

            minus_mask = lambda x, m: x - tf.expand_dims(1.0 - m, axis=-1) * 1e30
            mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
            masked_reduce_max = lambda x, m: tf.reduce_max(minus_mask(x, m), axis=1)
            masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
                    tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)

            with tf.variable_scope("pooling"):
                pooling_layer = [-2]
                if len(pooling_layer) == 1:
                    encoder_layer = model.all_encoder_layers[pooling_layer[0]]
                else:
                    all_layers = [model.all_encoder_layers[l] for l in pooling_layer]
                    encoder_layer = tf.concat(all_layers, -1)
                pooling_strategy = PoolingStrategy.REDUCE_MEAN
                input_mask = tf.cast(input_mask, tf.float32)
                if pooling_strategy == PoolingStrategy.REDUCE_MEAN:
                    pooled = masked_reduce_mean(encoder_layer, input_mask)
                elif pooling_strategy == PoolingStrategy.REDUCE_MAX:
                    pooled = masked_reduce_max(encoder_layer, input_mask)
                elif pooling_strategy == PoolingStrategy.REDUCE_MEAN_MAX:
                    pooled = tf.concat([masked_reduce_mean(encoder_layer, input_mask),
                                        masked_reduce_max(encoder_layer, input_mask)], axis=1)
                elif pooling_strategy == PoolingStrategy.FIRST_TOKEN or \
                        pooling_strategy == PoolingStrategy.CLS_TOKEN:
                    pooled = tf.squeeze(encoder_layer[:, 0:1, :], axis=1)
                elif pooling_strategy == PoolingStrategy.LAST_TOKEN or \
                        pooling_strategy == PoolingStrategy.SEP_TOKEN:
                    seq_len = tf.cast(tf.reduce_sum(input_mask, axis=1), tf.int32)
                    rng = tf.range(0, tf.shape(seq_len)[0])
                    indexes = tf.stack([rng, seq_len - 1], 1)
                    pooled = tf.gather_nd(encoder_layer, indexes)
                elif pooling_strategy == PoolingStrategy.NONE:
                    pooled = mul_mask(encoder_layer, input_mask)
                else:
                    raise NotImplementedError()

            if False:
                pooled = tf.cast(pooled, tf.float16)

            pooled = tf.identity(pooled, 'final_encodes')
            output_tensors = [pooled]
            tmp_g = tf.get_default_graph().as_graph_def()

        with tf.Session(config=config) as sess:
            lg.info('load parameters from checkpoint...')

            sess.run(tf.global_variables_initializer())
            dtypes = [n.dtype for n in input_tensors]
            lg.info('optimize...')
            tmp_g = optimize_for_inference(
                tmp_g,
                [n.name[:-2] for n in input_tensors],
                [n.name[:-2] for n in output_tensors],
                [dtype.as_datatype_enum for dtype in dtypes],
                False)

            lg.info('freeze...')
            tmp_g = convert_variables_to_constants(sess, tmp_g, [n.name[:-2] for n in output_tensors],
                                                   use_fp16=False)

        lg.info('write graph to a tmp file: %s' % model_pb_dir)
        with tf.gfile.GFile(pb_file, 'wb') as f:
            f.write(tmp_g.SerializeToString())

    except Exception :
        lg.error('fail to optimize the graph!', exc_info=True)
        lg.error(str(Exception))


def convert_variables_to_constants(sess,
                                   input_graph_def,
                                   output_node_names,
                                   variable_names_whitelist=None,
                                   variable_names_blacklist=None,
                                   use_fp16=False):
    from tensorflow.python.framework.graph_util_impl import extract_sub_graph
    from tensorflow.core.framework import graph_pb2
    from tensorflow.core.framework import node_def_pb2
    from tensorflow.core.framework import attr_value_pb2
    from tensorflow.core.framework import types_pb2
    from tensorflow.python.framework import tensor_util

    def patch_dtype(input_node, field_name, output_node):
        if use_fp16 and (field_name in input_node.attr) and (input_node.attr[field_name].type == types_pb2.DT_FLOAT):
            output_node.attr[field_name].CopyFrom(attr_value_pb2.AttrValue(type=types_pb2.DT_HALF))

    inference_graph = extract_sub_graph(input_graph_def, output_node_names)

    variable_names = []
    variable_dict_names = []
    for node in inference_graph.node:
        if node.op in ["Variable", "VariableV2", "VarHandleOp"]:
            variable_name = node.name
            if ((variable_names_whitelist is not None and
                 variable_name not in variable_names_whitelist) or
                    (variable_names_blacklist is not None and
                     variable_name in variable_names_blacklist)):
                continue
            variable_dict_names.append(variable_name)
            if node.op == "VarHandleOp":
                variable_names.append(variable_name + "/Read/ReadVariableOp:0")
            else:
                variable_names.append(variable_name + ":0")
    if variable_names:
        returned_variables = sess.run(variable_names)
    else:
        returned_variables = []
    found_variables = dict(zip(variable_dict_names, returned_variables))

    output_graph_def = graph_pb2.GraphDef()
    how_many_converted = 0
    for input_node in inference_graph.node:
        output_node = node_def_pb2.NodeDef()
        if input_node.name in found_variables:
            output_node.op = "Const"
            output_node.name = input_node.name
            dtype = input_node.attr["dtype"]
            data = found_variables[input_node.name]

            if use_fp16 and dtype.type == types_pb2.DT_FLOAT:
                output_node.attr["value"].CopyFrom(
                    attr_value_pb2.AttrValue(
                        tensor=tensor_util.make_tensor_proto(data.astype('float16'),
                                                             dtype=types_pb2.DT_HALF,
                                                             shape=data.shape)))
            else:
                output_node.attr["dtype"].CopyFrom(dtype)
                output_node.attr["value"].CopyFrom(attr_value_pb2.AttrValue(
                    tensor=tensor_util.make_tensor_proto(data, dtype=dtype.type,
                                                         shape=data.shape)))
            how_many_converted += 1
        elif input_node.op == "ReadVariableOp" and (input_node.input[0] in found_variables):
            # placeholder nodes
            # print('- %s | %s ' % (input_node.name, input_node.attr["dtype"]))
            output_node.op = "Identity"
            output_node.name = input_node.name
            output_node.input.extend([input_node.input[0]])
            output_node.attr["T"].CopyFrom(input_node.attr["dtype"])
            if "_class" in input_node.attr:
                output_node.attr["_class"].CopyFrom(input_node.attr["_class"])
        else:
            # mostly op nodes
            output_node.CopyFrom(input_node)

        patch_dtype(input_node, 'dtype', output_node)
        patch_dtype(input_node, 'T', output_node)
        patch_dtype(input_node, 'DstT', output_node)
        patch_dtype(input_node, 'SrcT', output_node)
        patch_dtype(input_node, 'Tparams', output_node)

        if use_fp16 and ('value' in output_node.attr) and (
                output_node.attr['value'].tensor.dtype == types_pb2.DT_FLOAT):
            # hard-coded value need to be converted as well
            output_node.attr['value'].CopyFrom(attr_value_pb2.AttrValue(
                tensor=tensor_util.make_tensor_proto(
                    output_node.attr['value'].tensor.float_val[0],
                    dtype=types_pb2.DT_HALF)))

        output_graph_def.node.extend([output_node])

    output_graph_def.library.CopyFrom(inference_graph.library)
    return output_graph_def


class BertVector:
    def __init__(self, max_seq_len=128, mask_cls_sep=False, gpu_memory_fraction=0.5, verbose=False,
                 mode='BERT', id2label=None):
        self.first_run = True
        self.predictions = None
        self.max_seq_len = max_seq_len
        self.mask_cls_sep = mask_cls_sep
        self.daemon = True
        self.prefetch_size = None  # set to zero for CPU-worker
        self.gpu_memory_fraction = gpu_memory_fraction
        self.verbose = verbose
        self.mode = mode
        self.id2label = id2label

        self.graph_path = optimize_bert_graph(model_pb_dir=fp.get_full_path('nlp_api/models/chinese_L-12_H-768_A-12/'),
                                              bert_model_dir=fp.get_full_path('nlp_api/models/chinese_L-12_H-768_A-12/'),
                                              config_name='bert_config.json')
        self.device_id = 0
        self.tf = import_tf(self.device_id, self.verbose)
        self.estimator = self.get_estimator(self.tf)
        self.vocab_file = os.path.join(fp.get_full_path('nlp_api/models/chinese_L-12_H-768_A-12/'),'vocab.txt')
        self.tokenizer = FullTokenizer(self.vocab_file)
        self.content =['你好']
        self.predictions = self.estimator.predict(input_fn=self.input_fn_builder(self.content, tf=self.tf),
                                                  yield_single_examples=False)
        #next(self.predictions)

    def input_fn_builder(self, content='',tf=None):

        def input_fn():
            a = tf.data.Dataset.from_generator(
                self.gen,
                output_types={'input_ids': tf.int32,
                              'input_mask': tf.int32,
                              'input_type_ids': tf.int32},
                output_shapes={
                    'input_ids': (None, self.max_seq_len),
                    'input_mask': (None, self.max_seq_len),
                    'input_type_ids': (None, self.max_seq_len)})
            return a

        return input_fn

    def gen(self):
        # 对接收到的字符进行切词，并且转化为id格式
        while True:
            print('next msg:%s, type:%s' % (self.content[0], type(self.content[0])))
            is_token = all(isinstance(el, list) for el in self.content)
            tmp_f = list(convert_lst_to_features(self.content, self.max_seq_len, self.tokenizer, lg,
                                                 is_token, self.mask_cls_sep))
            # print([f.input_ids for f in tmp_f])
            yield {
                'input_ids': [f.input_ids for f in tmp_f],
                'input_mask': [f.input_mask for f in tmp_f],
                'input_type_ids': [f.input_type_ids for f in tmp_f]}

    def get_estimator(self, tf):
        from tensorflow.python.estimator.estimator import Estimator
        from tensorflow.python.estimator.run_config import RunConfig
        from tensorflow.python.estimator.model_fn import EstimatorSpec

        def model_fn(features, labels, mode, params):
            with tf.gfile.GFile(self.graph_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            input_names = ['input_ids', 'input_mask', 'input_type_ids']

            output = tf.import_graph_def(graph_def,
                                         input_map={k + ':0': features[k] for k in input_names},
                                         return_elements=['final_encodes:0'])

            return EstimatorSpec(mode=mode, predictions={
                  'encodes': output[0]
            })

        config = tf.ConfigProto(device_count={'GPU': 0 if self.device_id < 0 else 1})
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = self.gpu_memory_fraction
        config.log_device_placement = False
        # session-wise XLA doesn't seem to work on tf 1.10
        # if args.xla:
        #     config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        return Estimator(model_fn=model_fn, config=RunConfig(session_config=config))

    def calc_vec(self, content=''):
        # 预测函数
        self.content.clear()
        self.content.append(content)
        result = next(self.predictions)
        if self.mode == 'BERT':
            print('pred_label_result:', result['encodes'])
            return result['encodes']
