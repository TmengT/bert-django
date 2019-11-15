#!/usr/bin/env python
# -*-coding: utf-8 -*-
"""
@File    :   bert_ner.py
@Project :   nlp_api
@License :   (C)Copyright  2019- 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019-10-2311:14    yuwc      0.1          bert_ner
"""

import os
import pickle
import collections
import codecs
from .. utils.logger import logger as lg
from .. utils.bert import modeling
from .. utils.bert.extract_features import convert_lst_to_features
from .. utils.bert.tokenization import FullTokenizer,convert_to_unicode,printable_text
from .. utils import fileprocessing as fp


def optimize_ner_model(model_pb_file='', max_seq_len=128,bert_model_dir='',model_dir='',num_labels=[]):
    """
    :param model_pb_file: ner pb模型文件路径
    :param max_seq_len: 最大长度
    :param bert_model_dir: bert模型文件所在目录
    :param model_dir: ner模型所在目录
    :param num_labels: 标签list
    :return:
    """
    lg.info('NER_MODEL, Loading...')
    try:
        # 如果PB文件已经存在则，返回PB文件的路径，否则将模型转化为PB文件，并且返回存储PB文件的路径

        pb_file = os.path.join(model_pb_file)
        if os.path.exists(pb_file):
            print('pb_file exits', pb_file)
            return pb_file
        # 不存在pb file ，则保存pb文件
        lg.info('%s dont exist,need create and save it! ' % pb_file)
        import tensorflow as tf

        graph = tf.Graph()
        with graph.as_default():
            with tf.Session() as sess:
                input_ids = tf.placeholder(tf.int32, (None, max_seq_len), 'input_ids')
                input_mask = tf.placeholder(tf.int32, (None, max_seq_len), 'input_mask')

                bert_config = modeling.BertConfig.from_json_file(os.path.join(bert_model_dir, 'bert_config.json'))
                from bert_base.train.models import create_model
                (total_loss, logits, trans, pred_ids) = create_model(
                    bert_config=bert_config, is_training=False, input_ids=input_ids, input_mask=input_mask, segment_ids=None,
                    labels=None, num_labels=num_labels, use_one_hot_embeddings=False, dropout_rate=1.0)
                pred_ids = tf.identity(pred_ids, 'pred_ids')
                saver = tf.train.Saver()

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                saver.restore(sess, tf.train.latest_checkpoint(model_dir))
                lg.info('freeze...')
                from tensorflow.python.framework import graph_util
                tmp_g = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), ['pred_ids'])
                lg.info('model cut finished !!!')
        # 存储二进制模型到文件中
        lg.info('write graph to a tmp file: %s' % pb_file)
        with tf.gfile.GFile(pb_file, 'wb') as f:
            f.write(tmp_g.SerializeToString())
        return pb_file
    except Exception as e:
        lg.error('fail to optimize the graph! %s' % e, exc_info=True)


def get_device_map(num_worker=1):
    #获取主机GPU的 序号
    lg.info('get devices')
    run_on_gpu = False
    device_map = [-1] * num_worker
    try:
        import GPUtil
        num_all_gpu = len(GPUtil.getGPUs())
        avail_gpu = GPUtil.getAvailable(order='memory', limit=min(num_all_gpu, num_worker))
        num_avail_gpu = len(avail_gpu)

        if num_avail_gpu >= num_worker:
            run_on_gpu = True
        else:
            lg.warning('no GPU available, fall back to CPU')

        if run_on_gpu:
            device_map = (avail_gpu * num_worker)[: num_worker]
    except FileNotFoundError:
        lg.warning('nvidia-smi is missing, often means no gpu on this machine. fall back to cpu!')
    lg.info('device map: \n\t\t%s' % '\n\t\t'.join(
        'worker %2d -> %s' % (w_id, ('gpu %2d' % g_id) if g_id >= 0 else 'cpu') for w_id, g_id in
        enumerate(device_map)))
    return device_map


def check_tf_version():
    import tensorflow as tf
    tf_ver = tf.__version__.split('.')
    assert int(tf_ver[0]) >= 1 and int(tf_ver[1]) >= 10, 'Tensorflow >=1.10 is required!'
    return tf_ver


def import_tf(device_id=-1, verbose=False, use_fp16=False):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' if device_id < 0 else str(device_id)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' if verbose else '3'
    os.environ['TF_FP16_MATMUL_USE_FP32_COMPUTE'] = '0' if use_fp16 else '1'
    os.environ['TF_FP16_CONV_USE_FP32_COMPUTE'] = '0' if use_fp16 else '1'
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.DEBUG if verbose else tf.logging.ERROR)
    return tf


def convert_id_to_label(pred_ids_result, idx2label, batch_size):
    """
    将id形式的结果转化为真实序列结果
    :param pred_ids_result:
    :param idx2label:
    :return:
    """
    result = []
    index_result = []
    for row in range(batch_size):
        curr_seq = []
        curr_idx = []
        ids = pred_ids_result[row]
        for idx, id in enumerate(ids):
            if id == 0:
                break
            curr_label = idx2label[id]
            if curr_label in ['[CLS]', '[SEP]']:
                if id == 102 and (idx < len(ids) and ids[idx + 1] == 0):
                    break
                continue
            # elif curr_label == '[SEP]':
            #     break
            curr_seq.append(curr_label)
            curr_idx.append(id)
        result.append(curr_seq)
        index_result.append(curr_idx)
    return result, index_result


def init_predict_var(path):
    """
    初始化NER所需要的一些辅助数据
    :param path:
    :return:
    """
    label_list_file = os.path.join(path, 'label_list.pkl')
    label_list = []
    if os.path.exists(label_list_file):
        with open(label_list_file, 'rb') as fd:
            label_list = pickle.load(fd)
    num_labels = len(label_list)

    with open(os.path.join(path, 'label2id.pkl'), 'rb') as rf:
        label2id = pickle.load(rf)
        id2label = {value: key for key, value in label2id.items()}
    return num_labels, label2id, id2label


def ner_result_to_json(predict_ids, id2label):
    """
    NER识别结果转化为真实标签结果进行返回
    :param predict_ids:
    :param id2label
    :return:
    """
    if False:
        return predict_ids
    pred_label_result, pred_ids_result =\
                convert_id_to_label(predict_ids, id2label, len(predict_ids))
    return pred_label_result, pred_ids_result


class BertWorker():
    def __init__(self,max_seq_len=128, mask_cls_sep=False,prefetch_size=10,gpu_memory_fraction=0.5,
                 verbose=False,mode='NER',id2label=None):
        self.first_run = True
        self.predictions = None
        self.max_seq_len = max_seq_len
        self.mask_cls_sep = mask_cls_sep
        self.daemon = True
        self.prefetch_size = None  # set to zero for CPU-worker
        self.gpu_memory_fraction = gpu_memory_fraction
        self.verbose = verbose
        self.mode = mode
        num_labels, label2id, id2label = init_predict_var(fp.get_full_path('nlp_api/models/bert_ner/'))
        self.id2label = id2label
        self.graph_path =optimize_ner_model(model_pb_file=fp.get_full_path('nlp_api/models/bert_ner/ner_model.pb'),
                                            max_seq_len=128,
                                            bert_model_dir=fp.get_full_path('nlp_api/models/chinese_L-12_H-768_A-12/'),
                                            model_dir=fp.get_full_path('nlp_api/models/bert_ner/'),
                                            num_labels=num_labels)
        self.device_id =0
        self.tf = import_tf(self.device_id, self.verbose)
        self.estimator = self.get_estimator(self.tf, self.device_id)
        self.vocab_file = os.path.join(fp.get_full_path('nlp_api/models/chinese_L-12_H-768_A-12/'),
                                                 'vocab.txt')
        self.tokenizer = FullTokenizer(self.vocab_file)
        self.content =['你好']
        self.predictions = self.estimator.predict(input_fn=self.input_fn_builder(self.content, tf=self.tf), yield_single_examples=False)
        next(self.predictions)

    def get_estimator(self, tf,device_id=0):
        # 加载图模型
        from tensorflow.python.estimator.estimator import Estimator
        from tensorflow.python.estimator.run_config import RunConfig
        from tensorflow.python.estimator.model_fn import EstimatorSpec

        def ner_model_fn(features, labels, mode, params):
            """
            命名实体识别模型的model_fn
            :param features:
            :param labels:
            :param mode:
            :param params:
            :return:
            """
            with tf.gfile.GFile(self.graph_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
            input_ids = features["input_ids"]
            input_mask = features["input_mask"]
            input_map = {"input_ids": input_ids, "input_mask": input_mask}
            pred_ids = tf.import_graph_def(graph_def, name='', input_map=input_map, return_elements=['pred_ids:0'])

            return EstimatorSpec(mode=mode, predictions={
                'encodes': pred_ids[0]
            })

        # 0 表示只使用CPU 1 表示使用GPU
        config = tf.ConfigProto(device_count={'GPU': 0 if device_id < 0 else 1})
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = self.gpu_memory_fraction
        config.log_device_placement = False
        # session-wise XLA doesn't seem to work on tf 1.10
        # if args.xla:
        #     config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        if self.mode == 'NER':
            return Estimator(model_fn=ner_model_fn, config=RunConfig(session_config=config))

    def predict(self, content=''):
        # 预测函数
        self.content.clear()
        self.content.append(content)
        result = next(self.predictions)
        if self.mode == 'NER':
            pred_label_result, pred_ids_result = ner_result_to_json(result['encodes'], self.id2label)
            print('pred_label_result:', pred_label_result)
            print('pred_ids_result:', pred_ids_result)
            # lg.info('job done\tsize: %s\t' % (r['encodes'].shape))
            return self.get_entity_from_source(pred_label_result=pred_label_result[0],content=content)

    def get_entity_from_source(self,pred_label_result=[],content=''):
        # 获取具体实体
        w_list = self.tokenizer.tokenize(content)
        begin_word = -1
        pos = ''
        entity_dict = {}
        for index,value in enumerate(pred_label_result):
            if value[0] == 'B' and begin_word == -1:
                begin_word = index
                pos = value
            elif begin_word != -1 and (value[0] != 'I'or index == len(pred_label_result)-1):
                word = ''.join(w_list[begin_word:index])
                if index == len(pred_label_result)-1:
                    word = ''.join(w_list[begin_word:])
                entity_dict[word] = pos
                begin_word = -1
                pos = ''
                if value[0] == 'B':
                    begin_word = index
                    pos = value
                continue
        return entity_dict

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

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = convert_to_unicode(line[1])
            label = convert_to_unicode(line[0])
            # if i == 0:
            #     print('label: ', label)
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid=None, text=None, label=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        # self.label_mask = label_mask


def filed_based_convert_examples_to_features(
        examples, max_seq_length, tokenizer, output_dir, mode=None,tf=None):
    """
    将数据转化为TF_Record 结构，作为模型数据输入
    :param examples:  样本
    :param label_list:标签list
    :param max_seq_length: 预先设定的最大序列长度
    :param tokenizer: tokenizer 对象
    :param output_file: tf.record 输出路径
    :param mode:
    :return:
    """
    # 遍历训练数据
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            lg.info("Writing example %d of %d" % (ex_index, len(examples)))
        # 对于每一个训练样本,
        feature = convert_single_example(ex_index, example, max_seq_length, tokenizer, output_dir, mode)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        # features["label_mask"] = create_int_feature(feature.label_mask)
        # tf.train.Example/Feature 是一种协议，方便序列化？？？
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        return tf_example


def convert_single_example(ex_index, example, max_seq_length, tokenizer, output_dir, mode):
    """
    将一个样本进行分析，然后将字转化为id, 标签转化为id,然后结构化到InputFeatures对象中
    :param ex_index: index
    :param example: 一个样本
    :param label_list: 标签列表
    :param max_seq_length:
    :param tokenizer:
    :param output_dir
    :param mode:
    :return:
    """
    label_map = {}
    with codecs.open(os.path.join(output_dir, 'label2id.pkl'), 'rb') as w:
        label_map = pickle.load(w)

    textlist = example.text.split(' ')
    labellist = example.label.split(' ')
    tokens = []
    labels = []
    for i, word in enumerate(textlist):
        # 分词，如果是中文，就是分字,但是对于一些不在BERT的vocab.txt中得字符会被进行WordPice处理（例如中文的引号），可以将所有的分字操作替换为list(input)
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_1 = labellist[i]
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:  # 一般不会出现else
                labels.append("X")
    # tokens = tokenizer.tokenize(example.text)
    # 序列截断
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
        labels = labels[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")  # 句子开始设置CLS 标志
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(label_map["[CLS]"])  # O OR CLS 没有任何影响，不过我觉得O 会减少标签个数,不过拒收和句尾使用不同的标志来标注，使用LCS 也没毛病
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    ntokens.append("[SEP]")  # 句尾添加[SEP] 标志
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)  # 将序列中的字(ntokens)转化为ID形式
    input_mask = [1] * len(input_ids)
    # label_mask = [1] * len(input_ids)
    # padding, 使用
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")
        # label_mask.append(0)
    # print(len(input_ids))
    assert len(input_ids) == max_seq_length  #vocablist中的 字的索引，不够max length的用0补齐
    assert len(input_mask) == max_seq_length #有字符的都是用1 ，不够max length的用0补齐
    assert len(segment_ids) == max_seq_length #第一句全部为0，不够max length的用0补齐，
    assert len(label_ids) == max_seq_length #labellist中标签的index，不够max length的用0补齐，
    # assert len(label_mask) == max_seq_length

    # 打印部分样本数据信息
    if ex_index < 5:
        lg.info("*** Example ***")
        lg.info("guid: %s" % (example.guid))
        lg.info("tokens: %s" % " ".join(
            [printable_text(x) for x in tokens]))
        lg.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        lg.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        lg.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        lg.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        # logger.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))

    # 结构化为一个类
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        # label_mask = label_mask
    )
    return feature


def file_based_input_fn_builder(tf_example, seq_length, drop_remainder,tf=None):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        # "label_ids":tf.VarLenFeature(tf.int64),
        # "label_mask": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(tf_example)
        d = d.apply(tf.data.experimental.map_and_batch(lambda record: _decode_record(record, name_to_features),
                                                       batch_size=batch_size,
                                                       num_parallel_calls=8,  # 并行处理数据的CPU核心数量，不要大于你机器的核心数
                                                       drop_remainder=drop_remainder))
        d = d.prefetch(buffer_size=4)
        return d

    return input_fn
