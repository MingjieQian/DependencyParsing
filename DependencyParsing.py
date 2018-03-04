import os
import re
import sys
import math
import nltk
import time
import pickle
import random
import shutil
import argparse
import itertools
import numpy as np
from enum import Enum
from gensim.models import word2vec
from collections import OrderedDict, deque
from nltk.tag.senna import SennaTagger
from nltk.tag.perceptron import PerceptronTagger

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.platform import tf_logging as logging

ver = tf.__version__.split('.')
if ver[0] == '0':
    from tensorflow.python.ops.rnn_cell import GRUCell
    from tensorflow.python.ops.rnn_cell import LSTMCell
else:
    from tensorflow.contrib.rnn import GRUCell
    from tensorflow.contrib.rnn import LSTMCell

from utilities import TextDataset, fetch_minibatches, get_data_input, get_glove_vocab, get_logger
from utilities import is_number, evaluate_memory_efficient, Word, build_feature_vector, pad_sequences

__author__ = "Mingjie Qian"
__date__ = "February 25th, 2018"

ROOT = "ROOT"
PAD = "PAD"
NUL = "NUL"
NUM = "NUM"
UNK = "UNK"
STT = "^start"
END = "end$"

tag_open_pattern = re.compile(r'<([A-Z0-9]+)\b[^>]*>(.+)')

# encoding = "latin-1"
encoding = "utf-8"
checkpoint_name = "model.ckpt"  # Checkpoint filename


def build_char_dictionary():
    char_dict = {PAD: 0}
    # chars = 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’’’/\|_@#$%ˆ&*~‘+-=<>()[]{}'
    chars = 'ROTNULabcdefghijklmnopqrstuvwxyz0123456789,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{}'
    for char in chars:
        char_dict[char] = len(char_dict)
    char_dict[UNK] = len(char_dict)
    return char_dict


def build_pos_dictionary():
    pos_dict = {}
    pos_tags = [NUL, '$', "''", '(', ')', ',', '--', '.', ':', 'CC', 'CD',
                'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN',
                'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR',
                'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP',
                'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '``']
    for pos_tag in pos_tags:
        pos_dict[pos_tag] = len(pos_dict)
    return pos_dict


def build_tag_dictionary(tag_type_list):
    # tag_dict = {shift: 0, l-root: 1, r-root: 2}
    tag_dict = {'shift': 0}
    rel_dict = {'null': 0}
    for tag_type in tag_type_list:
        # L[tag_type] = 'l-' + tag_type
        # R[tag_type] = 'r-' + tag_type
        tag_dict['l-' + tag_type] = len(tag_dict)
        tag_dict['r-' + tag_type] = len(tag_dict)
        rel_dict[tag_type] = len(rel_dict)
    # tag_dict['null'] = len(tag_dict)
    return tag_dict, rel_dict


def save_tag_type_list(model_dir, tag_type_list):
    filepath = os.path.join(model_dir, 'tag_type_list.txt')
    with open(filepath, 'w', encoding=encoding) as f:
        for tag_type in tag_type_list:
            f.write("%s\n" % tag_type)
    print("Tag types were saved in %s" % filepath)


def load_tag_type_list(model_dir):
    filepath = os.path.join(model_dir, 'tag_type_list.txt')
    tag_type_list = []
    with open(filepath, 'r', encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tag_type_list.append(line)
    return tag_type_list


def build_class_id_weight_dict(tag_dict, class_weights):
    class_id_weight_dict = [1.0] * len(class_weights)
    for (tag, id) in tag_dict.items():
        if tag in class_weights:
            class_id_weight_dict[id] = class_weights[tag]
    return class_id_weight_dict


def save_dictionary(model_dir, vocab):
    dict_path = os.path.join(model_dir, "vocab.txt")
    with open(dict_path, 'w', encoding=encoding) as f:
        for (word, index) in vocab.items():
            f.write("%s\t%d\n" % (word, index))
    print("Dictionary size: %s" % len(vocab))
    print("Dictionary file was saved in %s" % dict_path)


def load_dictionary(model_dir):
    dict_path = os.path.join(model_dir, "vocab.txt")
    vocab = {}
    with open(dict_path, 'r', encoding=encoding) as f:
        for line in f:
            line = line.strip()
            # print line
            if not line:
                continue
            entry = line.split("\t")
            vocab[entry[0]] = int(entry[1])
    return vocab


def save_class_weights(model_dir, class_weights):
    filepath = os.path.join(model_dir, 'class_weights.txt')
    with open(filepath, 'w', encoding=encoding) as f:
        for (tag, weight) in class_weights.items():
            f.write("%s\t%f\n" % (tag, weight))
    print("Class weights file was saved in %s" % filepath)


def save_TF_version(model_dir):
    filepath = os.path.join(model_dir, 'tf_version.txt')
    with open(filepath, 'w', encoding=encoding) as f:
        f.write(tf.__version__ + '\n')


def save_max_word_length(max_word_length):
    filepath = os.path.join(model_dir, 'max_word_length.txt')
    with open(filepath, 'w', encoding=encoding) as f:
        # f.write(max_word_length + '\n')
        # TypeError: unsupported operand type(s) for +: 'int' and 'str'
        f.write(str(max_word_length) + '\n')


def load_max_word_length():
    global max_word_length
    filepath = os.path.join(model_dir, 'max_word_length.txt')
    if not os.path.exists(filepath):
        return
    with open(filepath, 'r', encoding=encoding) as f:
        max_word_length = int(f.readline().strip())


def extract_word_xml(token, update_tags=False, tag_types=None):
    # if token.startswith('<'):
    match_o = tag_open_pattern.match(token)
    if match_o:
        tag_type = match_o.group(1)
        token = str(match_o.group(2))
        if update_tags:
            if tag_type not in tag_types:
                tag_types.add(tag_type)
    if token.endswith('>'):
        # match_c = tag_end_pattern.match(token)
        # if match_c:
        #     token = str(match_c.group(1))
        idx = token.rfind("</")
        if idx != -1:
            token = token[:idx]
    return token


def extract_word_bracket(token, update_tags=False, tag_types=None):
    if token.startswith('[') and len(token) > 1:
        tag_type = token[1:]
        token = ''
        if update_tags:
            if tag_type not in tag_types:
                tag_types.add(tag_type)
    elif token.endswith(']') and len(token) > 1:
        token = token[:-1]
    return token


def compute_counts(data_dir, training_filename, tag_types, counts=None, last_dataset=True):
    global max_word_length
    print('Counting frequency for each term...')
    train_path = os.path.join(data_dir, training_filename)
    if counts is None:
        counts = {NUM: 0}
    if annotation_scheme == 'CoNLL':
        cnt = 0
        with open(train_path, 'r', encoding=encoding) as f:
            split_pattern = re.compile(r'[ \t]')
            wrd_seq = []
            for line in f:
                line = line.strip()
                if not line or line.startswith('-DOCSTART-'):
                    if wrd_seq:
                        cnt += 1
                        if cnt % 100 == 0:
                            print("  read %d tagged examples" % cnt, end="\r")
                        wrd_seq = []
                    continue
                elif line.startswith('#'):
                    continue

                # container = line.split(' \t')
                container = split_pattern.split(line)
                if len(container) < 8:
                    print(line)
                token, rel = container[1], container[7]
                token = token.lower()
                if not token:
                    continue
                # if use_all_chars:
                max_word_length = max(max_word_length, len(token))
                if token in counts:
                    counts[token] += 1
                else:
                    if is_number(token):
                        counts[NUM] += 1
                    else:
                        counts[token] = 1
                tag_type = rel
                if tag_type and tag_type not in tag_types:
                    tag_types.add(tag_type)
                wrd_seq.append(token)
            print("  read %d tagged examples" % cnt)
    else:
        with open(train_path, "r", encoding=encoding) as f:
            cnt = 0
            for tagged_query in f:
                tagged_query = tagged_query.strip()
                if not tagged_query:
                    continue
                for token in tagged_query.split():
                    token = extract_word(token, True, tag_types)
                    if not token:
                        continue
                    max_word_length = max(max_word_length, len(token))
                    if token in counts:
                        counts[token] += 1
                    else:
                        if is_number(token):
                            counts[NUM] += 1
                        else:
                            counts[token] = 1
                cnt += 1
                if cnt % 100 == 0:
                    print("  read %d tagged examples" % cnt, end="\r")
            print("  read %d tagged examples" % cnt)
    print("counts['NUM']: %d" % counts[NUM])

    if last_dataset:
        del counts[NUM]
        counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return counts


def save_count_file(model_dir, counts):
    counts_path = os.path.join(model_dir, "counts.txt")
    with open(counts_path, 'w', encoding=encoding) as f:
        for (word, count) in counts:
            f.write("%s\t%d\n" % (word, count))
    print("Term frequency file was saved in %s" % counts_path)


def build_sequence_from_a_word_sequence_in_char_level(wrd_seq, char_dict, max_word_length):
    """
    wrd_seq must be generated from a lowercase well-formed clean query.

    :param wrd_seq:
    :param char_dict:
    :return: character id matrix with size [len(wrd_seq), max_word_length]
    """
    char_id_matrix = []
    for w in wrd_seq:
        char_id_seq = []
        i = 0
        while i < len(w) and i < max_word_length:
            char_id_seq.append(char_dict[w[i] if w[i] in char_dict else 'UNK'])
            i += 1
        while i < max_word_length:
            char_id_seq.append(char_dict[PAD])
            i += 1
        char_id_matrix.append(char_id_seq)
    return char_id_matrix


def export_glove_vectors(glove_filename, output_filename):
    """Saves glove vectors in numpy array

    Args:
        vocab: dictionary vocab[word] = index
        glove_filename: a path to a glove file
        trimmed_filename: a path where to store a matrix in npy
        dim: (int) dimension of embeddings

    """
    embeddings = {}
    # embeddings = np.zeros([len(vocab), dim])
    with open(glove_filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            embeddings[word] = np.asarray(embedding)
    # np.savez_compressed(output_filename, embeddings=embeddings)
    pickle_path = output_filename
    f = open(pickle_path, 'wb')
    # pickle.dump(pos_id_seq_map, f)
    pickle.dump(embeddings, f)
    f.close()


def get_glove_vectors(filename):
    """
    Args:
        filename: path to the npz file

    Returns:
        embeddings dictionary

    """
    pickle_path = filename
    with open(pickle_path, 'rb') as handle:
        # pos_id_seq_map = pickle.load(handle)
        embeddings = pickle.load(handle)
    return embeddings


def length(sequence, zero_padded=False):
    used = tf.sign(sequence + (1 if not zero_padded else 0))
    res = tf.reduce_sum(used, axis=1)
    res = tf.cast(res, tf.int32)
    return res


def length_with_padding(sequence):
    """
    Padding index must be zero.

    :param sequence:
    :return:
    """
    used = tf.sign(sequence)
    res = tf.reduce_sum(used, axis=2)
    res = tf.cast(res, tf.int32)
    return res


def create_model(model_dir,
                 chr_embedding_classes,
                 arc_embedding_classes,
                 pos_embedding_classes,
                 wrd_embedding_classes,
                 embedding_size,
                 hidden_size,
                 tag_dict,
                 vocab=None,
                 ):
    global initialized_by_pretrained_embedding
    global pretrained_embedding_path
    global train_word_embedding
    global zero_padded

    sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.

    arc_initializer = init_ops.random_uniform_initializer(-sqrt3, sqrt3, seed=1)
    arc_embedding = tf.get_variable("arc_embedding",
                                    [arc_embedding_classes, embedding_size],
                                    trainable=True,
                                    initializer=arc_initializer)

    pos_initializer = init_ops.random_uniform_initializer(-sqrt3, sqrt3, seed=2)
    pos_embedding = tf.get_variable("pos_embedding",
                                    [pos_embedding_classes, embedding_size],
                                    trainable=True,
                                    initializer=pos_initializer)

    if initialized_by_pretrained_embedding:
        if not os.path.exists(pretrained_embedding_path):
            print("Pre-trained embedding path %s doesn't exist." % pretrained_embedding_path, file=sys.stderr)
            shutil.rmtree(model_dir)
            exit()

        if pretrained_embedding_path.find('glove') == -1:
            pretrained_word_embedding = [[] for i in range(wrd_embedding_classes)]
            model = word2vec.KeyedVectors.load_word2vec_format(pretrained_embedding_path, binary=True)
            print("Pre-trained embedding size is %d." % model.syn0.shape[1])
            print("Specified embedding size is %d." % embedding_size)
            if embedding_size != model.syn0.shape[1]:
                print("Specified embedding size (%d) doesn't match pre-trained embedding size (%d)."
                      % (embedding_size, model.syn0.shape[1]), file=sys.stderr)
                print("Please make sure the pre-trained embedding dimensionality matches the specified embedding size.")
                shutil.rmtree(model_dir)
                exit()

            cnt = 0
            for (word, index) in vocab.items():
                if word in model.vocab:
                    pretrained_word_embedding[index] = model.word_vec(word)
                    cnt += 1
                else:
                    pretrained_word_embedding[index] = np.random.uniform(-sqrt3, sqrt3, (embedding_size,))
                    # pretrained_word_embedding[index] = np.zeros((embedding_size,), dtype=np.float32)
            pretrained_word_embedding = np.array(pretrained_word_embedding)
        else:
            pretrained_word_embedding = [[] for i in range(wrd_embedding_classes)]
            # trimmed_filepath = pretrained_embedding_path + '.trimmed.npz'
            glove_embedding_filepath = pretrained_embedding_path + '.npz'
            # embeddings = get_glove_vectors(glove_embedding_filepath)
            # embeddings = {}
            cnt = 0
            processed_words = set()
            with open(pretrained_embedding_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip().split(' ')
                    word = line[0]
                    embedding = [float(x) for x in line[1:]]
                    if word in vocab:
                        pretrained_word_embedding[vocab[word]] = np.asarray(embedding)
                        processed_words.add(word)
                        cnt += 1
            for (word, index) in vocab.items():
                if word not in processed_words:
                    pretrained_word_embedding[index] = np.zeros((embedding_size,), dtype=np.float32)
            pretrained_word_embedding = np.array(pretrained_word_embedding)
        print('wrd_embedding_classes:', wrd_embedding_classes)
        print('len(vocab):', len(vocab))
        print('Number of word existing in pretrained embedding vocab:', cnt)
        word_initializer = init_ops.constant_initializer(pretrained_word_embedding)
    else:
        # Default initializer for embeddings should have variance=1.
        sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
        word_initializer = init_ops.random_uniform_initializer(-sqrt3, sqrt3, seed=0)
    wrd_embedding = tf.get_variable("wrd_embedding",
                                    [wrd_embedding_classes, embedding_size],
                                    trainable=train_word_embedding,
                                    initializer=word_initializer)

    arc_id_arrs = tf.placeholder(tf.int32, shape=[None, None], name="arc_id_arrs")
    pos_id_arrs = tf.placeholder(tf.int32, shape=[None, None], name="pos_id_arrs")
    wrd_id_arrs = tf.placeholder(tf.int32, shape=[None, None], name="wrd_id_arrs")

    arc_embedded = embedding_ops.embedding_lookup(arc_embedding, arc_id_arrs)
    pos_embedded = embedding_ops.embedding_lookup(pos_embedding, pos_id_arrs)
    wrd_embedded = embedding_ops.embedding_lookup(wrd_embedding, wrd_id_arrs)

    # shape = (batch size, length of word array, max length of word)
    chr_id_mats = tf.placeholder(tf.int32, shape=[None, None, None], name="chr_id_mats")

    ver = tf.__version__.split('.')
    with tf.variable_scope("chars"):
        if use_chars:
            # get char embeddings matrix
            _char_embeddings = tf.get_variable(
                name="_char_embeddings",
                dtype=tf.float32,
                shape=[chr_embedding_classes, embedding_size_char])
            char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                                                     chr_id_mats, name="char_embeddings")

            # put the time dimension on axis=1
            s = tf.shape(char_embeddings)
            char_embeddings = tf.reshape(char_embeddings,
                                         shape=[s[0] * s[1], s[-2], embedding_size_char])
            word_lengths = length_with_padding(chr_id_mats)
            word_lengths = tf.reshape(word_lengths, shape=[s[0] * s[1]])

            # bi lstm on chars
            cell_fw = LSTMCell(hidden_size_char, state_is_tuple=True)
            cell_bw = LSTMCell(hidden_size_char, state_is_tuple=True)
            _output = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, char_embeddings,
                sequence_length=word_lengths, dtype=tf.float32)

            # read and concat output
            _, ((_, output_fw), (_, output_bw)) = _output
            print('output_fw:', output_fw)
            print('output_bw:', output_bw)
            # For TF version 0.12, concat_dim doesn't support -1. If -1 is used, output will have shape [?, 200, ?, 100]
            if ver[0] == '0':
                output = tf.concat(1, [output_fw, output_bw])
            else:
                output = tf.concat([output_fw, output_bw], 1)
            print('output:', output)

            # shape = (batch size, sentence length, 2*hidden_size_char)
            output = tf.reshape(output, shape=[s[0], s[1], 2 * hidden_size_char])
            print('output:', output)
            if ver[0] == '0':
                wrd_embedded = tf.concat(2, [wrd_embedded, output])
            else:
                wrd_embedded = tf.concat([wrd_embedded, output], 2)
            print('wrd_embedded:', wrd_embedded)

    if use_chars:
        wrd_embedded = tf.reshape(wrd_embedded,
                                  [-1, 18 * (embedding_size + 2 * hidden_size_char)],
                                  name='wrd_embedded'
                                  )
    else:
        wrd_embedded = tf.reshape(wrd_embedded,
                                  [-1, 18 * (embedding_size)],
                                  name='wrd_embedded'
                                  )
    pos_embedded = tf.reshape(pos_embedded,
                              [-1, 18 * (embedding_size)],
                              name='pos_embedded'
                              )
    arc_embedded = tf.reshape(arc_embedded,
                              [-1, 12 * (embedding_size)],
                              name='arc_embedded'
                              )
    if ver[0] == '0':
        embedded_concat = tf.concat(1,
                                    [wrd_embedded, pos_embedded, arc_embedded],
                                    "embedded_concat")
    else:
        embedded_concat = tf.concat([wrd_embedded, pos_embedded, arc_embedded],
                                    1,
                                    "embedded_concat")
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    embedded_dropout = tf.nn.dropout(embedded_concat, keep_prob, name='embedded_dropout')

    instance_length = embedded_concat.get_shape()[1].value
    if debug:
        print("instance_length: %d" % instance_length)
    weights_hidden = tf.Variable(tf.truncated_normal([instance_length, hidden_size], seed=1), name='weights_hidden')
    biases_hidden = tf.Variable(tf.truncated_normal([hidden_size], seed=1), name='biases_hidden')

    # activation_hidden = tf.matmul(embedded_reshape, weights_hidden) + biases_hidden
    activation_hidden = nn_ops.xw_plus_b(embedded_dropout, weights_hidden, biases_hidden, name='activation_hidden')
    # Polynomial kernel (cubic)
    # r0.7085-p0.8370-f0.7658-t110.0702s
    # Polynomial kernel (cubic) + ReLU
    # r0.7216-p0.8415-f0.7749-t99.5459s
    # Polynomial kernel (quadratic)
    # r0.7582-p0.8446-f0.7988-t97.6890s
    # outputs_hidden = activation_hidden * activation_hidden
    # tanh kernel
    # r0.7700-p0.8408-f0.8036-t71.4103s
    outputs_hidden = tf.nn.tanh(activation_hidden)
    outputs_hidden_dropout = tf.nn.dropout(outputs_hidden, keep_prob)

    w_fc = tf.Variable(tf.truncated_normal([hidden_size, len(tag_dict)], seed=1), name='w_fc')
    b_fc = tf.Variable(tf.truncated_normal([len(tag_dict)], seed=1), name='b_fc')

    logits = nn_ops.xw_plus_b(outputs_hidden_dropout, w_fc, b_fc, name='logits')
    # tf.argmax output is int64 which will generate a TypeError:
    # Input 'y' of 'Equal' Op has type int32 that does not match type int64 of argument 'x'
    # for tf.equal()

    # outputs_int64: Tensor("outputs:0", shape=(?,), dtype=int64)
    outputs_int64 = tf.argmax(logits, 1)
    # outputs: Tensor("Cast:0", shape=(?,), dtype=int32)
    outputs = math_ops.cast(outputs_int64, tf.int32, name='outputs')
    label_id_arr = tf.placeholder(tf.int32, shape=[None], name="label_id_arr")
    if debug:
        print(outputs)
        print(label_id_arr)
        print(tf.equal(outputs, label_id_arr))

    correct_prediction = tf.reduce_sum(tf.cast(tf.equal(outputs, label_id_arr), tf.int32), name='correct_prediction')
    cross_entropy = nn_ops.sparse_softmax_cross_entropy_with_logits(logits, label_id_arr)
    weights = tf.placeholder(tf.float32, [None], 'weights')
    batch_loss = cross_entropy * weights
    # batch_loss /= tf.reduce_sum(weights)
    # To do. weights.shape()[0] will return a tensor.
    # cost = batch_loss / math_ops.cast(array_ops.shape(weights)[0], tf.float32)
    # Using cost tensor of shape [None] gives r0.9322-p0.9388-f0.9355 for answer concept recognition.
    # Using cost scalar tensor gives r0.9307-p0.9392-f0.9349
    # It's a mystery that a cost tensor of shape [None] can still be applicable.
    # It seems that if the input tensor is not a scalar tensor, TF will sum up all entries first.
    # cost = batch_loss
    loss = tf.reduce_mean(batch_loss, name='loss')
    # weights.get_shape()[0].value would be None and
    # math_ops.cast(weights.get_shape()[0].value, tf.float32) will get an error.
    # cost = batch_loss / math_ops.cast(weights.get_shape()[0].value, tf.float32)

    # Gradients and SGD update operation for training the model.
    lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
    params = tf.trainable_variables()
    opt = tf.train.AdamOptimizer(lr)
    gradients = tf.gradients(loss, params)
    if max_gradient_norm > 0:
        clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
        updates = opt.apply_gradients(zip(clipped_gradients, params), name='updates')
    else:
        updates = opt.apply_gradients(zip(gradients, params), name='updates')


def f_measure(beta, precision, recall):
    if precision == 0.0 or recall == 0.0:
        return 0.0
    else:
        return (1 + beta * beta) * precision * recall / (beta * beta * precision + recall)


class CustomException(Exception):
    def __init__(self, value):
        self.parameter = value

    def __str__(self):
        return repr(self.parameter)


def softmax(z):
    # assert len(z.shape) == 2
    if len(z.shape) == 1:
        z = np.expand_dims(z, 0)
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]  # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]
    return e_x / div


################################################################################
# Training                                                                     #
################################################################################
def train_memory_efficient():
    begin = time.time()
    # Cannot use global filename as filename is used both as a parameter and a global
    # Here filename's value only comes from the function input argument filename rather
    # than the global parameter filename.
    print("Training on %s" % os.path.join(data_dir, training_filename))

    if use_chars:
        char_dict = build_char_dictionary()
    else:
        char_dict = {}

    # pickle_path = os.path.join(model_dir, 'training_data')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        tag_types = set()
        if chopoff >= 1:
            # Build and chop rare terms off vocabulary
            counts = compute_counts(data_dir, training_filename, tag_types, None,
                                    False if validation_filename else True)
            if validation_filename:
                counts = compute_counts(data_dir, validation_filename, tag_types, counts, True)
            save_count_file(model_dir, counts)
            vocab = {NUL: 0, ROOT: 1, UNK: 2, NUM: 3}
            glove_vocab = None
            if pretrained_embedding_path and pretrained_embedding_path.find('glove') != -1:
                glove_vocab = get_glove_vocab(pretrained_embedding_path)
            if extend_sequence:
                vocab[STT] = len(vocab)
                vocab[END] = len(vocab)
            for (token, count) in [pair for pair in counts if pair[1] >= chopoff]:
                if glove_vocab and token not in glove_vocab:
                    continue
                vocab[token] = len(vocab)
            update_vocab = False
            tag_type_list = sorted(tag_types)
            tag_dict, rel_dict = build_tag_dictionary(tag_type_list)
        else:
            vocab = {NUL: 0, ROOT: 1, UNK: 2, NUM: 3}
            if extend_sequence:
                vocab[STT] = len(vocab)
                vocab[END] = len(vocab)
            update_vocab = True
            tag_type_list = []
            tag_dict = {}
            rel_dict = {'null': 0}
            # TODO
            # If tag_dict is empty, both B and I would be empty, then parsing training data will be problematic.
        print('max_word_length:', max_word_length)
        save_max_word_length(max_word_length)
        print('Vocabulary size: %s' % len(vocab))
        pos_dict = build_pos_dictionary()
        save_dictionary(model_dir, vocab)
        save_tag_type_list(model_dir, tag_type_list)
    else:
        # Load existing formatted training data and dictionaries
        tag_type_list = load_tag_type_list(model_dir)
        tag_dict, rel_dict = build_tag_dictionary(tag_type_list)
        pos_dict = build_pos_dictionary()
        vocab = load_dictionary(model_dir)

    path_log = os.path.join(model_dir, "log.txt")
    logger = get_logger(path_log)

    tags_map = [""] * len(tag_dict)
    for (tag, index) in tag_dict.items():
        tags_map[index] = tag

    # label_counts = [0. for i in range(len(tag_dict) - 1)]
    # for T, label_seqs in label_seq_map.items():
    #     for label_seq in label_seqs:
    #         for label in label_seq:
    #             label_counts[label] += 1.
    # class_weights = {tags_map[tag_id]: label_counts[tag_dict[O]] / label_counts[tag_id] if label_counts[tag_id] > 0. else 0.
    #                  for tag_id in range(len(tag_dict) - 1) if tag_id != tag_dict[N]}
    class_weights = {tags_map[tag_id]: 1.0 for tag_id in range(len(tag_dict))}
    print('class_weights:', class_weights)
    class_id_weight_dict = build_class_id_weight_dict(tag_dict, class_weights)
    save_class_weights(model_dir, class_weights)
    load_validation_data = not True
    if validation_filename:
        validation_filepath = os.path.join(data_dir, validation_filename)
        print("Validation on %s" % validation_filepath)

    ################################################################################
    # Start session
    ################################################################################

    sess = tf.Session()

    rel_embedding_classes = len(rel_dict)
    chr_embedding_classes = len(char_dict)
    pos_embedding_classes = len(pos_dict)
    wrd_embedding_classes = len(vocab)
    create_model(model_dir,
                 chr_embedding_classes,
                 rel_embedding_classes,
                 pos_embedding_classes,
                 wrd_embedding_classes,
                 embedding_size,
                 hidden_size,
                 tag_dict,
                 vocab,
                 )
    # for variable in tf.global_variables():
    #     print(variable)

    # Training loop
    counter = 0
    start = time.time()
    saver = tf.train.Saver(tf.global_variables())
    if resume and tf.train.get_checkpoint_state(model_dir):
        ckpt = tf.train.get_checkpoint_state(model_dir)
        model_checkpoint_path = ckpt.model_checkpoint_path
        if not os.path.exists(model_checkpoint_path):
            model_checkpoint_path = os.path.join(model_dir, os.path.basename(model_checkpoint_path))
        saver.restore(sess, model_checkpoint_path)
        model_checkpoint_name = os.path.basename(model_checkpoint_path)
        epoch = int(model_checkpoint_name[model_checkpoint_name.rindex('-') + 1:])
    else:
        # tf.add_to_collection('outputs', outputs)
        sess.run(tf.global_variables_initializer())
        epoch = 0
        # Use a saver_def to get the "magic" strings to restore
        # saver_def = saver.as_saver_def()
        # print(saver_def.filename_tensor_name)
        # print(saver_def.restore_op_name)
        # tf.train.write_graph(sess.graph_def, '.', os.path.join(model_dir, 'model.proto'), as_text=False)
        # tf.train.write_graph(sess.graph_def, '.', os.path.join(model_dir, 'model.txt'), as_text=True)

    train_path = os.path.join(data_dir, training_filename)
    data_train = TextDataset(train_path, annotation_scheme, extend_sequence)
    if validation_filename and not load_validation_data:
        validation_filepath = os.path.join(data_dir, validation_filename)
        data_dev = TextDataset(validation_filepath, annotation_scheme, extend_sequence)

    data_size = len(data_train)
    print("data_size: %s" % data_size)

    recall_best = {tag_type: 0.0 for tag_type in tag_type_list}
    precision_best = {tag_type: 0.0 for tag_type in tag_type_list}
    F1_best = {tag_type: 0.0 for tag_type in tag_type_list}
    F_all_best = 0.0
    epoch_best = 0
    loss_best = 0.0
    accuracy_best = 0.0
    acc_dev_best = 0.0
    metric_best = 0
    LAS_best = 0
    UAS_best = 0
    LS_best = 0
    lr = learning_rate
    save_TF_version(model_dir)
    while True:

        accuracy = 0.0
        total_loss = 0.0
        num_token = 0

        # One epoch
        nbatches = (len(data_train) + batch_size - 1) // batch_size

        # iterate over dataset
        processed = 0
        for i, (words_batch, pos_seq_batch, heads_batch, rel_seq_batch) in enumerate(fetch_minibatches(data_train, batch_size)):
            num_token += sum(map(lambda words: len(words) - (2 if extend_sequence else 0), words_batch))
            wrd_id_arrs, pos_id_arrs, arc_id_arrs, chr_id_mats, label_id_arr = \
                get_data_input(words_batch, pos_seq_batch, heads_batch, rel_seq_batch,
                               vocab, pos_dict, tag_dict, rel_dict, use_chars, char_dict)
            # Set the feed dictionary
            feed_dict = {  # 'input_tag_ids:0': tag_id_seqs[data_index:end_index],
                # 'input_pos_ids:0': pos_id_seqs[data_index:end_index],
                'wrd_id_arrs:0': wrd_id_arrs,
                'pos_id_arrs:0': pos_id_arrs,
                'arc_id_arrs:0': arc_id_arrs,
                # 'chr_id_mats:0': chr_id_mats,
                'label_id_arr:0': label_id_arr,
                'weights:0': [class_id_weight_dict[label_id] for label_id in label_id_arr],
                'keep_prob:0': keep_prob,
                'lr:0': lr}
            if use_chars:
                feed_dict['chr_id_mats:0'] = chr_id_mats
            # print(feed_dict)
            _, loss, correct_hits = sess.run(['updates',
                                              'loss:0',
                                              'correct_prediction:0',
                                              ],
                                             feed_dict=feed_dict)
            print("  trained [%d, %d] of %d - batch loss %.4f" % (processed,
                                                                  processed + len(words_batch),
                                                                  data_size, loss), end="\r")
            accuracy += correct_hits
            loss *= len(label_id_arr)
            total_loss += loss
            counter += 1
            processed += len(words_batch)
        epoch += 1
        # print("Total training loss: %f, epoch: %d" % (total_loss, epoch))
        accuracy /= len(wrd_id_arrs)
        if epoch % 200 == 0:
            lr *= decay
        # print("Training accuracy: %f, epoch: %d" % (accuracy, epoch))
        if validation_filename and epoch % 10 == 0:
            metrics = evaluate_memory_efficient(sess, data_dev, vocab, pos_dict, tag_dict, tags_map, rel_dict, batch_size, use_chars, char_dict)
            msg = "Epoch %d - training loss: %.4f acc: %.4f - val" % (epoch, total_loss, accuracy)
            print(msg, end='')
            LAS = metrics['LAS']
            UAS = metrics['UAS']
            LS = metrics['LS']
            metric = 0.5 * LAS + 0.25 * UAS + 0.25 * LS
            # for tag_type in tag_type_list:
            #     print(" %s: r%.4f p%.4f f%.4f" % (tag_type, R[tag_type], P[tag_type], F[tag_type]), end='')
            print(" LAS: %.2f - UAS: %.2f - LS: %.2f" % (100 * LAS, 100 * UAS, 100 * LS), end='')
            logger.info(msg + " LAS: %.2f - UAS: %.2f - LS: %.2f" % (100 * LAS, 100 * UAS, 100 * LS))
            print()
            # F1_avg = sum(F.values()) / len(F)
            if metric > metric_best:
                epoch_best = epoch
                loss_best = total_loss
                accuracy_best = accuracy
                metric_best = metric
                LAS_best = LAS
                UAS_best = UAS
                LS_best = LS
                # recall_best = R
                # precision_best = P
                # F1_best = F
                # F_all_best = F_all
                # acc_dev_best = acc
                now = time.time()
                delta = now - start
                print("Counter: %d, delta time = %f" % (counter, delta))
                logger.info("Counter: %d, delta time = %f" % (counter, delta))
                start = time.time()
                checkpoint_path = os.path.join(model_dir, checkpoint_name)
                saver.save(sess, checkpoint_path, global_step=epoch)
        else:
            print("Epoch %d\t- training loss: %.4f\taccuracy: %.4f" % (epoch, total_loss, accuracy))
            if epoch % 1000 == 0:
                now = time.time()
                delta = now - start
                print("Counter: %d, delta time = %f" % (counter, delta))
                start = time.time()
                checkpoint_path = os.path.join(model_dir, checkpoint_name)
                saver.save(sess, checkpoint_path, global_step=epoch)

        if acc_stop and accuracy >= acc_stop:
            if not validation_filename:
                now = time.time()
                delta = now - start
                print("Counter: %d, delta time = %f" % (counter, delta))
                start = time.time()
                checkpoint_path = os.path.join(model_dir, checkpoint_name)
                saver.save(sess, checkpoint_path, global_step=epoch)
            break
        elif 0 <= max_iter <= epoch:
            if not validation_filename:
                now = time.time()
                delta = now - start
                print("Counter: %d, delta time = %f" % (counter, delta))
                start = time.time()
                checkpoint_path = os.path.join(model_dir, checkpoint_name)
                saver.save(sess, checkpoint_path, global_step=epoch)
            break
    elapsed_time = time.time() - begin
    if validation_filename:
        print("Best model - epoch %d - loss %.4f - acc %.4f - val" % (epoch_best, loss_best, accuracy_best), end='')
        logger.info("Best model - epoch %d - loss %.4f - acc %.4f - val" % (epoch_best, loss_best, accuracy_best))
        # for tag_type in tag_type_list:
        #     print(" %s: r%.4f p%.4f f%.4f" % (
        #     tag_type, recall_best[tag_type], precision_best[tag_type], F1_best[tag_type]), end='')
        #     logger.info(" %s: r%.4f p%.4f f%.4f" % (
        #         tag_type, recall_best[tag_type], precision_best[tag_type], F1_best[tag_type]))
        print()
        # print("F1: %.2f - acc: %.2f" % (100 * F_all_best, 100 * acc_dev_best))
        # logger.info("F1: %.2f - acc: %.2f" % (100 * F_all_best, 100 * acc_dev_best))
        LAS = LAS_best
        UAS = UAS_best
        LS = LS_best
        print(" LAS: %.2f - UAS: %.2f - LS: %.2f" % (100 * LAS, 100 * UAS, 100 * LS))
        logger.info(" LAS: %.2f - UAS: %.2f - LS: %.2f" % (100 * LAS, 100 * UAS, 100 * LS))
        # Save validation metrics
        with open(os.path.join(model_dir, 'validation_metrics.txt'), 'w', encoding=encoding) as f:
            f.write("Best model - epoch %d - training loss %.4f - acc %.4f\n" % (epoch_best, loss_best, accuracy_best))
            f.write("Model name: %s\n" % os.path.basename(model_dir))
            # f.write("F1: %.2f - acc: %.2f\n" % (100 * F_all_best, 100 * acc_dev_best))
            f.write(" LAS: %.2f - UAS: %.2f - LS: %.2f\n" % (100 * LAS, 100 * UAS, 100 * LS))
            # for tag_type in tag_type_list:
            #     f.write("%s: r%.4f p%.4f f%.4f\n" % (
            #     tag_type, recall_best[tag_type], precision_best[tag_type], F1_best[tag_type]))
            f.write("Elapsed time: %.4fs\n" % elapsed_time)
    print("Elapsed time: %.4fs\n" % elapsed_time)
    logger.info("Elapsed time: %.4fs" % elapsed_time)


################################################################################
# Prediction                                                                   #
################################################################################
def predict(model_dir, hidden_size, cell_type, test_filepath):
    sess = restore_session(model_dir)

    tag_type_list = load_tag_type_list(model_dir)
    tag_dict = build_tag_dictionary(tag_type_list)
    # pos_dict = build_pos_dictionary()
    vocab = load_dictionary(model_dir)

    tags_map = [""] * len(tag_dict)
    for (tag, index) in tag_dict.items():
        tags_map[index] = tag

    exit_command_set = {":q", ":quit", "quit()", "exit()", '#exit#'}

    f = None
    if test_filepath:
        f = open(test_filepath, 'r', encoding=encoding)
    else:
        f = sys.stdin

    for query in f:
        query = query.strip()
        if not query:
            continue

        if query in exit_command_set:
            break

        # sys.stdout.write(query)
        # for tag_type in tag_type_list:
        #     sys.stdout.write('\t')
        #     sys.stdout.write(','.join(concepts[tag_type]))
        # sys.stdout.write('\n')

    f.close()
    sess.close()


def restore_session(model_dir):
    filepath = os.path.join(model_dir, 'tf_version.txt')
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding=encoding) as f:
            tf_version = f.readline().strip()
            if tf_version != tf.__version__:
                raise Exception('model was trained on TF %s, but the TF version on current machine is %s' %
                                (tf_version, tf.__version__))
    sess = tf.Session()
    ckpt = tf.train.get_checkpoint_state(model_dir)
    model_checkpoint_path = ckpt.model_checkpoint_path
    if not os.path.exists(model_checkpoint_path):
        model_checkpoint_path = os.path.join(model_dir, os.path.basename(model_checkpoint_path))
    saver = tf.train.import_meta_graph(model_checkpoint_path + '.meta')
    saver.restore(sess, model_checkpoint_path)
    return sess


def predict_interactive(model_dir, hidden_size, cell_type):
    # global max_word_length
    sess = restore_session(model_dir)

    tag_type_list = load_tag_type_list(model_dir)
    tag_dict, rel_dict = build_tag_dictionary(tag_type_list)
    pos_dict = build_pos_dictionary()
    vocab = load_dictionary(model_dir)
    if use_chars:
        char_dict = build_char_dictionary()
    else:
        char_dict = None
    # load_max_word_length()

    tags_map = [""] * len(tag_dict)
    for (tag, index) in tag_dict.items():
        tags_map[index] = tag

    pos_tagger = PerceptronTagger()

    exit_command_set = {":q", ":quit", "quit()", "exit()", '#exit#'}

    sys.stdout.write("> ")
    sys.stdout.flush()
    query = sys.stdin.readline()
    while query:
        query = query.strip()
        if not query:
            continue

        if query in exit_command_set:
            break
        begin = time.time()

        query = query.strip()
        tokens = nltk.word_tokenize(query)
        words = [token.lower() for token in tokens]
        pos_seq = [pos_pair[1] for pos_pair in pos_tagger.tag(tokens)]

        elements = []
        for i, word in enumerate(words):
            element = Word(i + 1, -1, '')
            elements.append(element)
        projective = True
        stack = [Word(0, -1, 'null')]
        queue = deque(elements)
        curr_idx = 0
        while len(queue) > 0 or len(stack) > 1:
            # Build an example
            # a “shortest stack” oracle which always prefers LEFT-ARCl over SHIFT.
            word_arr, pos_arr, arc_arr = build_feature_vector(stack, elements, curr_idx, words, pos_seq)
            wrd_id_arr = [vocab[w if w in vocab else NUM if is_number(w) else UNK] for w in word_arr]
            pos_id_arr = [pos_dict[pos_tag if pos_tag in pos_dict else NUL] for pos_tag in pos_arr]
            arc_id_arr = [rel_dict[arc] for arc in arc_arr]
            wrd_id_arrs_ = []
            pos_id_arrs_ = []
            arc_id_arrs_ = []
            chr_id_mats_ = []
            wrd_id_arrs_.append(wrd_id_arr)
            pos_id_arrs_.append(pos_id_arr)
            arc_id_arrs_.append(arc_id_arr)
            if use_chars:
                chr_id_mat = []
                for word in word_arr:
                    chr_id_seq = [char_dict[ch] for ch in word]
                    chr_id_mat.append(chr_id_seq)
                chr_id_mats_.append(chr_id_mat)
                # Padding
                chr_id_mats_, _ = pad_sequences(chr_id_mats_, pad_tok=0, nlevels=2)
            # Set the feed dictionary
            feed_dict = {  # 'input_tag_ids:0': tag_id_seqs[data_index:end_index],
                # 'input_pos_ids:0': pos_id_seqs[data_index:end_index],
                'wrd_id_arrs:0': wrd_id_arrs_,
                'pos_id_arrs:0': pos_id_arrs_,
                'arc_id_arrs:0': arc_id_arrs_,
                # 'chr_id_mats:0': chr_id_mats,
                # 'label_id_arr:0': label_id_arr,
                # 'weights:0': [class_id_weight_dict[label_id] for label_id in label_id_arr],
                'keep_prob:0': 1.0,
            }
            if use_chars:
                feed_dict['chr_id_mats:0'] = chr_id_mats_
            # print(feed_dict)
            logits = sess.run('logits:0', feed_dict=feed_dict)
            logits = logits[0]
            top_idx = np.argmax(logits)
            if top_idx == tag_dict['shift'] and len(queue) > 0:
                action = tags_map[top_idx]
            elif len(stack) <= 1 and len(queue) > 0:
                action = 'shift'
            else:
                second_idx = np.argmax(logits[1:]) + 1
                action = tags_map[second_idx]
            # action = tags_map[outputs[0]]
            if action.startswith('l-'):  # reduce left
                stack[-2].head = stack[-1].id
                stack[-2].rel = action.split('-')[1]
                stack[-1].l_children.appendleft(stack[-2])
                del stack[-2]
            elif action.startswith('r-'):  # reduce right
                stack[-1].head = stack[-2].id
                stack[-1].rel = action.split('-')[1]
                stack[-2].r_children.append(stack[-1])
                del stack[-1]
            else:  # shift
                stack.append(queue.popleft())
                curr_idx += 1

        print('> ID\t%s\tPOS\tHead\tRelation' % 'Word'.rjust(10, ' '))
        for i, word in enumerate(words):
            element = elements[i]
            print('> %d\t%10s\t%s\t%d\t%s' % (i + 1, tokens[i], pos_seq[i], element.head, element.rel))

        elapsed_time = time.time() - begin
        print("> Elapsed time: %.4fs\n" % elapsed_time)
        sys.stdout.flush()
        sys.stdout.write("> ")
        sys.stdout.flush()
        query = sys.stdin.readline()

    sess.close()


################################################################################
# Evaluation                                                                   #
################################################################################
def evaluate(model_dir, hidden_size, cell_type, eval_filepath, output_filepath):
    begin = time.time()
    if eval_filepath:
        print("Doing evaluation on %s" % eval_filepath)
    else:
        print("Doing evaluation from standard input")
    print('model directory:', model_dir)

    sess = restore_session(model_dir)

    tag_type_list = load_tag_type_list(model_dir)
    tag_dict, rel_dict = build_tag_dictionary(tag_type_list)
    pos_dict = build_pos_dictionary()
    vocab = load_dictionary(model_dir)
    if use_chars:
        char_dict = build_char_dictionary()
    else:
        char_dict = None
    # load_max_word_length()

    tags_map = [""] * len(tag_dict)
    for (tag, index) in tag_dict.items():
        tags_map[index] = tag

    if eval_filepath:
        f = open(eval_filepath, 'r', encoding=encoding)
    else:
        f = sys.stdin

    # if output_filepath:
    #     parent_dir = os.path.dirname(output_filepath)
    #     if not os.path.exists(parent_dir):
    #         os.makedirs(parent_dir)
    #     fout = open(output_filepath, 'w')
    # else:
    #     fout = None

    data_test = TextDataset(None, annotation_scheme, extend_sequence, file=f)
    metrics = evaluate_memory_efficient(sess, data_test, vocab, pos_dict, tag_dict, tags_map, rel_dict, batch_size, use_chars, char_dict)
    LAS = metrics['LAS']
    UAS = metrics['UAS']
    LS = metrics['LS']
    print(''.rjust(80, '-'))
    print(" LAS: %.2f - UAS: %.2f - LS: %.2f" % (100 * LAS, 100 * UAS, 100 * LS))
    # for tag_type in tag_type_list:
    #     print(''.rjust(80, '-'))
    #     print(('-%s' % tag_type).ljust(80, '-'))
    #     print("-Recall Precision F1".ljust(80, '-'))
    #     print(("-%.4f %.4f %.4f" % (R[tag_type], P[tag_type], F[tag_type])).ljust(80, '-'))
    # print(''.rjust(80, '-'))
    # print(('-%s' % 'All').ljust(80, '-'))
    # print("-Recall Precision F1 Accuracy".ljust(80, '-'))
    # print(("-%.4f %.4f %.4f %.4f" % (R_all, P_all, F_all, acc)).ljust(80, '-'))
    print(''.rjust(80, '-'))
    print()
    elapsed_time = time.time() - begin
    print("Elapsed time: %.4fs\n" % elapsed_time)
    # if fout:
    #     fout.close()
    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="A neural network model to recognize answer and context concepts for a query. "
                    "The type of recognized concepts depends on the specified model type.")
    parser.add_argument("-d", '--data_dir',
                        type=str,
                        help='training data directory path',
                        default='.'
                        )
    parser.add_argument("-t", '--training_filename',
                        type=str,
                        help='filename for the training data',
                        default='ManualLabeledConcepts.txt'
                        )
    parser.add_argument("-v", '--validation_filename',
                        type=str,
                        help='filename for the validation or development data',
                        default=''
                        )
    parser.add_argument("-T", '--task',
                        type=str,
                        help='task',
                        choices=['train', 'online', 'cv', 'predict', 'eval'],
                        default='train'
                        )
    parser.add_argument("-i", '--maxiter',
                        type=int,
                        help='maximal number of iterations',
                        default=-1
                        )
    # parser.add_argument("-m", '--model_dir_name',
    #                     type=str,
    #                     help='model directory name',
    #                     default='model'
    #                     )
    parser.add_argument("-l", '--learning_rate',
                        type=float,
                        help='learning rate',
                        default=0.001
                        )
    parser.add_argument("-e", '--embedding_size',
                        type=int,
                        help='embedding size',
                        default=64
                        )
    parser.add_argument("-H", '--hidden_size',
                        type=int,
                        help='hidden size',
                        default=64
                        )
    parser.add_argument('--hidden_size_char',
                        type=int,
                        help='hidden size of LSTM on characters',
                        default=100
                        )
    parser.add_argument("-b", '--batch_size',
                        type=int,
                        help='mini-batch size',
                        default=10
                        )
    parser.add_argument("-w", '--window_size',
                        type=int,
                        help='window size',
                        default=4
                        )
    # parser.add_argument("-M", '--model_type',
    #                     type=str,
    #                     help='model type',
    #                     choices=['answer', 'context'],
    #                     default="answer"
    #                     )
    parser.add_argument("-s", '--split_size',
                        type=int,
                        help='split size',
                        default=5
                        )
    parser.add_argument("-a", '--acc_stop',
                        type=float,
                        help='accuracy to stop training',
                        default=None
                        )
    parser.add_argument('-f', '--fix_word_embedding',
                        dest='fix_word_embedding',
                        action='store_true',
                        help='if word embedding is fixed',
                        default=False
                        )
    parser.add_argument("-p", '--pretrained_embedding_path',
                        type=str,
                        help='pretrained embedding path',
                        default=''
                        )
    parser.add_argument('--verbose',
                        dest='verbose',
                        action='store_true',
                        help='if more information is going to be displayed',
                        default=False
                        )
    parser.add_argument('--resume',
                        dest='resume',
                        action='store_true',
                        help='if model training is resumed on top of last checkpoint',
                        default=False
                        )
    parser.add_argument('--use_senna',
                        dest='use_senna',
                        action='store_true',
                        help='if SENNA is used for POS tagging',
                        default=False
                        )
    parser.add_argument("-n", '--num_epochs_without_improvement',
                        type=int,
                        help='number of epochs without improvement',
                        default=25
                        )
    parser.add_argument("-k", '--keep_prob',
                        type=float,
                        help='number of epochs without improvement',
                        default=0.5
                        )
    parser.add_argument('-m', '--model_dir',
                        type=str,
                        help='model directory path',
                        default=''
                        )
    parser.add_argument('--test_filepath',
                        type=str,
                        help='test filepath',
                        default=''
                        )
    parser.add_argument('--eval_filepath',
                        type=str,
                        help='path to the evaluation file',
                        default=''
                        )
    parser.add_argument('--output_filepath',
                        type=str,
                        help='path to the evaluation file',
                        default=''
                        )
    parser.add_argument('--cv_dir',
                        type=str,
                        help='directory path to save cross validation results',
                        default=''
                        )
    parser.add_argument("-c", '--chopoff',
                        type=int,
                        help='minimal frequency a vocabulary term should have',
                        default=1
                        )
    parser.add_argument('-C', '--cell_type',
                        type=str,
                        help='Cell type for RNN',
                        choices=['GRU', 'LSTM'],
                        default='LSTM'
                        )
    parser.add_argument('-A', '--annotation_scheme',
                        type=str,
                        help='annotation scheme for tagged training data. \'<>\' for XML tags or \'[]\' '
                             'for named bracket tags',
                        choices=['<>', '[]', 'CoNLL'],
                        default='CoNLL'
                        )
    parser.add_argument("-E", '--embedding_size_char',
                        type=int,
                        help='embedding size for characters',
                        default=100
                        )
    parser.add_argument('--use_chars',
                        dest='use_chars',
                        action='store_true',
                        help='if character embedding is used',
                        default=False
                        )
    parser.add_argument('--decay',
                        type=float,
                        help='decay for learning rate',
                        default=0.98
                        )
    # parser.add_argument('--shift_backward',
    #                     dest='shift_backward',
    #                     action='store_true',
    #                     help='if the last state vectors are not used for backward state sequence',
    #                     default=False
    #                     )
    parser.add_argument('--exhaust_backward',
                        dest='exhaust_backward',
                        action='store_true',
                        help='if all state vectors are used for backward state sequence',
                        default=False
                        )
    # parser.add_argument('--use_all_chars',
    #                     dest='use_all_chars',
    #                     action='store_true',
    #                     help='if all characters are used for character level embedding, i.e., no clipping',
    #                     default=False
    #                     )
    parser.add_argument('--max_gradient_norm',
                        type=float,
                        help='max gradient norm for clipping, if negative no clipping',
                        default=-1
                        )
    parser.add_argument('--extend_sequence',
                        dest='extend_sequence',
                        action='store_true',
                        help='if an input sequence is extended by extra tokens, e.g., "^start" or "end$"',
                        default=False
                        )
    parser.add_argument('--yield_data',
                        dest='yield_data',
                        action='store_true',
                        help='if fetch a mini-batch raw data each time from a text file to save memory',
                        default=False
                        )
    parser.add_argument('-B', '--brnn_type',
                        type=str,
                        help='BRNN architecture type',
                        choices=['vanilla', 'backward_shift', 'residual'],
                        default='vanilla'
                        )
    parser.add_argument('--debug',
                        dest='debug',
                        action='store_true',
                        help='if in debug mode',
                        default=False
                        )
    args = parser.parse_args()

    data_dir = args.data_dir
    training_filename = args.training_filename
    task = args.task
    max_iter = args.maxiter

    learning_rate = args.learning_rate
    batch_size = args.batch_size
    embedding_size = args.embedding_size
    hidden_size = args.hidden_size
    window_size = args.window_size
    # model_type = ModelType.ANSWER if args.model_type == "answer" else ModelType.CONTEXT
    # By default train_word_embedding = True
    fix_word_embedding = args.fix_word_embedding
    train_word_embedding = not args.fix_word_embedding
    pretrained_embedding_path = args.pretrained_embedding_path
    verbose = args.verbose
    validation_filename = args.validation_filename
    keep_prob = args.keep_prob
    chopoff = args.chopoff
    cell_type = args.cell_type
    annotation_scheme = args.annotation_scheme
    if annotation_scheme == '<>':
        extract_word = extract_word_xml
    elif annotation_scheme == '[]':
        extract_word = extract_word_bracket
    embedding_size_char = args.embedding_size_char
    hidden_size_char = args.hidden_size_char
    use_chars = args.use_chars
    decay = args.decay
    exhaust_backward = args.exhaust_backward
    shift_backward = not exhaust_backward
    # use_all_chars = args.use_all_chars
    max_gradient_norm = args.max_gradient_norm
    extend_sequence = args.extend_sequence
    # yield_data = args.yield_data
    yield_data = True
    brnn_type = args.brnn_type

    debug = args.debug
    if debug:
        data_dir = r"C:\Users\miqian\Data\Dependency Parsing\UD_English-EWT"
        training_filename = 'en-ud-train.conllu'
        validation_filename = ''
        task = 'train'
        maxiter = 500
        learning_rate = 0.001
        embedding_size = 256
        hidden_size = 300
        embedding_size_char = 100
        hidden_size_char = 100
        batch_size = 20
        fix_word_embedding = not True
        pretrained_embedding_path = r"C:\Users\miqian\Documents\Visual Studio 2015\Projects\PythonExamples\GENSIMExamples\Model\text8-256.bin"
        chopoff = 1
        annotation_scheme = 'CoNLL'
        use_chars = True
        decay = 1.0
        max_gradient_norm = -1
        extend_sequence = False
        yield_data = True
        brnn_type = 'vanilla'

    # By default initialized_by_pretrained_embedding = False
    initialized_by_pretrained_embedding = pretrained_embedding_path != ''
    zero_padded = yield_data

    # model_dir_name = args.model_dir_name
    model_dir = args.model_dir
    model_dir_name = os.path.basename(model_dir) if model_dir != '' else 'model-NN'

    if args.use_senna:
        senna_path = r"C:\Users\miqian\Software\senna-v3.0"
        pos_tagger = SennaTagger(senna_path)
        print("SENNA is used for POS tagging.")

    if task == "train":
        resume = args.resume
        acc_stop = args.acc_stop
        if model_dir == '':

            # if yield_data:
            #     model_dir_name += "-x"

            if extend_sequence:
                model_dir_name += "-^"

            # if not shift_backward:
            #     model_dir_name += "-$"
            # if brnn_type == 'vanilla':
            #     model_dir_name += "-B1"
            # elif brnn_type == 'backward_shift':
            #     model_dir_name += "-B2"
            # elif brnn_type == 'residual':
            #     model_dir_name += "-B3"

            # if use_all_chars:
            #     model_dir_name += "-A"

            if max_gradient_norm > 0:
                model_dir_name += "-m%g" % max_gradient_norm

            if use_chars:
                model_dir_name += "-E%d-h%d" % (embedding_size_char, hidden_size_char)
            model_dir_name += "-e%d-H%d-k%s-l%s-b%d" % \
                              (embedding_size, hidden_size, keep_prob, learning_rate, batch_size)
            if acc_stop:
                model_dir_name += "-a%g" % acc_stop
            elif max_iter > 0:
                model_dir_name += "-i%d" % max_iter

            if chopoff > 1:
                model_dir_name += "-c%d" % chopoff

            if initialized_by_pretrained_embedding:
                model_dir_name += "-%s" % os.path.basename(pretrained_embedding_path)

            if fix_word_embedding:
                model_dir_name += "-fixed"

            model_dir_name += "-%s" % os.path.splitext(os.path.basename(training_filename))[0]

            if validation_filename:
                model_dir_name += "-%s" % os.path.splitext(os.path.basename(validation_filename))[0]

            model_dir = os.path.join(data_dir, model_dir_name)
        # if not os.path.exists(model_dir):
        #     os.makedirs(model_dir)
        parameters = {"l": learning_rate,
                      "e": embedding_size,
                      "H": hidden_size,
                      "k": keep_prob,
                      # "M": model_type,
                      "i": max_iter,
                      "a": acc_stop,
                      "c": chopoff,
                      "C": cell_type,
                      "b": batch_size}
        for (key, value) in parameters.items():
            print("Parameter %s: %s" % (key, value))
        print("Python script name: %s" % os.path.splitext(os.path.basename(__file__))[0])
        train_memory_efficient()
        print("Models are saved in %s" % os.path.abspath(model_dir))
    elif task == "predict":
        match = re.compile(r'-H([\d]+)-').findall(model_dir_name)
        if match:
            hidden_size = int(match[0])
        match = re.compile(r'model-([\w]+)-([\w]+)-').findall(model_dir_name)
        if match:
            cell_type = match[0][1]
        test_filepath = args.test_filepath
        predict(model_dir, hidden_size, cell_type, test_filepath)
    elif task == "eval":
        match = re.compile(r'-H([\d]+)-').findall(model_dir_name)
        if match:
            hidden_size = int(match[0])
        match = re.compile(r'model-([\w]+)-([\w]+)-').findall(model_dir_name)
        if match:
            cell_type = match[0][1]
        if model_dir_name.find('-^-') != -1:
            extend_sequence = True
        # if model_dir_name.find('-A-') != -1:
        #     use_all_chars = True
        match = re.compile(r'-E([\d]+)-').findall(model_dir_name)
        if match:
            use_chars = True
        eval_filepath = args.eval_filepath
        output_filepath = args.output_filepath
        evaluate(model_dir, hidden_size, cell_type, eval_filepath, output_filepath)
    elif task == "online":
        match = re.compile(r'-H([\d]+)-').findall(model_dir_name)
        if match:
            hidden_size = int(match[0])
        match = re.compile(r'model-([\w]+)-([\w]+)-').findall(model_dir_name)
        if match:
            cell_type = match[0][1]
        if model_dir_name.find('-^-') != -1:
            extend_sequence = True
        # if model_dir_name.find('-A-') != -1:
        #     use_all_chars = True
        match = re.compile(r'-E([\d]+)-').findall(model_dir_name)
        if match:
            use_chars = True
        predict_interactive(model_dir, hidden_size, cell_type)
