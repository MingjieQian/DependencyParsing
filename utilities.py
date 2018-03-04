import re
import sys
import time
import logging
import numpy as np
from collections import deque

__author__ = "Mingjie Qian"
__date__ = "February 25th, 2018"

NUL = 'NUL'
NUM = 'NUM'
UNK = 'UNK'

encoding = 'utf-8'


class TextDataset(object):
    """Class that iterates over a text Dataset

    __iter__ method yields a tuple (words, tags)
        words: list of raw words
        tags: list of raw tags

    If processing_word and processing_tag are not None,
    optional preprocessing is appplied

    Example:
        ```python
        data = CoNLLDataset(filename)
        for sentence, tags in data:
            pass
        ```

    """
    def __init__(self, filepath, annotation_scheme=None, extend_sequence=False, max_iter=None, file=None):
        """
        Args:
            filepath: path to the file
            processing_words: (optional) function that takes a word as input
            processing_tags: (optional) function that takes a tag as input
            max_iter: (optional) max number of sentences to yield

        """
        self.filepath = filepath
        self.annotation_scheme = annotation_scheme
        self.extend_sequence = extend_sequence
        self.max_iter = max_iter
        self.file = file
        self.length = None

    def __iter__(self):
        niter = 0
        if self.file is not None:
            f = self.file
        else:
            f = open(self.filepath, 'r', encoding=encoding)
        if self.annotation_scheme == 'CoNLL':
            split_pattern = re.compile(r'[ \t]')
            # with open(self.filepath, 'r', encoding=encoding) as f:
            # Word Indices are based on 1. Root's index is 0. Root's head is -1.
            words, heads = [], []
            pos_seq = []
            rel_seq = []
            for line in f:
                line = line.strip()
                if len(line) == 0 or line.startswith("-DOCSTART-"):
                    if len(words) != 0:
                        niter += 1
                        if self.max_iter is not None and niter > self.max_iter:
                            break
                        yield words, pos_seq, heads, rel_seq
                        words, pos_seq, heads, rel_seq = [], [], [], []
                elif line.startswith('#'):
                    continue
                else:
                    # ls = line.split(' ')
                    ls = split_pattern.split(line)
                    if ls[0].find('.') != -1:
                        print(line)
                        continue
                    if ls[6].find('_') != -1:
                        print(line)
                    word, head = ls[1], int(ls[6])
                    pos, rel = ls[4], ls[7]
                    word = word.lower()
                    words += [word]
                    heads += [head]
                    pos_seq.append(pos)
                    rel_seq.append(rel)
        else:
            pass
        f.close()

    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length


def fetch_minibatches(data, minibatch_size):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)

    Yields:
        list of word sequences, list of tag sequences

    """
    words_batch, pos_seq_batch, heads_batch, rel_seq_batch = [], [], [], []
    for (words, pos_seq, heads, rel_seq) in data:
        if len(words_batch) == minibatch_size:
            yield words_batch, pos_seq_batch, heads_batch, rel_seq_batch
            words_batch, pos_seq_batch, heads_batch, rel_seq_batch = [], [], [], []

        # if type(x[0]) == tuple:
        #     x = zip(*x)
        words_batch += [words]
        pos_seq_batch += [pos_seq]
        heads_batch += [heads]
        rel_seq_batch += [rel_seq]

    if len(words_batch) != 0:
        yield words_batch, pos_seq_batch, heads_batch, rel_seq_batch


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with

    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        nlevels: "depth" of padding, for the case where we have characters ids

    Returns:
        a list of list where each sublist has same length

    """
    if nlevels == 1:
        max_length = max(map(lambda x : len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                                          pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq))
                               for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x : len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded,
                                            [pad_tok]*max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0,
                                            max_length_sentence)

    return sequence_padded, sequence_length


class Word:
    def __init__(self, id, head, rel):
        self.id = id
        self.head = head
        self.rel = rel
        self.l_children = deque()
        self.r_children = []

    def __repr__(self):
        return '[%d, %d, %s]' % (self.id, self.head, self.rel)


def build_feature_vector(stack, elements, curr_idx, words, pos_seq):

    word_idx_arr = []
    # (1) The top 3 words on the stack and buffer s1; s2; s3; b1; b2; b3
    word_idx_arr.append(stack[-1].id - 1)
    word_idx_arr.append(-2 if len(stack) < 2 else stack[-2].id - 1)
    word_idx_arr.append(-2 if len(stack) < 3 else stack[-3].id - 1)
    word_idx_arr.append(-2 if curr_idx >= len(elements) else elements[curr_idx].id - 1)
    word_idx_arr.append(-2 if curr_idx + 1 >= len(elements) else elements[curr_idx + 1].id - 1)
    word_idx_arr.append(-2 if curr_idx + 2 >= len(elements) else elements[curr_idx + 2].id - 1)
    arc_idx_arr = []
    arc_arr = []
    # (2) lc1(si); rc1(si); lc2(si); rc2(si), i = 1; 2.
    for i in [1, 2]:
        if len(stack) > i - 1 and len(stack[-i].l_children) > 0:
            element = stack[-i].l_children[0]
            arc_idx_arr.append(element.id - 1)
            arc_arr.append(element.rel)
        else:
            arc_idx_arr.append(-2)
            arc_arr.append('null')
        if len(stack) > i - 1 and len(stack[-i].r_children) > 0:
            element = stack[-i].r_children[-1]
            arc_idx_arr.append(element.id - 1)
            arc_arr.append(element.rel)
        else:
            arc_idx_arr.append(-2)
            arc_arr.append('null')

        if len(stack) > i - 1 and len(stack[-i].l_children) > 1:
            element = stack[-i].l_children[1]
            arc_idx_arr.append(element.id - 1)
            arc_arr.append(element.rel)
        else:
            arc_idx_arr.append(-2)
            arc_arr.append('null')
        if len(stack) > i - 1 and len(stack[-i].r_children) > 1:
            element = stack[-i].r_children[-2]
            arc_idx_arr.append(element.id - 1)
            arc_arr.append(element.rel)
        else:
            arc_idx_arr.append(-2)
            arc_arr.append('null')

    # (3) lc1(lc1(si)); rc1(rc1(si)), i = 1; 2.
    for i in [1, 2]:
        if len(stack) > i - 1 and len(stack[-i].l_children) > 0 and len(stack[-i].l_children[0].l_children) > 0:
            element = stack[-i].l_children[0].l_children[0]
            arc_idx_arr.append(element.id - 1)
            arc_arr.append(element.rel)
        else:
            arc_idx_arr.append(-2)
            arc_arr.append('null')
        if len(stack) > i - 1 and len(stack[-i].r_children) > 0 and len(stack[-i].r_children[-1].r_children) > 0:
            element = stack[-i].r_children[-1].r_children[-1]
            arc_idx_arr.append(element.id - 1)
            arc_arr.append(element.rel)
        else:
            arc_idx_arr.append(-2)
            arc_arr.append('null')

    # S_w
    word_arr = [words[i] if i >= 0 else 'ROOT' if i == -1 else 'NUL' for i in word_idx_arr]
    word_arr.extend([words[i] if i >= 0 else 'ROOT' if i == -1 else 'NUL' for i in arc_idx_arr])

    # S_t
    pos_arr = [pos_seq[i] if i >= 0 else 'NUL' for i in word_idx_arr]
    pos_arr.extend([pos_seq[i] if i >= 0 else 'NUL' for i in arc_idx_arr])

    # S_l

    return word_arr, pos_arr, arc_arr


def is_number(s):
    return s.isdigit()


def is_number_(s):
    # i = 0
    # while i < len(s) and (s[i] == '-' or s[i] == '+'):
    #     i += 1
    # s = s[i:]
    s = s.strip().lstrip('+-')
    if not s:
        return False
    if s == '.':
        return False
    if s[0] == '.':
        s = s[1:]
    if s[0] == 'e':
        return False
    # Finite state machine
    # 0: before 'e'
    # 1: 1st position after '.'
    # 2: 2nd or later position after '.'
    # 3: 1st position after 'e'
    # 4: 2nd or later position after 'e'
    state = 0
    for c in s:
        if state == 0:
            if '0' <= c <= '9':
                pass
            elif c == '.':
                state = 1
            elif c == 'e':
                state = 3
            else:
                return False
        elif state == 1:
            if '0' <= c <= '9':
                state = 2
            elif c == 'e':
                state = 3
            else:
                return False
        elif state == 2:
            if '0' <= c <= '9':
                pass
            elif c == 'e':
                state = 3
            else:
                return False
        elif state == 3:
            if '0' <= c <= '9':
                pass
            elif c == '+' or c == '-':
                pass
            else:
                return False
            state = 4
        elif state == 4:
            if '0' <= c <= '9':
                pass
            else:
                return False
    return state != 3


def evaluate_memory_efficient(sess, data_dev, vocab, pos_dict, tag_dict, tags_map, rel_dict, batch_size, use_chars=False, char_dict=None):
    LAS = 0
    UAS = 0
    LS = 0
    cnt = 0
    for words_batch, pos_seq_batch, heads_batch, rel_seq_batch in fetch_minibatches(data_dev, batch_size):
        for words, pos_seq, heads, rel_seq in zip(words_batch, pos_seq_batch, heads_batch, rel_seq_batch):
            cnt += len(words)
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
            # Evaluate
            for i, (word, head, rel) in enumerate(zip(words, heads, rel_seq)):
                element = elements[i]
                if element.head == head:
                    UAS += 1
                    if element.rel == rel:
                        LAS += 1
                if element.rel == rel:
                    LS += 1
    LAS /= cnt
    UAS /= cnt
    LS /= cnt
    F1 = {}
    P = {}
    R = {}
    return {'LAS': LAS, 'UAS': UAS, 'LS': LS, 'F1': F1, 'P': P, 'R': R}


def get_data_input(words_batch, pos_seq_batch, heads_batch, rel_seq_batch, vocab, pos_dict, tag_dict, rel_dict, use_chars=False, char_dict=None):
    # from DependencyParsing import is_number
    wrd_id_arrs = []
    pos_id_arrs = []
    arc_id_arrs = []
    chr_id_mats = []
    label_id_arr = []
    for words, pos_seq, heads, rel_seq in zip(words_batch, pos_seq_batch, heads_batch, rel_seq_batch):
        # wrd_id_seq = [vocab[w if w in vocab else NUM if is_number(w) else UNK] for w in words]
        unproc = {id: 0 for id in range(0, len(words) + 1)}
        for head in heads:
            if head in unproc:
                unproc[head] += 1
            # else:
            #     unproc[head] = 0

        elements = []
        for i, (word, head, rel) in enumerate(zip(words, heads, rel_seq)):
            element = Word(i + 1, head, rel)
            elements.append(element)

        # if 'moi' in words:
        #     # words.index('mol') != -1:
        #     print(words)

        wrd_id_arrs_ = []
        pos_id_arrs_ = []
        arc_id_arrs_ = []
        chr_id_mats_ = []
        label_id_arr_ = []

        projective = True
        stack = [Word(0, -1, 'null')]
        queue = deque(elements)
        # [queue.append(Word(i + 1, head, rel)) for i, word, head, rel in enumerate(zip(words, heads, rel_seq))]
        curr_idx = 0
        while len(queue) > 0 or len(stack) > 1:
            # Build a labeled example
            # a “shortest stack” oracle which always prefers LEFT-ARCl over SHIFT.
            word_arr, pos_arr, arc_arr = build_feature_vector(stack, elements, curr_idx, words, pos_seq)
            if len(stack) == 1:  # shift
                label_id_arr_.append(tag_dict['shift'])
                stack.append(queue.popleft())
                curr_idx += 1
            # elif stack[-2].head == stack[-1].id and unproc[stack[-2].id] == 0:  # reduce left
            elif stack[-2].head == stack[-1].id:  # reduce left
                tag = 'l-%s' % stack[-2].rel
                label_id_arr_.append(tag_dict[tag])
                unproc[stack[-1].id] -= 1
                # stack[-1].l_children.append(stack[-2])
                stack[-1].l_children.appendleft(stack[-2])
                del stack[-2]
            elif stack[-1].head == stack[-2].id and unproc[stack[-1].id] == 0:  # reduce right
                tag = 'r-%s' % stack[-1].rel
                label_id_arr_.append(tag_dict[tag])
                unproc[stack[-2].id] -= 1
                stack[-2].r_children.append(stack[-1])
                del stack[-1]
            else:  # shift
                if len(queue) == 0:
                    # print(words)
                    projective = False
                    break
                label_id_arr_.append(tag_dict['shift'])
                stack.append(queue.popleft())
                curr_idx += 1

            wrd_id_arr = [vocab[w if w in vocab else NUM if is_number(w) else UNK] for w in word_arr]
            pos_id_arr = [pos_dict[pos_tag if pos_tag in pos_dict else NUL] for pos_tag in pos_arr]
            arc_id_arr = [rel_dict[arc] for arc in arc_arr]
            wrd_id_arrs_.append(wrd_id_arr)
            pos_id_arrs_.append(pos_id_arr)
            arc_id_arrs_.append(arc_id_arr)
            if use_chars:
                chr_id_mat = []
                for word in word_arr:
                    chr_id_seq = [char_dict[ch] for ch in word]
                    chr_id_mat.append(chr_id_seq)
                chr_id_mats_.append(chr_id_mat)
        if projective:
            wrd_id_arrs += wrd_id_arrs_
            pos_id_arrs += pos_id_arrs_
            arc_id_arrs += arc_id_arrs_
            if use_chars:
                chr_id_mats += chr_id_mats_
            label_id_arr += label_id_arr_
    if use_chars:
        chr_id_mats, _ = pad_sequences(chr_id_mats, pad_tok=0, nlevels=2)

    return wrd_id_arrs, pos_id_arrs, arc_id_arrs, chr_id_mats, label_id_arr


def get_glove_vocab(filename):
    """Load vocab from file

    Args:
        filename: path to the glove vectors

    Returns:
        vocab: set() of strings
    """
    print("Building vocab...")
    vocab = set()
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip().split(' ')[0]
            vocab.add(word)
    print("- done. {} tokens".format(len(vocab)))
    return vocab


def get_logger(filename):
    """Return a logger instance that writes in filename

    Args:
        filename: (string) path to log.txt

    Returns:
        logger: (instance of logger)

    """
    logger = logging.getLogger('logger')
    # print(logger.handlers)  # A child logger initially doesn't have a handler

    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)

    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    # print(logging.getLogger().handlers)

    lhStdout = logging.getLogger().handlers[0]  # stdout is the only handler initially for the root logger
    # Remove the handler to stdout from the root's handlers list so that the logging information won't
    # display in the stdout.
    logging.getLogger().removeHandler(lhStdout)

    return logger


class Progbar(object):
    """Progbar class copied from keras (https://github.com/fchollet/keras/)

    Displays a progress bar.
    Small edit : added strict arg to update
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=[], exact=[], strict=[]):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far),
                                      current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]

        for k, v in strict:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = v

        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('='*(prog_width-1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.'*(self.width-prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if type(self.sum_values[k]) is list:
                    info += ' - %s: %.4f' % (k,
                                             self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k,
                                             self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=[]):
        self.update(self.seen_so_far+n, values)
