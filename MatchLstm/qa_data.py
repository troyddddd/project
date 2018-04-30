from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import argparse
import numpy as np
from os.path import join as pjoin

_PAD = b"<pad>"
_SOS = b"<sos>"
_UNK = b"<unk>"
_START_VOCAB = [_PAD, _SOS, _UNK]

PAD_ID = 0
SOS_ID = 1
UNK_ID = 2


def setup_args():
    parser = argparse.ArgumentParser()
    vocab_dir = os.path.join("embedding")
    glove_dir = os.path.join("glove.6B")
    source_dir = os.path.join("data")
    parser.add_argument("--source_dir", default=source_dir)
    parser.add_argument("--glove_dir", default=glove_dir)
    parser.add_argument("--vocab_dir", default=vocab_dir)
    parser.add_argument("--glove_dim", default=300, type=int)
    parser.add_argument("--random_init", default=True, type=bool)
    return parser.parse_args()


def basic_tokenizer(sentence):
    words = []
    for space_separated_fragment in str(sentence).strip().split():
        words.extend(re.split(" ", space_separated_fragment))
    return [w for w in words if w]


def initialize_vocabulary(vocabulary_path):
    # map vocab to word embeddings
    if os.path.exists(vocabulary_path):
        rev_vocab = []
        with open(vocabulary_path, "r", encoding="utf-8") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def process_glove(args, vocab_list, save_path, size=4e5, random_init=True):
    if not os.path.exists(save_path + ".npz"):
        glove_path = os.path.join(args.glove_dir, "glove.6B.{}d.txt".format(args.glove_dim))
        if random_init:
            glove = np.random.randn(len(vocab_list), args.glove_dim)
        else:
            glove = np.zeros((len(vocab_list), args.glove_dim))
        found = 0
        with open(glove_path, "r", encoding="utf-8") as fh:
            for line in fh:
                array = line.lstrip().rstrip().split(" ")
                word = array[0]
                vector = list(map(float, array[1:]))
                if word in vocab_list:
                    idx = vocab_list.index(word)
                    glove[idx, :] = vector
                    found += 1
                if word.capitalize() in vocab_list:
                    idx = vocab_list.index(word.capitalize())
                    glove[idx, :] = vector
                    found += 1
                if word.upper() in vocab_list:
                    idx = vocab_list.index(word.upper())
                    glove[idx, :] = vector
                    found += 1

        print("{}/{} of word vocab have corresponding vectors in {}".format(found, len(vocab_list), glove_path))
        np.savez_compressed(save_path, glove=glove)
        print("saved trimmed glove matrix at: {}".format(save_path))


def create_vocabulary(vocabulary_path, data_paths, tokenizer=None):
    if not os.path.exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, str(data_paths)))
        vocab = {}
        for path in data_paths:
            with open(path, "r", encoding="utf-8") as f:
                counter = 0
                for line in f:
                    counter += 1
                    if counter % 100000 == 0:
                        print("processing line %d" % counter)
                    tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                    for w in tokens:
                        if w in vocab:
                            vocab[w] += 1
                        else:
                            vocab[w] = 1
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        print("Vocabulary size: %d" % len(vocab_list))
        with open(vocabulary_path, "w", encoding="utf-8") as vocab_file:
            for w in vocab_list:
                vocab_file.write(str(w) + "\n")


def sentence_to_token_ids(sentence, vocabulary, tokenizer=None):
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    return [vocabulary.get(w, UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None):
    if not os.path.exists(target_path):
        print("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with open(data_path, "r", encoding="utf-8") as data_file:
            with open(target_path, "w", encoding="utf-8") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 5000 == 0:
                        print("tokenizing line %d" % counter)
                    token_ids = sentence_to_token_ids(line, vocab, tokenizer)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")

def load_dataset(fns):
    def read_file(fn):
        res = []
        with open(fn, "r", encoding="utf-8") as fh:
            for line in fh:
                res.append([int(i) for i in line.split()])
        return res
    content = [read_file(fn) for fn in fns]
    data = []
    for i in range(len(content[0])):
        data.append([c[i] for c in content])
    return data

def create_minibatch(dataset, batch_size, start):
    minibatch = dataset[start : start+batch_size]
    if len(minibatch) < batch_size:
        minibatch.extend(dataset[-(batch_size-len(minibatch)):])
    question_length = max(len(q) for q,p,a in minibatch)
    paragraph_length = max(len(p) for q,p,a in minibatch)
    questions, paragraphs, answers = [], [], []
    for q,p,a in minibatch:
        questions.append(q + [0 for _ in range(question_length-len(q))])
        paragraphs.append(p + [0 for _ in range(paragraph_length-len(p))])
        answers.append(a)
    return (np.array(questions), np.array(paragraphs), np.array(answers),
            np.array([question_length for _ in range(batch_size)]),
            np.array([paragraph_length for _ in range(batch_size)]))

if __name__ == '__main__':
    args = setup_args()
    vocab_path = pjoin(args.vocab_dir, "vocab.dat")

    train_path = pjoin(args.source_dir, "train")
    valid_path = pjoin(args.source_dir, "val")
    dev_path = pjoin(args.source_dir, "dev")

    create_vocabulary(vocab_path,
                      [pjoin(args.source_dir, "train.context"),
                       pjoin(args.source_dir, "train.question"),
                       pjoin(args.source_dir, "val.context"),
                       pjoin(args.source_dir, "val.question")])
    vocab, rev_vocab = initialize_vocabulary(pjoin(args.vocab_dir, "vocab.dat"))

    # ======== Trim Distributed Word Representation =======
    # If you use other word representations, you should change the code below

    process_glove(args, rev_vocab, args.vocab_dir + "/glove.trimmed.{}".format(args.glove_dim),
                  random_init=args.random_init)

    # ======== Creating Dataset =========
    # We created our data files seperately
    # If your model loads data differently (like in bulk)
    # You should change the below code

    x_train_dis_path = args.vocab_dir + "/train.ids.context"
    y_train_ids_path = args.vocab_dir + "/train.ids.question"
    data_to_token_ids(train_path + ".context", x_train_dis_path, vocab_path)
    data_to_token_ids(train_path + ".question", y_train_ids_path, vocab_path)

    x_dis_path = args.vocab_dir + "/val.ids.context"
    y_ids_path = args.vocab_dir + "/val.ids.question"
    data_to_token_ids(valid_path + ".context", x_dis_path, vocab_path)
    data_to_token_ids(valid_path + ".question", y_ids_path, vocab_path)