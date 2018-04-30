from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import json
import argparse
import tensorflow as tf

from qa_model import Encoder, QASystem, Decoder
from qa_data import load_dataset
from os.path import join as pjoin

import logging

tf.flags._global_parser = argparse.ArgumentParser()
logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("keep_prob", 1, "Fraction of units randomly kept on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 30, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 5, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("num_hidden_unit", 250, "Size of each model layer.")
tf.app.flags.DEFINE_integer("embedding_size", 300, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_string("data_dir", "embedding", "SQuAD directory (default ./embedding)")
tf.app.flags.DEFINE_string("train_dir", "./train/train007", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("log_dir", "log", "Logging directory (default: ./log).")

FLAGS = tf.app.flags.FLAGS


def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (os.path.exists(ckpt.model_checkpoint_path) or os.path.exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model


def initialize_vocab(vocab_path):
    if os.path.exists(vocab_path):
        rev_vocab = []
        with open(vocab_path, "r", encoding="utf-8") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)

def main(_):

    dataset = [load_dataset(["embedding/train.ids.question",
                             "embedding/train.ids.context", "data/train.span"]),
               load_dataset(["embedding/val.ids.question",
                             "embedding/val.ids.context", "data/val.span"])]

    embed_path = pjoin("embedding", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    vocab_path = pjoin(FLAGS.data_dir, "vocab.dat")
    vocab, rev_vocab = initialize_vocab(vocab_path)

    tf.reset_default_graph()
    encoder = Encoder(FLAGS.num_hidden_unit, tf.contrib.rnn.GRUCell)
    decoder = Decoder(FLAGS.num_hidden_unit, tf.contrib.rnn.BasicLSTMCell)
    embedding = np.load(embed_path)["glove"]
    
    qa = QASystem(encoder, decoder, embedding, FLAGS.keep_prob)
    
    if not os.path.exists(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    with tf.Session() as sess:
        initialize_model(sess, qa, FLAGS.train_dir)
        qa.train(sess, dataset, FLAGS.epochs, FLAGS.batch_size, FLAGS.train_dir)

if __name__ == "__main__":
    tf.app.run()
