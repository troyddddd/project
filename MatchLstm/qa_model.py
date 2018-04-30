from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
import random
import json
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from qa_data import create_minibatch
from evaluate import exact_match_score, f1_score
from src.rnn import bidirectional_dynamic_rnn, dynamic_rnn, static_rnn
from src.attention_wrapper import BahdanauAttention, AttentionWrapper # _maybe_mask_score

logging.basicConfig(level=logging.INFO)


class Encoder(object):
    def __init__(self, num_hidden_unit, cell=tf.contrib.rnn.BasicLSTMCell):
        self.num_hidden_unit = num_hidden_unit
        self.cell = cell

    def encode(self, question, paragraph, Q, P):
        """
        :param question/paragraph: vector representation of the question/paragraph.
        :param Q/P: length of question/paragraph.
        :return: encoded representation of the input.
        """
        
        # run bi-directional lstm to get question/paragraph representation
        with tf.variable_scope("encoded_question"):
            question_cell = self.cell(self.num_hidden_unit)
            encoded_question, _ = bidirectional_dynamic_rnn(
                    question_cell, question_cell, question, Q, dtype=tf.float32)
            
        with tf.variable_scope("encoded_paragraph"):
            paragraph_cell = self.cell(self.num_hidden_unit)
            encoded_paragraph, _ = bidirectional_dynamic_rnn(
                    paragraph_cell, paragraph_cell, paragraph, P, dtype=tf.float32)

        return tf.reduce_sum(encoded_question, axis=0), tf.reduce_sum(encoded_paragraph, axis=0)


class Decoder(object):
    def __init__(self, num_hidden_unit, cell=tf.contrib.rnn.BasicLSTMCell):
        self.num_hidden_unit = num_hidden_unit
        self.cell = cell

    def decode(self, encoded_question, encoded_paragraph, Q, P, label):
        """
        Takes in knowledge representation from encoder.
        Output prediction (probability) for answer start/end position.
        """
        attention = self.match_lstm(encoded_question, encoded_paragraph, Q, P)
        logits = self.pointer_net(attention, Q, P, label)
        return logits
    
    def match_lstm(self, encoded_question, encoded_paragraph, Q, P):
        with tf.variable_scope("match_lstm"):
            attention_mechanism = BahdanauAttention(
                    encoded_question.get_shape()[-1], encoded_question, memory_sequence_length=Q)
            cell = self.cell(self.num_hidden_unit)
            attention = AttentionWrapper(cell, attention_mechanism, output_attention=False,
                                         attention_input_fn=lambda x, y: tf.concat([x, y], axis=-1))
            
            output_forward, _ = dynamic_rnn(attention, encoded_paragraph,
                                            dtype=tf.float32, scope="attention")
            
            reversed_encoded_paragraph = array_ops.reverse_sequence(encoded_paragraph, P, 1, 0)
            output_backward, _ = dynamic_rnn(attention, reversed_encoded_paragraph,
                                             dtype=tf.float32, scope="attention")
            output_backward = array_ops.reverse_sequence(output_backward, P, 1, 0)
        
        return tf.concat([output_forward, output_backward], axis=-1)
    
    def pointer_net(self, attention, Q, P, label):
        with tf.variable_scope("answer_pointer"):
            attention_mechanism = BahdanauAttention(
                    attention.get_shape()[-1], attention, memory_sequence_length=P)
            cell = self.cell(self.num_hidden_unit)
            output = AttentionWrapper(cell, attention_mechanism, cell_input_fn=lambda x,y:y)
            logits, _ = static_rnn(output, tf.unstack(label, axis=1), dtype=tf.float32)
        return logits
    

class QASystem(object):
    def __init__(self, encoder, decoder, embedding, keep_prob):
        
        # ==== set up placeholder tokens ========
        self.setup_placeholers()

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings(embedding)
            self.setup_system(encoder, decoder)
            self.setup_loss()
            self.keep_prob = keep_prob
            self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
            self.saver = tf.train.Saver()

        # ==== set up training/updating procedure ====
    
    def setup_placeholers(self):
        # placeholders for input data ids
        self.question = tf.placeholder(tf.int32, shape=[None,None], name="question")
        self.paragraph = tf.placeholder(tf.int32, shape=[None,None], name="paragraph")
        
        # placeholders for input data length
        self.Q = tf.placeholder(tf.int32, shape=[None], name="Q")
        self.P = tf.placeholder(tf.int32, shape=[None], name="P")
        
        # placeholder for label and dropout
        self.label = tf.placeholder(tf.int32, shape=[None, 2], name="label")
        self.dropout = tf.placeholder(tf.float32, shape=[], name="dropout")

    def setup_system(self, encoder, decoder):
        # setup the system, which includes a bi-directional lstm as encoder,
        # and match-lstm with answer pointer as decoder.
        encoded_question, encoded_paragraph = encoder.encode(
                self.question_embedding, self.paragraph_embedding, self.Q, self.P)
        encoded_question_dropout = tf.nn.dropout(encoded_question, self.dropout)
        encoded_paragraph_dropout = tf.nn.dropout(encoded_paragraph, self.dropout)
        self.logits = decoder.decode(encoded_question_dropout, encoded_paragraph_dropout,
                                     self.Q, self.P, self.label)


    def setup_loss(self):
        loss_start = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits[0], labels=self.label[:,0])
        loss_end = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits[1], labels=self.label[:,1])
        self.loss = tf.reduce_mean(loss_start + loss_end)

    def setup_embeddings(self, embedding):
        with vs.variable_scope("embedding"):
            _embedding = tf.Variable(embedding, name="_embedding",
                                     dtype=tf.float32, trainable=False)
            question_embedding = tf.nn.embedding_lookup(
                    _embedding, self.question, name="question_embedding")
            paragraph_embedding = tf.nn.embedding_lookup(
                    _embedding, self.paragraph, name="paragraph_embedding")
            self.question_embedding = tf.nn.dropout(question_embedding, self.dropout+(1-self.dropout)/2)
            self.paragraph_embedding = tf.nn.dropout(paragraph_embedding, self.dropout+(1-self.dropout)/2)

    def feed(self, mini_batch, train=True):
        questions, paragraphs, answers, Q, P = mini_batch
        input_feed = {
                self.question : questions,
                self.paragraph : paragraphs,
                self.Q : Q,
                self.P : P,
                self.label : answers,
                self.dropout : self.keep_prob if train else 1
        }
        return input_feed
    
    def optimize(self, session, mini_batch):
        # one iteration of training
        _, loss = session.run([self.train_op, self.loss], self.feed(mini_batch))
        return loss

    def test(self, session, mini_batch):
        # given new data, predict probability, or calculate loss (if answer provided)
        logits, loss = session.run([self.logits, self.loss], self.feed(mini_batch, False))
        return logits, loss

    def answer(self, session, sample):
        # given data, predict start/end position
        (yp1, yp2), _ = self.test(session, sample)
        a_s = np.argmax(yp1, axis=1)
        a_e = np.argmax(yp2, axis=1)
        return a_s, a_e, yp1, yp2

    def evaluate_answer(self, session, dataset, save=False):
        # Evaluate the model's performance using the harmonic mean of F1 and
        # Exact Match (EM) with the set of true answer labels.
        
        res = []
        prob = []
        for j in range(0, len(dataset)):
            sample = create_minibatch(dataset, 1, j)
            s, e, p1, p2 = self.answer(session, sample)
            _,p,a,_,_ = sample
            idx = list(p[0])
            res.append((idx[s[0]:e[0]+1], idx[a[0][0]:a[0][1]+1]))
            
            # save prediciton probability for future use
            if save:
                prob.append((p1.tolist(), p2.tolist(), p.tolist(), a.tolist()))

        f1 = 0.
        em = 0.
        for p, g in res:
            text_p = " ".join(str(i) for i in p)
            text_g = " ".join(str(i) for i in g)
            f1 += f1_score(text_p, text_g)
            em += exact_match_score(text_p, text_g)

        return f1/len(dataset), em/len(dataset), prob

    def train(self, session, dataset, num_epoch, batch_size, train_dic):
        # main training loop
        
        train, val = dataset
        scores = []
        start = time.time()
        
        for i in range(num_epoch):
            
            # shuffle data before each epoch
            random.shuffle(train)
            
            # run SGD (Adam)
            start = time.time()
            for j in range(0, len(train), batch_size):
                mini_batch = create_minibatch(train, batch_size, j)
                loss = self.optimize(session, mini_batch)
                
                # periodically monitor loss
                if j % (10 * batch_size) == 0:
                    print("epoch {} iteration {}, elapsed {:.2f} min, train loss: {:.5f}".format(
                            i, j / batch_size, (time.time()-start)/60, loss))
                
                # periodically evaluate loss/EM/F1 using validation batch
                if j % (100 * batch_size) == 0:
                    val_batch = random.sample(val, batch_size)
                    mini_batch = create_minibatch(val_batch, batch_size, 0)
                    _, loss = self.test(session, mini_batch)
                    f1, em, _ = self.evaluate_answer(session, val_batch)
                    print("epoch {} iteration {}, elapsed {:.2f} min, "
                          "val loss: {:.5f}, EM: {:.5f}, F1: {:.5f}\n".format(
                            i, j / batch_size, (time.time()-start)/60, loss, em, f1))
            
            # validate against whole validation set after each epoch.
            f1, em, prob = self.evaluate_answer(session, val, True)
            print("EM: {}, F1: {} (whole dataset)\n".format(em, f1))
            scores.append((em, f1))
            self.saver.save(session, train_dic + "/model_{}_epoch.chk".format(i+1))
            with open(train_dic + "/model_{}_epoch_prediction.json".format(i+1), "w") as f:
                json.dump(prob, f)
        
        print("EM/F1 scores after each epoch: ", scores)
