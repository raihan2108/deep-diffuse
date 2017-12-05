import numpy as np
import tensorflow as tf


class VanillaRNN(object):

    def __init__(self, input_size, hidden_state_size, target_size, vocab_size, lr):

        self.input_size = input_size
        self.state_size = hidden_state_size
        self.target_size = target_size
        self.vocab_size = vocab_size
        self.model_init = tf.contrib.layers.xavier_initializer
        self.learning_rate =lr
        #
        # self.emb = tf.get_variable('emb', [self.vocab_size, self.hidden_state_size])
        # self.rnn_inputs = tf.nn.embedding_lookup(self.emb)

    def step(self, h_prev, x):


        W = tf.get_variable('W', shape=[self.state_size, self.state_size], initializer=self.model_init)
        U = tf.get_variable('U', shape=[self.state_size, self.state_size], initializer=self.model_init)
        b = tf.get_variable('U', shape=[self.state_size], initializer=self.model_init)
        h = tf.tanh(tf.matmul(h_prev, W) + tf.matmul(x, U) + b)

        return h

    def build_graph(self):

        tf.reset_default_graph()

        xs_ = tf.placeholder(shape=[None, None], dtype=tf.int32)
        ys_ = tf.placeholder(shape=[None], dtype=tf.int32)

        self.emb = tf.get_variable('emb', [self.vocab_size, self.hidden_state_size])
        self.rnn_inputs = tf.nn.embedding_lookup(self.emb, xs_)

        self.init_state = tf.placeholder(shape=[None, self.state_size], dtype=tf.float32, name='initial_state')
        states = tf.scan(fn=self.step, elems=tf.transpose(self.rnn_inputs, [1, 0, 2]), initializer=self.init_state)
        self.last_state = states[-1]

        self.V = tf.get_variable('V', shape=[self.state_size, self.vocab_size], initializer=self.model_init)
        self.b = tf.get_variable('b', shape=[self.state_size], initializer=self.model_init)

        state_reshaped = tf.reshape(states, [-1, self.state_size])
        logits = tf.matmul(state_reshaped, self.V) + self.b
        self.predictions = tf.nn.softmax(logits)

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, ys_)
        self.loss = tf.reduce_mean(losses)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)


    def train(self):
        pass