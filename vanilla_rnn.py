__author__ = 'Raihan'
__email__ = 'raihan2108@gmail.com'


import time
import numpy as np
import tensorflow as tf
from os.path import join


class VanillaRNN(object):

    def __init__(self, input_size, hidden_state_size, target_size, vertex_size, batch_size, seq_len, lr):

        self.input_size = input_size
        self.state_size = hidden_state_size
        self.target_size = target_size
        self.vocab_size = vertex_size
        self.model_init = tf.contrib.layers.xavier_initializer
        self.learning_rate =lr
        self.batch_size = batch_size
        self.seq_len = seq_len
        #
        # self.emb = tf.get_variable('emb', [self.vocab_size, self.hidden_state_size])
        # self.rnn_inputs = tf.nn.embedding_lookup(self.emb)

    def step(self, h_prev, x):
        W = tf.get_variable('W', shape=[self.state_size, self.state_size], initializer=self.model_init())
        U = tf.get_variable('U', shape=[self.state_size, self.state_size], initializer=self.model_init())
        b = tf.get_variable('b', shape=[self.state_size], initializer=self.model_init())
        h = tf.tanh(tf.matmul(h_prev, W) + tf.matmul(x, U) + b)

        return h

    def init_variable(self):
        self.xs_ = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.ys_ = tf.placeholder(shape=[None, None], dtype=tf.int32)

        self.emb = tf.get_variable('emb', [self.vocab_size, self.state_size])
        self.rnn_inputs = tf.nn.embedding_lookup(self.emb, self.xs_)
        self.Vo = tf.get_variable('Vo', shape=[self.state_size, self.vocab_size], initializer=self.model_init())
        self.bo = tf.get_variable('bo', shape=[self.vocab_size], initializer=tf.constant_initializer(0.0))

    def build_graph(self):
        self.init_state = tf.placeholder(shape=[None, self.state_size], dtype=tf.float32, name='initial_state')
        states = tf.scan(fn=self.step, elems=tf.transpose(self.rnn_inputs, [1, 0, 2]), initializer=self.init_state)
        self.last_state = states[-1]

        state_reshaped = tf.reshape(states, [-1, self.state_size])
        self.logits = tf.matmul(state_reshaped, self.Vo) + self.bo
        self.predictions = tf.nn.softmax(self.logits)

        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=tf.reshape(self.ys_, [-1]))
        # self.loss = tf.reduce_mean(loss)
        self.cost = tf.reduce_sum(self.loss) / self.batch_size / self.seq_len
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        tf.summary.histogram('logits', self.logits)
        tf.summary.histogram('loss', self.loss)
        tf.summary.scalar('cost', self.cost)

    def train(self, sess, data_iterator, n_epochs):
        total_loss = 0.0

        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(join('saved_model', time.strftime("%Y-%m-%d-%H-%M-%S")))
        writer.add_graph(sess.graph)
        saver = tf.train.Saver(tf.global_variables())
        for e in range(0, n_epochs):
            data_iterator.reset_batch_pointer()
            init_state = np.zeros((self.batch_size, self.state_size))
            for b in range(0, data_iterator.num_batches):
                x, y = data_iterator.next_batch_inf()
                if sum(x) == 0 or sum(y) == 0:
                    continue
                rnn_args = {self.xs_: x, self.ys_: y, self.init_state: init_state}
                summ, _, train_cost, last_state = sess.run([summaries, self.optimizer, self.cost, self.last_state], feed_dict=rnn_args)

                writer.add_summary(summ, e * data_iterator.num_batches + b)
                if ((e * data_iterator.num_batches + b) % 1000 == 0) or \
                        ((e == self.batch_size-1) and (b == data_iterator.num_batches - 1)):
                    checkpoint_path = join('saved_model', 'model.ckpt')
                    saver.saver(sess, checkpoint_path, global_step=e * data_iterator.num_batches + b)
                    print('model saved to %s' % checkpoint_path)

                print('[%d/%d] epoch: %d, batch: %d, train loss: %.4f' % (e * data_iterator.num_batches + b, n_epochs * data_iterator.num_batches
                      , e + 1, b + 1, train_cost))
                # total_loss += train_cost
            # print('total_loss %.4f' % total_loss)
