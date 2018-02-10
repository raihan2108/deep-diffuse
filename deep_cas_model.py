import numpy as np
import tensorflow as tf
# import metrics
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
import metrics
from tensorflow.python.ops import rnn, rnn_cell
from sklearn.metrics import f1_score, precision_score, recall_score


class DeepCasModel:
    def __init__(self, options, state_size, vertex_size, batch_size, seq_len, learning_rate, max_diff,
                 n_samples=20, loss_type='mse', use_att=False, node_pred=False):
        self.options = options
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.vertex_size = vertex_size
        self.loss_type = loss_type
        self.node_pred = node_pred
        self.max_diff = max_diff
        self.loss_trade_off = 0.01
        self.n_samples = n_samples
        self.n_hidden_1 = self.state_size // 2
        self.n_hidden_2 = self.state_size // 2
        self.n_steps = 10
        self.dropout_prob = 1.
        self.attention_size = self.seq_len

    def init_variables(self):
        self.input_nodes = tf.placeholder(shape=[None, None], dtype=tf.float32)
        self.input_times = tf.placeholder(shape=[None, None], dtype=tf.float32)
        self.output_node = tf.placeholder(shape=[None], dtype=tf.float32)
        self.output_time = tf.placeholder(shape=[None], dtype=tf.float32)
        self.emb = tf.get_variable('emb', initializer=tf.truncated_normal(shape=[self.vertex_size, self.state_size]))

        self.rnn_inputs_nodes = tf.nn.embedding_lookup(self.emb, tf.cast(self.input_nodes, dtype=tf.int32))
        self.rnn_inputs_times = tf.expand_dims(self.input_times, axis=-1)
        self.comb_rnn_inputs = tf.concat([self.rnn_inputs_nodes, self.rnn_inputs_times], axis=2)
        self.gru_fw_cell = rnn_cell.GRUCell(self.state_size)
        self.gru_bw_cell = rnn_cell.GRUCell(self.state_size)
        self.W_omega = tf.Variable(tf.random_normal([2 * self.state_size, self.attention_size], stddev=0.1))
        self.b_omega = tf.Variable(tf.random_normal([self.attention_size], stddev=0.1))
        self.u_omega = tf.Variable(tf.random_normal([self.attention_size], stddev=0.1))

        # self.p_step = tf.get_variable('p_step', initializer=tf.random_normal_initializer([1, self.n_steps]), dtype=tf.float32)
        # self.a_geo = tf.get_variable('a_geo', initializer=tf.random_normal_initializer([1]))

        self.Vn = tf.get_variable('Vn', initializer=tf.truncated_normal(shape=[2 * self.state_size, self.vertex_size]))
        self.bn = tf.get_variable('bn', shape=[self.vertex_size], initializer=tf.constant_initializer(0.0))

        self.dense1 = tf.get_variable('dense1',
                                      initializer=tf.truncated_normal(shape=[2 * self.state_size, self.n_hidden_1]))
        self.dense2 = tf.get_variable('dense2',
                                      initializer=tf.truncated_normal(shape=[self.n_hidden_1, self.n_hidden_2]))
        self.size_out = tf.get_variable('ws', initializer=tf.truncated_normal(shape=[self.n_hidden_2, 1]))

        self.bias1 = tf.get_variable('bias1', initializer=tf.truncated_normal(shape=[self.n_hidden_1]))
        self.bias2 = tf.get_variable('bias2', initializer=tf.truncated_normal(shape=[self.n_hidden_2]))
        self.size_bias = tf.get_variable('bs', initializer=tf.truncated_normal(shape=[1]))

    def attention(self, states):
        v = tf.tanh(tf.tensordot(states, self.W_omega, axes=[[2], [0]]) + self.b_omega)
        vu = tf.tensordot(v, self.u_omega, axes=1)
        alphas = tf.nn.softmax(vu)
        output = tf.reduce_sum(states * tf.expand_dims(alphas, -1), 1)
        return output

    def build_graph(self):
        self.init_state = tf.placeholder(shape=[None, self.state_size], dtype=tf.float32, name='initial_state')

        outputs, last_state = rnn.bidirectional_dynamic_rnn(self.gru_fw_cell, self.gru_bw_cell, self.rnn_inputs_nodes,
                                              dtype=tf.float32)
        self.states = tf.reshape(tf.stack(outputs, axis=2), shape=[self.batch_size, self.seq_len, -1])
        self.outputs = tf.transpose(self.states, [0, 1, 2])
        self.last_state = self.attention(self.outputs)
        # self.hidden_states = tf.transpose(tf.stack(last_state), [1, 0, 2])
        # (n_sequences*batch_size, n_steps, 2*n_hidden_gru)

        self.time_cost = tf.constant(0.0)
        self.node_cost = tf.constant(0.0)
        self.cost = self.calc_size_loss() + self.calc_node_loss()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def calc_node_loss(self):
        state_reshaped = tf.reshape(self.last_state, [-1, 2 * self.state_size])
        self.logits = tf.matmul(state_reshaped, self.Vn) + self.bn
        self.probs = tf.nn.softmax(self.logits)

        passable_output = tf.cast(tf.reshape(self.output_node, [-1]), dtype=tf.int32)
        self.node_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                   labels=passable_output)
        self.node_cost = tf.reduce_sum(self.node_loss)
        return self.node_cost

    def calc_size_loss(self):
        state_reshaped = tf.reshape(self.last_state, [-1, 2 * self.state_size])
        dense1 = tf.nn.relu(tf.nn.xw_plus_b(state_reshaped, self.dense1, self.bias1))
        dense2= tf.nn.relu(tf.nn.xw_plus_b(dense1, self.dense2, self.bias2))
        self.size_pred = tf.nn.relu(tf.nn.xw_plus_b(dense2, self.size_out, self.size_bias))

        self.size_cost = tf.sqrt(tf.reduce_mean(tf.pow(self.size_pred - self.output_time, 2)))
        return self.size_cost

    def predict_size(self, sess, size_label, time_seq, node_seq):
        rnn_args = { self.input_times: time_seq,
                     self.input_nodes: node_seq,
                     self.init_state: np.zeros((self.batch_size, self.state_size))
                   }
        pred_size = sess.run([self.size_pred], feed_dict=rnn_args)
        pred_size = np.reshape(pred_size[0], [self.batch_size, -1])
        ret = mean_squared_error(size_label, pred_size)
        # print(pred_size.shape, size_label.shape)
        return ret, pred_size

    def run_model(self, train_it, test_it, options):
        tf.reset_default_graph()
        self.init_variables()
        self.build_graph()
        num_batches = len(train_it)
        with tf.Session() as sess:
            tf.global_variables_initializer().run(session=sess)
            for e in range(options['epochs']):
                global_cost = 0.
                global_time_cost = 0.
                global_node_cost = 0.
                init_state = np.zeros((self.batch_size, self.state_size))
                for b in range(num_batches):
                    one_batch = train_it()
                    seq, time, seq_mask, label_n, label_t = one_batch
                    if seq.shape[0] < self.batch_size:
                        continue
                    rnn_args = {
                        self.input_nodes: seq,
                        self.input_times: time,
                        self.output_time: label_t,
                        self.output_node: label_n,
                        self.init_state: init_state
                    }
                    # print(seq.shape, time.shape, label_n.shape, label_t.shape)
                    _, cost, node_cost, time_cost, last_state = \
                        sess.run([self.optimizer, self.cost, self.node_cost, self.time_cost, self.last_state],
                                 feed_dict=rnn_args)
                    '''_, last_state, time_hat = \
                        sess.run([self.optimizer, self.last_state, self.time_hat],
                                 feed_dict=rnn_args)
                    print(last_state.shape, time_hat.shape)'''
                    global_cost += cost
                    global_node_cost += node_cost
                    global_time_cost += time_cost

                if e != 0 and e % options['disp_freq'] == 0:
                    print('[%d/%d] epoch: %d, batch: %d, train loss: %.4f, node loss: %.4f, time loss: %.4f' % (
                    e * num_batches + b, options['epochs'] * num_batches, e + 1, b + 1, global_cost, global_node_cost,
                    global_time_cost))

                if e != 0 and e % options['test_freq'] == 0:
                    scores = self.evaluate_model(sess, test_it, last_state)
                    print(scores)

    def evaluate_model(self, sess, test_it, last_state):
        test_batch_size = len(test_it)
        # y = None
        # y_prob = None
        node_scores = []
        time_scores = []
        for i in range(0, test_batch_size):
            test_batch = test_it()

            seq, time, seq_mask, label_n, label_t = test_batch
            if seq.shape[0] < self.batch_size:
                continue
            else:
                node_score, time_score = self.evaluate_batch(test_batch, sess)
                node_scores.append(node_score)
                time_scores.append(time_score)
            '''y_ = label_n
            rnn_args = {self.input_nodes: seq,
                        self.input_times: time
                        # self.init_state: np.zeros((2, self.batch_size, self.state_size))
                        }
            y_prob_ = sess.run([self.probs], feed_dict=rnn_args)

            y_prob_ = y_prob_[0]
            for j, p in enumerate(y_prob_):
                test_seq_len = test_batch[2][j]
                test_seq = test_batch[0][j][0: int(sum(test_seq_len))]
                p[test_seq.astype(int)] = 0
                y_prob_[j, :] = p / float(np.sum(p))

            if y_prob is None:
                y_prob = y_prob_
                y = y_
            else:
                y = np.concatenate((y, y_), axis=0)
                y_prob = np.concatenate((y_prob, y_prob_), axis=0)'''
        return self.get_average_score(node_scores), np.mean(np.asarray(time_scores))

    def evaluate_batch(self, test_batch, sess):
        y = None
        y_prob = None
        seq, time, seq_mask, label_n, label_t = test_batch
        y_ = label_n

        if self.node_pred:
            rnn_args = {self.input_nodes: seq,
                        self.input_times: time,
                        self.init_state: np.zeros((self.batch_size, self.state_size))
                        }
            y_prob_ = sess.run([self.probs], feed_dict=rnn_args)
            y_prob_ = y_prob_[0]
            # print(y_prob_.shape, log_lik.shape)
            for j, p in enumerate(y_prob_):
                test_seq_len = test_batch[2][j]
                test_seq = test_batch[0][j][0: int(sum(test_seq_len))]
                p[test_seq.astype(int)] = 0
                y_prob_[j, :] = p / float(np.sum(p))

            if y_prob is None:
                y_prob = y_prob_
                y = y_
            else:
                y = np.concatenate((y, y_), axis=0)
                y_prob = np.concatenate((y_prob, y_prob_), axis=0)
            node_score = metrics.portfolio(y_prob, y, k_list=[10, 50, 100])
        else:
            node_score = {}

        return node_score, 0

    def get_average_score(self, scores):
        df = pd.DataFrame(scores)
        return dict(df.mean())