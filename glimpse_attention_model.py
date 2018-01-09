import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import legacy_seq2seq
from tensorflow.contrib import seq2seq

distributions = tf.contrib.distributions
import metrics


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.01)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.0, shape=shape)
  return tf.Variable(initial)


def loglikelihood(mean_arr, sampled_arr, sigma):
  mu = tf.stack(mean_arr)  # mu = [timesteps, batch_sz, loc_dim]
  sampled = tf.stack(sampled_arr)  # same shape as mu
  gaussian = distributions.Normal(mu, sigma)
  logll = gaussian.log_pdf(sampled)  # [timesteps, batch_sz, loc_dim]
  logll = tf.reduce_sum(logll, 2)
  logll = tf.transpose(logll)  # [batch_sz, timesteps]

  return logll


class GlimpseAttentionModel:
    def __init__(self, options, use_att):
        self.batch_size = options['batch_size']
        self.seq_len = options['seq_len']
        self.state_size = options['state_size']
        self.learning_rate = options['learning_rate']
        self.vertex_size = options['node_size']
        self.loss_type = options['time_loss']
        self.win_len = options['win_len']
        self.emb_size = options['embedding_size']
        self.loss_trade_off = 0.01
        self.options = options
        self.use_att = False
        '''if use_att:
            self.use_att = True
            self.attention_size = self.win_len'''

    def init_variables(self):
        self.input_nodes = tf.placeholder(shape=[None, None], dtype=tf.float32)
        self.input_times = tf.placeholder(shape=[None, None], dtype=tf.float32)

        self.output_node = tf.placeholder(shape=[None], dtype=tf.float32)
        self.output_time = tf.placeholder(shape=[None], dtype=tf.float32)

        self.emb = tf.get_variable('emb', initializer=tf.truncated_normal(shape=[self.vertex_size, self.emb_size]))
        self.Vn = tf.get_variable('Vn', initializer=tf.truncated_normal(shape=[self.state_size, self.vertex_size]))
        self.bn = tf.get_variable('bn', shape=[self.vertex_size], initializer=tf.constant_initializer(0.0))

        self.Vt = tf.get_variable('Vt', initializer=tf.truncated_normal(shape=[self.state_size, 1]))
        self.bt = tf.get_variable('bt', shape=[1], initializer=tf.constant_initializer(0.0))
        self.wt = tf.get_variable("wo", shape=[1], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer)

        if self.use_att:
            self.W_omega = tf.Variable(tf.random_normal([self.state_size, self.attention_size], stddev=0.1))
            self.b_omega = tf.Variable(tf.random_normal([self.attention_size], stddev=0.1))
            self.u_omega = tf.Variable(tf.random_normal([self.attention_size], stddev=0.1))

    def build_graph(self):
        loc_mean_arr = []
        sampled_loc_arr = []

        # with tf.variable_scope('glimpse_net'):
        gl = GlimpseNet(self.options, self.input_nodes, self.input_times, self.emb)
        # with tf.variable_scope('loc_net'):
        loc_net = LocNet(self.options)

        def get_next_input(output, i):
            loc, loc_mean = loc_net(output)
            gl_next = gl(loc)
            loc_mean_arr.append(loc_mean)
            sampled_loc_arr.append(loc)
            return gl_next

        init_loc = tf.random_uniform((self.batch_size, 1), maxval=self.seq_len-self.win_len, minval=0, dtype=tf.int32)
        init_glimpse = gl(init_loc)
        lstm_cell = tf.nn.rnn_cell.LSTMCell(self.state_size, activation=tf.nn.tanh, state_is_tuple=False)
        self.init_state = lstm_cell.zero_state(self.batch_size, tf.float32)
        inputs = [init_glimpse]
        # inputs.extend([0] * (self.options['num_glimpse']))
        self.outputs, _ = legacy_seq2seq.rnn_decoder(inputs, self.init_state, lstm_cell, loop_function=get_next_input)
        '''if self.use_att:
            self.output = self.attention(self.outputs)
        else:
            self.output = self.outputs[-1]'''

        self.cost = self.calc_node_loss() + self.loss_trade_off * self.calc_time_loss(self.output_time)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    '''def attention(self, states):
        v = tf.tanh(tf.tensordot(states, self.W_omega, axes=1) + self.b_omega)
        vu = tf.tensordot(v, self.u_omega, axes=1)
        alphas = tf.nn.softmax(vu)
        output = tf.reduce_sum(states * tf.expand_dims(alphas, -1), 1)
        return output'''

    def calc_node_loss(self):
        state_reshaped = tf.reshape(self.outputs, [-1, self.state_size])
        self.logits = tf.matmul(state_reshaped, self.Vn) + self.bn
        self.probs = tf.nn.softmax(self.logits)

        passable_output = tf.cast(tf.reshape(self.output_node, [-1]), dtype=tf.int32)
        self.node_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                   labels=passable_output)
        self.node_cost = tf.reduce_mean(self.node_loss)
        return self.node_cost

    def calc_time_loss(self, current_time):
        if self.loss_type == "intensity":
            state_reshaped = tf.reshape(self.outputs, [-1, self.state_size])
            self._hist_influence = tf.matmul(state_reshaped, self.Vt)
            self._curr_influence = self.wt * current_time
            self._rate_t = self._hist_influence + self._curr_influence + self.bt
            self._loglik = (self._rate_t + tf.exp(self._hist_influence + self.bt) * (1 / self.wt)
                     - (1 / self.wt) * tf.exp(self._rate_t))
            return - self._loglik
        elif self.loss_type == "mse":
            state_reshaped = tf.reshape(self.outputs, [-1, self.state_size])
            time_hat = tf.matmul(state_reshaped, self.Vt) + self.bt
            self.time_loss = tf.abs(tf.reshape(time_hat, [-1]) - current_time)
        self.time_cost = tf.reduce_mean(self.time_loss)
        return self.time_cost

    def run_model(self, train_it, test_it, options):
        tf.reset_default_graph()
        self.init_variables()
        self.build_graph()
        num_batches = len(train_it)
        with tf.Session() as sess:
            tf.global_variables_initializer().run(session=sess)
            for e in range(1, options['epochs']+ 1):
                global_cost = 0.
                global_time_cost = 0.
                global_node_cost = 0.
                # init_state = np.zeros((2, self.batch_size, self.state_size))
                for b in range(num_batches):
                    one_batch = train_it()
                    seq, time, seq_mask, label_n, label_t = one_batch
                    assert seq.shape == time.shape
                    if seq.shape[0] < self.batch_size:
                        continue
                    rnn_args = {
                        self.input_nodes: seq,
                        self.input_times: time,
                        self.output_time: label_t,
                        self.output_node: label_n
                        # self.init_state: init_state
                    }
                    _, cost, node_cost, time_cost = \
                        sess.run([self.optimizer, self.cost, self.node_cost, self.time_cost],
                                 feed_dict=rnn_args)
                    global_cost += cost
                    global_node_cost += node_cost
                    global_time_cost += time_cost
                    '''output = sess.run(self.outputs, feed_dict=rnn_args)
                    print(output[0].shape)'''

                if e % options['disp_freq'] == 0:
                    print('[%d/%d] epoch: %d, batch: %d, train loss: %.4f, node loss: %.4f, time loss: %.4f' % (
                    e * num_batches + b, options['epochs'] * num_batches, e + 1, b + 1, global_cost, global_node_cost,
                    global_time_cost))

                if e % options['test_freq'] == 0:
                    scores = self.evaluate_model(sess, test_it)
                    print(scores)

    def evaluate_batch(self, test_batch, sess):
        y = None
        y_prob = None
        seq, time, seq_mask, label_n, label_t = test_batch
        y_ = label_n
        rnn_args = {self.input_nodes: seq,
                    self.input_times: time
                    # self.init_state: np.zeros((2, self.batch_size, self.state_size))
                    }
        y_prob_ = sess.run([self.probs], feed_dict=rnn_args)
        y_prob_ = y_prob_[0]
        # print(y_prob_.shape)
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
        return metrics.portfolio(y_prob, y, k_list=[10, 50, 100])

    def get_average_score(self, scores):
        df = pd.DataFrame(scores)
        return dict(df.mean())

    def evaluate_model(self, sess, test_it):
        test_batch_size = len(test_it)
        # y = None
        # y_prob = None
        scores = []

        for i in range(0, test_batch_size):
            test_batch = test_it()

            seq, time, seq_mask, label_n, label_t = test_batch
            if seq.shape[0] < self.batch_size:
                continue
            else:
                score = self.evaluate_batch(test_batch, sess)
                scores.append(score)
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
        return self.get_average_score(scores)
        # return metrics.portfolio(y_prob, y, k_list=[10, 50, 100])


class GlimpseNet:
    def __init__(self, options, input_node_ph, input_time_ph, emb):
        self.original_size = options['seq_len']
        self.win_len = options['win_len']
        self.batch_size = options['batch_size']
        self.seq_len = options['seq_len']
        self.vertex_size = options['node_size']
        self.state_size = options['state_size']
        self.emb_size = options['embedding_size']
        self.sensor_size = self.win_len * (self.emb_size + 1)
        self.emb = emb

        self.hg_size = options['hg_size']
        self.hl_size = options['hl_size']
        self.g_size = options['g_size']
        self.loc_dim = options['loc_dim']

        self.input_node_ph = input_node_ph
        self.input_time_ph = input_time_ph
        self.init_variables()

    def init_variables(self):
        self.w_g0 = weight_variable((self.sensor_size, self.hg_size))
        self.b_g0 = bias_variable((self.hg_size,))
        self.w_l0 = weight_variable((self.loc_dim, self.hl_size))
        self.b_l0 = bias_variable((self.hl_size,))
        self.w_g1 = weight_variable((self.hg_size, self.g_size))
        self.b_g1 = bias_variable((self.g_size,))
        self.w_l1 = weight_variable((self.hl_size, self.g_size))
        self.b_l1 = weight_variable((self.g_size,))

    def get_glimpse(self, loc):
        out_node = []
        out_time = []
        for i in range(0, self.batch_size):
            '''if loc[i] + self.win_size > self.seq_len:
                f_s = self.seq_len - loc[i]
            else:
                f_s = self.win_size'''
            f_s = self.win_len
            begin = tf.cast(loc[i][0], dtype=tf.int32)
            end = tf.cast(loc[i][0], dtype=tf.int32) + f_s
            # t_node = tf.slice(self.input_node_ph[i, :], tf.cast(loc[i], dtype=tf.int32), [f_s])
            # t_time = tf.slice(self.input_time_ph[i, :], tf.cast(loc[i], dtype=tf.int32), [f_s])
            t_node = self.input_node_ph[i, begin:end]
            t_time = self.input_time_ph[i, begin:end]
            # assert t_node.shape[0] == self.win_len
            # assert t_time.shape[0] == self.win_len
            '''if t_node.shape[0] < self.win_size:
                zero_padding = tf.zeros([self.win_size - tf.cast(t_node.shape[0], dtype=tf.int32)], dtype=tf.float32)
                t_node = tf.concat([t_node, zero_padding], axis=0)
                t_time = tf.concat([t_node, zero_padding], axis=0)'''
            out_node.append(t_node)
            out_time.append(t_time)
        out_node = tf.convert_to_tensor(out_node)
        out_time = tf.convert_to_tensor(out_time)
        # out_node = tf.stack(out_node)
        # out_time = tf.stack(out_time)
        out_node = tf.reshape(out_node, [tf.shape(loc)[0], -1])
        out_time = tf.reshape(out_time, [tf.shape(loc)[0], -1])
        return out_node, out_time

    def __call__(self, loc):
        glimpse_input_node, glimpse_input_time = self.get_glimpse(loc)
        # self.emb = tf.get_variable('emb', initializer=tf.truncated_normal(shape=[self.vertex_size, self.emb_size]))
        self.rnn_inputs_nodes = tf.nn.embedding_lookup(self.emb, tf.cast(glimpse_input_node, dtype=tf.int32))
        self.rnn_inputs_times = tf.expand_dims(glimpse_input_time, axis=-1)
        self.comb_glimpse_inputs = tf.concat([self.rnn_inputs_nodes, self.rnn_inputs_times], axis=2)
        self.comb_glimpse_inputs = tf.reshape(self.comb_glimpse_inputs, [self.batch_size, -1])
        '''glimpse_input = tf.reshape(glimpse_input,
                                   (tf.shape(loc)[0], self.sensor_size))'''
        g = tf.nn.relu(tf.nn.xw_plus_b(self.comb_glimpse_inputs, self.w_g0, self.b_g0))
        g = tf.nn.xw_plus_b(g, self.w_g1, self.b_g1)
        l = tf.nn.relu(tf.nn.xw_plus_b(tf.cast(loc, tf.float32), self.w_l0, self.b_l0))
        l = tf.nn.xw_plus_b(l, self.w_l1, self.b_l1)
        g = tf.nn.relu(g + l)
        return g


class LocNet:
    def __init__(self, options):
        self.loc_dim = options['loc_dim']
        self.input_dim = options['state_size']
        self.seq_len = options['seq_len']
        self.win_len = options['win_len']
        self.loc_std = 0.2
        self._sampling = True

        self.init_variables()

    def init_variables(self):
        self.w = weight_variable((self.input_dim, self.loc_dim))
        self.b = bias_variable((self.loc_dim,))

    def __call__(self, input):
        mean = tf.clip_by_value(tf.nn.xw_plus_b(input, self.w, self.b), 0, self.seq_len-self.win_len)
        mean = tf.stop_gradient(mean)
        if self._sampling:
            loc = mean + tf.random_normal(
                (tf.shape(input)[0], self.loc_dim), stddev=self.loc_std)
            loc = tf.clip_by_value(loc, 0, self.seq_len-self.win_len)
        else:
            loc = mean
        loc = tf.stop_gradient(loc)
        return loc, mean

    @property
    def sampling(self):
        return self._sampling

    @sampling.setter
    def sampling(self, sampling):
        self._sampling = sampling