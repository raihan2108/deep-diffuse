import numpy as np
import tensorflow as tf
import metrics


class RNNModel:
    def __init__(self, state_size, vertex_size, batch_size, seq_len, learning_rate, loss_type='mse', use_att=False):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.vertex_size = vertex_size
        self.loss_type = loss_type
        self.loss_trade_off = 0.01
        self.use_att = False
        if use_att:
            self.use_att = True
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

    def step(self, h_prev, x):
        W = tf.get_variable('W', shape=[self.state_size, self.state_size], initializer=tf.orthogonal_initializer())
        U = tf.get_variable('U', shape=[x.shape[-1], self.state_size], initializer=tf.orthogonal_initializer())
        b = tf.get_variable('b', shape=[self.state_size], initializer=tf.constant_initializer(0.0))
        h = tf.tanh(tf.matmul(h_prev, W) + tf.matmul(x, U) + b)
        return h

    def attention(self, states):
        v = tf.tanh(tf.tensordot(states, self.W_omega, axes=[[2], [0]]) + self.b_omega)
        vu = tf.tensordot(v, self.u_omega, axes=1)
        alphas = tf.nn.softmax(vu)
        output = tf.reduce_sum(states * tf.expand_dims(alphas, -1), 1)
        return output

    def build_graph(self):
        self.init_state = tf.placeholder(shape=[None, self.state_size], dtype=tf.float32, name='initial_state')
        states = tf.scan(fn=self.step, elems=tf.transpose(self.comb_rnn_inputs, [1, 0, 2]),
                         initializer=self.init_state)
        self.states = tf.transpose(states, [1, 0, 2])
        if self.use_att:
            self.last_state = self.attention(self.states)
        else:
            self.last_state = self.states[:, -1, :]

        self.cost = self.calc_node_loss() + self.loss_trade_off * self.calc_time_loss(self.output_time)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def calc_node_loss(self):
        state_reshaped = tf.reshape(self.last_state, [-1, self.state_size])
        self.logits = tf.matmul(state_reshaped, self.Vn) + self.bn
        self.probs = tf.nn.softmax(self.logits)

        passable_output = tf.cast(tf.reshape(self.output_node, [-1]), dtype=tf.int32)
        self.node_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                   labels=passable_output)
        self.node_cost = tf.reduce_sum(self.node_loss)
        return self.node_cost

    def calc_time_loss(self, current_time):
        if self.loss_type == "intensity":
            state_reshaped = tf.reshape(self.last_state, [-1, self.state_size])
            self.hist_influence = tf.matmul(state_reshaped, self.Vt)
            self.curr_influence = self.wt * current_time
            self.rate_t = self.hist_influence + self.curr_influence + self.bt
            self.loglik = (self.rate_t + tf.exp(self.hist_influence + self.bt) * (1 / self.wt)
                           - (1 / self.wt) * tf.exp(self.rate_t))
            return -self.loglik
        elif self.loss_type == "mse":
            state_reshaped = tf.reshape(self.last_state, [-1, self.state_size])
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
        y = None
        y_prob = None
        for i in range(0, test_batch_size):
            test_batch = test_it()
            seq, time, seq_mask, label_n, label_t = test_batch
            if seq.shape[0] < self.batch_size:
                continue
            y_ = label_n
            rnn_args = {self.input_nodes: seq,
                        self.input_times: time,
                        self.init_state: np.zeros((self.batch_size, self.state_size))}
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
                y_prob = np.concatenate((y_prob, y_prob_), axis=0)

        return metrics.portfolio(y_prob, y, k_list=[10, 50, 100])