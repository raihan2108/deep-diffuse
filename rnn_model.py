import numpy as np
import tensorflow as tf
import metrics

class RNNModel:
    def __init__(self, state_size, vertex_size, batch_size, seq_len, learning_rate):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.vertex_size = vertex_size

    def init_variables(self):
        self.input_nodes = tf.placeholder(shape=[None, None], dtype=tf.float32)
        self.output_nodes = tf.placeholder(shape=[None], dtype=tf.float32)
        self.emb = tf.get_variable('emb', initializer=tf.truncated_normal(shape=[self.vertex_size, self.state_size]))
        self.rnn_inputs = tf.nn.embedding_lookup(self.emb, tf.cast(self.input_nodes, dtype=tf.int32))

        self.Vo = tf.get_variable('Vo', initializer=tf.truncated_normal(shape=[self.state_size, self.vertex_size]))
        self.bo = tf.get_variable('bo', shape=[self.vertex_size], initializer=tf.constant_initializer(0.0))

    def step(self, h_prev, x):
        W = tf.get_variable('W', shape=[self.state_size, self.state_size], initializer=tf.orthogonal_initializer())
        U = tf.get_variable('U', shape=[self.state_size, self.state_size], initializer=tf.orthogonal_initializer())
        b = tf.get_variable('b', shape=[self.state_size], initializer=tf.constant_initializer(0.0))
        h = tf.tanh(tf.matmul(h_prev, W) + tf.matmul(x, U) + b)
        return h

    def build_graph(self):
        self.init_state = tf.placeholder(shape=[None, self.state_size], dtype=tf.float32, name='initial_state')
        states = tf.scan(fn=self.step, elems=tf.transpose(self.rnn_inputs, [1, 0, 2]), initializer=self.init_state)
        self.last_state = states[-1]

        state_reshaped = tf.reshape(self.last_state, [-1, self.state_size])
        self.logits = tf.matmul(state_reshaped, self.Vo) + self.bo
        self.probs = tf.nn.softmax(self.logits)

        passable_output = tf.cast(tf.reshape(self.output_nodes, [-1]), dtype=tf.int32)
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                   labels=passable_output)
        # self.loss = tf.reduce_mean(loss)
        self.cost = tf.reduce_sum(self.loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

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
                    seq, seq_mask, label = one_batch
                    rnn_args = {
                        self.input_nodes: seq,
                        self.output_nodes: label,
                        self.init_state: init_state
                    }
                    _, node_cost, last_state, probs = sess.run([self.optimizer, self. cost, self.last_state, self.probs], feed_dict=rnn_args)
                    # print(last_state.shape, probs.shape)
                    # global_cost += cost
                    global_node_cost += node_cost
                    # global_time_cost += time_loss

                print('[%d/%d] epoch: %d, batch: %d, train loss: %.4f' % (
                e * num_batches + b, options['epochs'] * num_batches, e + 1, b + 1, global_node_cost))
                scores = self.evaluate_model(sess, test_it, last_state)
                print(scores)

    def evaluate_model(self, sess, test_it, last_state):
        test_batch_size = len(test_it)
        y = None
        y_prob = None
        for i in range(0, test_batch_size):
            test_batch = test_it()
            seq, seq_mask, label = test_batch
            y_ = label
            rnn_args = {self.input_nodes: seq,
                        self.init_state: last_state}
            y_prob_ = sess.run([self.probs], feed_dict=rnn_args)

            y_prob_ = y_prob_[0]
            for j, p in enumerate(y_prob_):
                test_seq_len = test_batch[2][j]
                test_seq = test_batch[0][j][0: test_seq_len]
                p[test_seq.astype(int)] = 0
                y_prob_[j, :] = p / float(np.sum(p))

            if y_prob is None:
                y_prob = y_prob_
                y = y_
            else:
                y = np.concatenate((y, y_), axis=0)
                y_prob = np.concatenate((y_prob, y_prob_), axis=0)

        return metrics.portfolio(y_prob, y, k_list=[10, 50, 100])