__author__ = 'Raihan'
__email__ = 'raihan2108@gmail.com'

import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from os.path import join
from sklearn.model_selection import train_test_split

import data_iterator as dt
from utils import DataLoader
from vanilla_rnn import VanillaRNN


data_dir = 'data'
batch_size = 100
epochs = 30
state_size = 128
seq_len = 100

with open(join(data_dir, 'digg.pkl'), 'rb') as read_file:
    friend_id, reverse_friend_id, friend_network, cascade_set = pickle.load(read_file)

'''infections = []
timestamps = []

for c in cascade_set:
    t_node = cascade_set[c]['node']
    t_time = cascade_set[c]['time']

    infections.append(t_node)
    timestamps.append(t_time)

infections = pd.DataFrame(infections).fillna(0).as_matrix()
timestamps = pd.DataFrame(timestamps).fillna(0).as_matrix()
print(infections.shape)
node_size = len(friend_id)
# print(node_size, friend_network.number_of_nodes())

train_infection, test_infection, train_timestamp, test_timestamp = \
    train_test_split(infections, timestamps, test_size=0.25, random_state=42)
train_data = np.stack([train_timestamp.T, train_infection.T]).T
print(train_data.shape)
data_iterator = dt.PaddedDataIterator(train_data, 0, MARK=True, DIFF=True)'''

node_size = len(friend_id)
dl = DataLoader(join(data_dir, 'digg.pkl'), batch_size=batch_size, seq_len=seq_len, T=0, DIFF=True, MARK=True)
dl.create_batches()
tf.reset_default_graph()
with tf.Session() as sess:
    v_rnn = VanillaRNN(100, state_size, 40, lr=0.1, vertex_size=node_size, batch_size=batch_size, seq_len=seq_len)
    v_rnn.init_variable()
    v_rnn.build_graph()
    tf.global_variables_initializer().run(session=sess)
    v_rnn.train(sess=sess, data_iterator=dl, n_epochs=epochs)
