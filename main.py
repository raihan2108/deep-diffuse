import pickle
import numpy as np
from os.path import join
from sklearn.model_selection import train_test_split

import data_iterator as dt


data_dir = 'data'
batch_size = 50
epochs = 50

with open(join(data_dir, 'digg.pkl'), 'rb') as read_file:
    friend_id, reverse_friend_id, friend_network, cascade_set = pickle.load(read_file)

infections = []
timestamps= []

for c in cascade_set:
    t_node = cascade_set[c]['node']
    t_time = cascade_set[c]['time']

    infections.append(t_node)
    timestamps.append(t_time)

infections = np.asarray(infections)
timestamps = np.asarray(timestamps)
# print(infections.shape)
node_size = len(friend_id)
# print(node_size, friend_network.number_of_nodes())

train_infection, test_infection, train_timestamp, test_timestamp = \
    train_test_split(infections, timestamps, test_size=0.25, random_state=42)
train_data = np.stack([train_timestamp.T, train_infection.T])

data_iterator = dt.PaddedDataIterator(train_data, 0, MARK=True, DIFF=True)