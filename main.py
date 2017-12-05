import pickle
from os.path import join

data_dir = 'data'
batch_size = 50
epochs = 50

with open(join(data_dir, 'digg.pkl'), 'rb') as read_file:
    friend_id, reverse_friend_id, friend_network, cascade_set = pickle.load(read_file)