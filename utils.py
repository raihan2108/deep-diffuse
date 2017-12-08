import pickle
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataLoader():

    def __init__(self, data_file, batch_size, seq_len, T, DIFF, MARK):
        self.data_path = data_file
        self.MARK = MARK
        self.DIFF = DIFF
        self.epochs = 0
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.T = T
        self.reset_batch_pointer()
        # self.shuffle()

        with open(data_file, 'rb') as read_file:
            friend_id, reverse_friend_id, friend_network, cascade_set = pickle.load(read_file)
        infections = []
        timestamps = []

        for c in cascade_set:
            t_node = cascade_set[c]['node']
            t_time = cascade_set[c]['time']

            infections.append(t_node)
            timestamps.append(t_time)

        infections = pd.DataFrame(infections).fillna(0).as_matrix()
        timestamps = pd.DataFrame(timestamps).fillna(0).as_matrix()

        data = np.stack([infections.T, timestamps.T]).T
        random.shuffle(data)

        self.infections = data[:, :, 0]
        self.timestamps = data[:, :, 1]
        self.size = data.shape

        self.train_infection, self.test_infection, self.train_timestamp, self.test_timestamp = \
            train_test_split(infections, timestamps, test_size=0.25, random_state=42)
        # self.train_data = np.stack([train_timestamp.T, train_infection.T]).T

        self.size = self.train_infection.shape
        self.length = [len(item) for item in self.train_infection]
        print(self.size)

    def create_batches(self):
        self.train_data = np.stack([self.train_timestamp.T, self.train_infection.T]).T
        self.num_batches = (self.size[0] * self.size[1]) // (self.batch_size * self.seq_len)

        self.inf_tensor = self.infections.ravel()[: self.num_batches * self.batch_size * self.seq_len]
        self.time_tensor = self.timestamps.ravel()[: self.num_batches * self.batch_size * self.seq_len]

        x_inf_data = self.inf_tensor
        y_inf_data = np.copy(self.inf_tensor)
        y_inf_data[:-1] = x_inf_data[1:]
        y_inf_data[-1] = x_inf_data[0]

        x_time_data = self.time_tensor
        y_time_data = np.copy(self.time_tensor)
        y_time_data[:-1] = x_time_data[1:]
        y_time_data[-1] = x_time_data[0]

        self.x_batches_inf = np.split(x_inf_data.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches_inf = np.split(y_inf_data.reshape(self.batch_size, -1), self.num_batches, 1)
        self.x_batches_time = np.split(x_time_data.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches_time = np.split(y_time_data.reshape(self.batch_size, -1), self.num_batches, 1)

        np.save('saved_model/test_infection.npy', self.test_infection)
        np.save('saved_model/test_timestamp.npy', self.test_timestamp)

        del self.test_timestamp
        del self.test_infection
        del self.train_infection
        del self.train_timestamp

    def next_batch_inf(self):
        x, y = self.x_batches_inf[self.pointer], self.y_batches_inf[self.pointer]
        self.pointer += 1
        return x, y

    def next_batch_time(self):
        x, y = self.x_batches_time[self.pointer], self.y_batches_time[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0