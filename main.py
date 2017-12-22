import numpy as np
import networkx as nx
from os.path import join

import utils
# from basic_model import VanillaRNN
# from vanilla_rnn import VanillaRNN
from rnn_model import RNNModel
from lstm_model import LSTMModel

if __name__ == '__main__':
    options = utils.load_params()
    data_path = join(options['data_dir'], options['dataset_name'])
    # utils.write_seen_nodes(join(options['data_dir'], options['dataset_name']), 30)
    node_index = utils.load_graph(data_path)
    options['node_size'] = len(node_index)
    # print(nx.info(G))
    train_instances = utils.load_instances(data_path, 'train', node_index, options['seq_len'], limit=100)
    test_instances = utils.load_instances(data_path, 'test', node_index, options['seq_len'], limit=10)
    print(len(train_instances), len(test_instances))

    '''train_dt = utils.DataIterator(train_instances, options)
    # new_batch = train_dt.next_batch()
    test_dt = utils.DataIterator(test_instances, options)'''

    '''v_rnn = VanillaRNN(options['state_size'], options['node_size'], options['batch_size'], options['seq_len'],
                       options['learning_rate'])
    v_rnn.run_model(train_dt, test_dt, options)'''
    train_loader = utils.Loader(train_instances, options)
    test_loader = utils.Loader(test_instances, options)

    '''rnn_ins = RNNModel(options['state_size'], options['node_size'], options['batch_size'], options['seq_len'],
                       options['learning_rate'])
    rnn_ins.run_model(train_loader, test_loader, options)'''
    lstm_ins = LSTMModel(options['state_size'], options['node_size'], options['batch_size'], options['seq_len'],
                       options['learning_rate'])
    lstm_ins.run_model(train_loader, test_loader, options)
