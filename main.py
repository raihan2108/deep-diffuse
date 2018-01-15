import numpy as np
import networkx as nx
from os.path import join

import utils
import data_utils as du
# from basic_model import VanillaRNN
# from vanilla_rnn import VanillaRNN
from rnn_model import RNNModel
from lstm_model import LSTMModel
from glimpse_attention_model import GlimpseAttentionModel

if __name__ == '__main__':
    options = utils.load_params()
    data_path = join(options['data_dir'], options['dataset_name'])
    # utils.write_seen_nodes(join(options['data_dir'], options['dataset_name']), 30)
    node_index = utils.load_graph(data_path)
    options['node_size'] = len(node_index)
    # print(nx.info(G))
    train_instances, max_diff_train = utils.load_instances(data_path, 'train', node_index, options['seq_len'], limit=-1)
    test_instances, max_diff_test = utils.load_instances(data_path, 'test', node_index, options['seq_len'], limit=-1)
    options['max_diff'] = max_diff_train
    print(len(train_instances), len(test_instances))

    train_loader = utils.Loader(train_instances, options)
    test_loader = utils.Loader(test_instances, options)
    if options['cell_type'] == 'rnn':
        print('running rnn model')
        print('using attention:' + str(options['use_attention']))
        print('node pred:' + str(options['node_pred']))
        rnn_ins = RNNModel(options['state_size'], options['node_size'], options['batch_size'], options['seq_len'],
                           options['learning_rate'], max_diff_train, loss_type=options['time_loss'],
                           use_att=options['use_attention'],
                           node_pred=options['node_pred'])
        rnn_ins.run_model(train_loader, test_loader, options)
    elif options['cell_type'] == 'lstm':
        print('running lstm model')
        print('using attention:' + str(options['use_attention']))
        print('node pred:' + str(options['node_pred']))
        lstm_ins = LSTMModel(options['state_size'], options['node_size'], options['batch_size'], options['seq_len'], options['win_len'],
                           options['learning_rate'], max_diff_train, loss_type=options['time_loss'],
                             use_att=options['use_attention'],
                             node_pred=options['node_pred'])
        lstm_ins.run_model(train_loader, test_loader, options)
    elif options['cell_type'] == 'glimpse':
        print('running glimpse attention model')
        print('using attention:' + str(options['use_attention']))
        glimpse_ins = GlimpseAttentionModel(options, options['use_attention'])
        glimpse_ins.run_model(train_loader, test_loader, options)