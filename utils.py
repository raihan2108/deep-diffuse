import pickle
import numpy as np
import configparser
import pandas as pd
import networkx as nx
from os.path import join, isfile


def process_timestamps(timestamps):
    arr = np.asarray(timestamps)
    return list(np.diff(arr))


def load_graph(data_path):
    node_file = join(data_path, 'seen_nodes.txt')
    with open(node_file, 'r') as f:
        seen_nodes = [int(x.strip()) for x in f]

    # builds node index
    node_index = {v: i for i, v in enumerate(seen_nodes)}

    # loads graph
    '''graph_file = join(data_path, 'graph.txt')
    pkl_file = join(data_path, 'graph.pkl')

    if isfile(pkl_file):
        G = pickle.load(open(pkl_file, 'rb'))
    else:
        G = nx.Graph()
        G.name = data_path
        n_nodes = len(node_index)
        G.add_nodes_from(range(n_nodes))
        with open(graph_file, 'r') as f:
            next(f)
            for line in f:
                u, v = map(int, line.strip().split())
                if (u in node_index) and (v in node_index):
                    u = node_index[u]
                    v = node_index[v]
                    G.add_edge(u, v)
        pickle.dump(G, open(pkl_file, 'wb+'))'''

    return node_index


def load_instances(data_path, file_type, node_index, seq_len, limit, ratio=1.0, testing=False):
    max_diff = 0
    pkl_path = join(data_path, file_type + '.pkl')
    if isfile(pkl_path):
        instances = pickle.load(open(pkl_path, 'rb'))
    else:
        file_name = join(data_path, file_type + '.txt')
        instances = []
        with open(file_name, 'r') as read_file:
            for i, line in enumerate(read_file):
                query, cascade = line.strip().split(' ', 1)
                cascade_nodes = list(map(int, cascade.split(' ')[::2]))
                cascade_times = list(map(float, cascade.split(' ')[1::2]))
                if seq_len is not None:
                    cascade_nodes = cascade_nodes[:seq_len+1]
                    cascade_times = cascade_times[:seq_len+2]
                    cascade_times = process_timestamps(cascade_times)
                    assert len(cascade_nodes) == len(cascade_times)
                cascade_nodes = [node_index[x] for x in cascade_nodes]
                max_diff = max(max_diff, max(cascade_times))
                ins = process_cascade(cascade_nodes, cascade_times, testing)
                instances.extend(ins)
                if limit is not None and i == limit:
                    break
        # pickle.dump(instances, open(pkl_path, 'wb+'))
    total_samples = len(instances)
    indices = np.random.choice(total_samples, int(
        total_samples * ratio), replace=False)
    sampled_instances = [instances[i] for i in indices]
    return sampled_instances, max_diff


def process_cascade(cascade, timestamps, testing=False):
    size = len(cascade)
    examples = []
    for i, node in enumerate(cascade):
        if i == size - 1 and not testing:
            return examples
        if i < size - 1 and testing:
            continue
        prefix_c = cascade[: i + 1]
        prefix_t = timestamps[: i + 1]
        # predecessors = set(network[node]) & set(prefix_c)
        # others = set(prefix_c).difference(set(predecessors))

        '''if i == 0:
            times.extend([0.0])
        else:
            # print(i)
            times.extend([(timestamps[i-1] - timestamps[i])])'''

        if not testing:
            label_n = cascade[i+1]
            label_t = timestamps[i+1]
        else:
            label_n = None
            label_t = None

        example = {'sequence': prefix_c, 'time': prefix_t,
                   'label_n': label_n, 'label_t': label_t}

        if not testing:
            examples.append(example)
        else:
            return example


def load_params(param_file='params.ini'):
    options = {}
    config = configparser.ConfigParser()
    config.read(param_file)
    options['data_dir'] = config['general']['data_dir']
    options['dataset_name'] = config['general']['dataset_name']
    options['batch_size'] = int(config['general']['batch_size'])
    options['seq_len'] = int(config['general']['seq_len'])
    options['win_len'] = int(config['general']['win_len'])
    options['cell_type'] = str(config['general']['cell_type'])
    options['epochs'] = int(config['general']['epochs'])
    options['learning_rate'] = float(config['general']['learning_rate'])
    options['state_size'] = int(config['general']['state_size'])
    options['shuffle'] = bool(config['general']['shuffle'])
    options['embedding_size'] = int(config['general']['embedding_size'])
    options['n_samples'] = int(config['general']['n_samples'])
    options['use_attention'] = config.getboolean('general', 'use_attention')
    options['time_loss'] = str(config['general']['time_loss'])
    options['num_glimpse'] = int(config['general']['num_glimpse'])
    options['hl_size'] = int(config['general']['h_size'])
    options['hg_size'] = int(config['general']['h_size'])
    options['g_size'] = int(config['general']['g_size'])
    options['loc_dim'] = int(config['general']['loc_dim'])
    options['clipping_val'] = float(config['general']['clipping_val'])

    options['test_freq'] = int(config['general']['test_freq'])
    options['disp_freq'] = int(config['general']['disp_freq'])

    return options


def prepare_minibatch(tuples, inference=False, options=None):
    seqs = [t['sequence'] for t in tuples]
    times = [t['time'] for t in tuples]
    lengths = list(map(len, seqs))
    n_timesteps = max(lengths)
    n_samples = len(tuples)
    '''try:
        assert n_timesteps == options['seq_len']
    except AssertionError:
        print(n_timesteps, options['seq_len'])'''

    # prepare sequences data
    seqs_matrix = np.zeros((options['seq_len'], n_samples)).astype('int32')
    for i, seq in enumerate(seqs):
        seqs_matrix[: lengths[i], i] = seq
    seqs_matrix = np.transpose(seqs_matrix)

    times_matrix = np.zeros((options['seq_len'], n_samples)).astype('int32')
    for i, time in enumerate(times):
        times_matrix[: lengths[i], i] = time
    times_matrix = np.transpose(times_matrix) 
    # prepare topo-masks data
    '''topo_masks = [t['topo_mask'] for t in tuples]
    topo_masks_tensor = np.zeros(
        (n_timesteps, n_samples, n_timesteps)).astype(np.float)
    for i, topo_mask in enumerate(topo_masks):
        topo_masks_tensor[: lengths[i], i, : lengths[i]] = topo_mask'''

    # prepare sequence masks
    len_masks_matrix = np.zeros((n_timesteps, n_samples)).astype(np.float)
    for i, length in enumerate(lengths):
        len_masks_matrix[: length, i] = 1.
    len_masks_matrix = np.transpose(len_masks_matrix)

    # prepare labels data
    if not inference:
        labels_n = [t['label_n'] for t in tuples]
        labels_t = [t['label_t'] for t in tuples]
        labels_vector_n = np.array(labels_n).astype('int32')
        labels_vector_t = np.array(labels_t).astype('int32')
    else:
        labels_vector_t = None
        labels_vector_n = None

    return (seqs_matrix,
            times_matrix,
            len_masks_matrix,
            labels_vector_n, labels_vector_t)


class Loader:
    def __init__(self, data, options=None):
        self.batch_size = options['batch_size']
        self.idx = 0
        self.data = data
        self.shuffle = True
        self.n = len(data)
        self.n_words = options['node_size']
        self.indices = np.arange(self.n, dtype="int32")
        self.options = options

    def __len__(self):
        return len(self.data) // self.batch_size + 1

    def __call__(self):
        if self.shuffle and self.idx == 0:
            np.random.shuffle(self.indices)

        batch_indices = self.indices[self.idx: self.idx + self.batch_size]
        batch_examples = [self.data[i] for i in batch_indices]

        self.idx += self.batch_size
        if self.idx >= self.n:
            self.idx = 0

        return prepare_minibatch(batch_examples,
                                 inference=False,
                                 options=self.options)
