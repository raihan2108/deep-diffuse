import random
import numpy as np


class SimpleDataIterator():
    def __init__(self, df, T, MARK, DIFF=False):
        self.df = df
        self.T = T
        self.MARK = MARK
        self.DIFF = DIFF
        self.size = len(self.df)
        self.length = [len(item) for item in self.df]
        self.epochs = 0
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.df)
        self.length = [len(item) for item in self.df]
        self.cursor = 0

    def next_batch(self, n):
        if self.cursor + n-1 > self.size:
            self.epochs += 1
            self.shuffle()
        res = self.df[self.cursor:self.cursor+n]
        seqlen = self.length[self.cursor:self.cursor+n]
        self.cursor += n
        return res, seqlen


class PaddedDataIterator(SimpleDataIterator):
    def next_batch(self, n):
        if self.cursor + n > self.size:
            self.epochs += 1
            self.shuffle()
        res = self.df[self.cursor:self.cursor + n]
        seqlen = self.length[self.cursor:self.cursor + n]
        self.cursor += n

        # Pad sequences with 0s so they are all the same length
        maxlen = max(seqlen)
        # x = np.zeros([n, maxlen,1], dtype=np.float32)
        if self.MARK:
            x = np.ones([n, maxlen, 2], dtype=np.float32) * self.T
        else:
            x = np.ones([n, maxlen, 1], dtype=np.float32) * self.T
        for i, x_i in enumerate(x):
            if self.MARK:
                x_i[:seqlen[i], :] = res[i]  # asarray
            else:
                x_i[:seqlen[i], 0] = res[i]

        if self.DIFF == True:
            if self.MARK:
                xt = np.concatenate([x[:, 0:1, 0:1], np.diff(x[:, :, 0:1], axis=1)], axis=1)
                x = np.concatenate([xt, x[:, :, 1:]], axis=2)
            else:
                x = np.concatenate([x[:, 0:1, :], np.diff(x, axis=1)], axis=1)
        return x, np.asarray(seqlen)