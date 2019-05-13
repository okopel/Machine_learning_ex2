"""
Ori Kopel 205533151 kopelor
Shlomo Rabinovich 308432517 rabinos6
"""

import numpy as np


class Perceptron:
    def __init__(self, t_data, t_label, clssesNum, eta=0.25, epochs=50):
        self.t_data = t_data  # data of training set
        self.t_label = t_label  # label of training set
        # self.n = len(t_data)  # set size
        self.m = len(t_data[0])  # parameters of one entry
        self.eta = eta
        self.epochs = epochs  # how many iterats
        self.classesNum = clssesNum  # how many multy class

    def train(self):
        w = np.zeros((self.classesNum, self.m))
        for e in range(self.epochs):
            self.t_data, self.t_label = shuffle2np(self.t_data, self.t_label)
            for x, y in zip(self.t_data, self.t_label):
                y_hat = np.argmax(np.dot(w, x))
                if y != y_hat:
                    etax = self.eta * np.array(x)
                    w[int(y)] += etax
                    w[int(y_hat)] -= etax
        return w


def shuffle2np(x, y):  # todo is it work?
    ret = np.c_[x, y]
    ret = np.random.mtrand.shuffle(ret)
    x = ret[:, :-1]
    y = ret[:, -1]
    return x, y
