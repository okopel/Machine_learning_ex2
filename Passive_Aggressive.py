"""
Ori Kopel 205533151 kopelor
Shlomo Rabinovich 308432517 rabinos
"""

import numpy as np


class PassiveAgressive:
    def __init__(self, t_data, t_label, clssesNum, eta=0.25, epochs=50):
        self.t_data = t_data  # data of training set
        self.t_label = t_label  # label of training set
        self.n = len(t_data)  # set size
        self.m = len(t_data[0])  # parameters of one entry
        self.eta = eta
        self.epochs = epochs  # how many iterates
        self.classesNum = clssesNum  # how many multi class

    def train(self):
        w = np.zeros((self.classesNum, self.m))
        for e in range(self.epochs):
            # todo shuffle?
            for x, y in zip(self.t_data, self.t_label):
                x = np.array(x)
                y_hat = np.argmax(np.dot(w, x))
                if y != y_hat:
                    t = (max(0, 1 - (np.dot(w[int(y)], x)) + (np.dot(w[int(y_hat)], x)))) / (np.linalg.norm(x) ** 2)
                    w[int(y)] += (t * x)
                    w[int(y_hat)] -= (t * x)
        return w
