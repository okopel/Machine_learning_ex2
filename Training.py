"""
Ori Kopel 205533151 kopelor
Shlomo Rabinovich 308432517 rabinos6
"""

import numpy as np


class Training:
    def __init__(self, t_data, t_label, clssesNum, lamda, eta=0.25, epochs=50):
        self.t_data = t_data  # data of training set
        self.t_label = t_label  # label of training set
        self.lamda = lamda
        self.eta = eta
        self.epochs = epochs  # how many iterats
        self.w_perceptron = np.zeros((clssesNum, len(t_data[0])))
        self.w_pa = np.zeros((clssesNum, len(t_data[0])))
        self.w_svm = np.zeros((clssesNum, len(t_data[0])))

    def train(self):
        for e in range(self.epochs):
            self.t_data, self.t_label = shuffle2np(self.t_data, self.t_label)
            for x, y in zip(self.t_data, self.t_label):
                y = int(float(y))
                x = np.array(x).astype(float)
                t1 = self.perceptron(x, y)
                t2 = self.passiveAgressive(x, y)
                t3 = self.svm(x, y)
                print("real:", y, " per:", t1, " pa:", t2, " svm:", t3)
        return self.w_perceptron, self.w_pa, self.w_svm

    def perceptron(self, x, y):
        y_hat = int(np.argmax(np.dot(self.w_perceptron, x)))
        if y != y_hat:
            etax = self.eta * x
            self.w_perceptron[y] += etax
            self.w_perceptron[y_hat] -= etax
        return y_hat

    def passiveAgressive(self, x, y):
        y_hat = int(np.argmax(np.dot(self.w_pa, x)))
        if y != y_hat:
            t = (max(0, 1 - (np.dot(self.w_pa[int(y)], x)) + (np.dot(self.w_pa[int(y_hat)], x)))) / (
                    np.linalg.norm(x) ** 2)
            tx = t * x
            self.w_pa[y] += tx
            self.w_pa[y_hat] -= tx
        return y_hat

    def svm(self, x, y):
        y_hat = int(np.argmax(np.dot(self.w_svm, x)))
        if y != y_hat:
            etaLamda = 1 - self.eta * self.lamda
            etax = self.eta * x
            self.w_svm[y] *= etaLamda
            self.w_svm[y] += etax
            self.w_svm[y_hat] *= etaLamda
            self.w_svm[y_hat] -= etax
        return y_hat


def shuffle2np(x, y):  # todo is it work?
    ret = np.c_[x, y]
    np.random.shuffle(ret)
    x = ret[:, :-1]
    y = ret[:, -1]
    return x, y
