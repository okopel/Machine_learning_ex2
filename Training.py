"""
Ori Kopel 205533151 kopelor
Shlomo Rabinovich 308432517 rabinos6
"""

import numpy as np


class Training:
    def __init__(self, t_data, t_label, params):  # clssesNum, lamda, etaPer, etaSVM, epochsPA, epochsSVM, epochPER):
        self.td = t_data  # data of training set
        self.tl = t_label  # label of training set
        self.lamda = params["lamda"]
        self.etaPer = params["etaPer"]
        self.etaSVM = params["etaSvm"]
        self.epochsOfPA = params["epochsPA"]
        self.epochsOfSVM = params["epochsSVM"]
        self.epochsOfPER = params["epochPER"]
        clssesNum = params["clssesNum"]
        ltd = len(t_data[0])
        self.w_perceptron = np.zeros((clssesNum, ltd))
        self.w_pa = np.zeros((clssesNum, ltd))
        self.w_svm = np.zeros((clssesNum, ltd))

    def train(self, index):
        # self.t_data, self.t_label = self.shuffle2np(self.t_data, self.t_label)
        for e in range(max(self.epochsOfSVM, self.epochsOfPER, self.epochsOfPA)):
            # self.td, self.tl = self.shuffle2np(self.td, self.tl)
            i = -1
            for x, y in zip(self.td, self.tl):
                i += 1
                if i % 5 == index:  # this part saves to testing
                    continue
                y = int(float(y))
                x = np.array(x).astype(float)
                if e < self.epochsOfPER:
                    self.perceptron(x, y)
                if e < self.epochsOfPA:
                    self.passiveAgressive(x, y)
                if e < self.epochsOfSVM:
                    self.svm(x, y)
        return self.w_perceptron, self.w_pa, self.w_svm

    def perceptron(self, x, y):
        y_hat = int(np.argmax(np.dot(self.w_perceptron, x)))
        if y != y_hat:
            etax = self.etaPer * x
            self.w_perceptron[y] += etax
            self.w_perceptron[y_hat] -= etax
        return y_hat

    def passiveAgressive(self, x, y):
        y_hat = int(np.argmax(np.dot(self.w_pa, x)))
        if y != y_hat:
            c = max(0, (1 - (np.dot(self.w_pa[int(y)], x)) + (np.dot(self.w_pa[int(y_hat)], x))))
            d = 2 * (np.linalg.norm(x) ** 2)
            if d == 0:
                d = 1
            t = c / d
            tx = t * x
            self.w_pa[y] += tx
            self.w_pa[y_hat] -= tx
        return y_hat

    def svm(self, x, y):
        y_hat = int(np.argmax(np.dot(self.w_svm, x)))
        etaLamda = 1 - self.etaSVM * self.lamda
        if y != y_hat:
            etax = self.etaSVM * x
            for i in range(len(self.w_svm)):
                if i == y:
                    self.w_svm[y] *= etaLamda
                    self.w_svm[y] += etax
                elif i == y_hat:
                    self.w_svm[y_hat] *= etaLamda
                    self.w_svm[y_hat] -= etax
                else:
                    self.w_svm[i] *= etaLamda
        else:
            for i in range(len(self.w_svm)):
                if i != y:
                    self.w_svm[i] *= etaLamda
        return y_hat

    @staticmethod
    def shuffle2np(x, y):
        ret = np.c_[x, y]
        np.random.shuffle(ret)
        x = ret[:, :-1]
        y = ret[:, -1]
        return x, y
