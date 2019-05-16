"""
Ori Kopel 205533151 kopelor
Shlomo Rabinovich 308432517 rabinos6
"""
import numpy as np


class Testing:
    def __init__(self, test_data, test_label, w_per, w_pa, w_svm):
        self.test_data = test_data
        self.test_label = test_label
        self.w_per = w_per
        self.w_pa = w_pa
        self.w_svm = w_svm
        self.data_count = len(test_data)

    def test(self, test):
        for t in range(len(test)):
            vec = np.array(test[t]).astype(float)
            y_h_per = np.argmax(np.dot(self.w_per, vec))
            y_h_pa = np.argmax(np.dot(self.w_pa, vec))
            y_h_svm = np.argmax(np.dot(self.w_svm, vec))
            print("perceptron: {}, svm: {}, pa: {}".format(y_h_per, y_h_svm, y_h_pa))

    def testStatistic(self, i):
        t1 = 0
        t2 = 0
        t3 = 0
        for t in range(self.data_count):
            if t % 5 != i:
                continue
            vec = np.array(self.test_data[t]).astype(float)
            y = self.test_label[t]
            t1 += self.testPerceptrom(y, vec)
            t2 += self.testPA(y, vec)
            t3 += self.testSVM(y, vec)
        s = self.data_count / 5
        return t1 / s, t2 / s, t3 / s

    def testPerceptrom(self, t, vec):
        y_hat = np.argmax(np.dot(self.w_per, vec))
        if t == y_hat:
            return 1
        else:
            return 0

    def testPA(self, t, vec):
        y_hat = np.argmax(np.dot(self.w_pa, vec))
        if t == y_hat:
            return 1
        else:
            return 0

    def testSVM(self, t, vec):
        y_hat = np.argmax(np.dot(self.w_svm, vec))
        if t == y_hat:
            return 1
        else:
            return 0
