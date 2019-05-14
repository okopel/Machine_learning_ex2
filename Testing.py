"""
Ori Kopel 205533151 kopelor
Shlomo Rabinovich 308432517 rabinos6
"""
import numpy as np


class Testing:
    def __init__(self, test_data, w_per, w_pa, w_svm):
        self.test_data = test_data
        self.w_per = w_per
        self.w_pa = w_pa
        self.w_svm = w_svm
        self.data_count = len(test_data)

    def test(self):
        for t in range(self.data_count):
            vec = np.array(self.test_data[t]).astype(float)
            y_h_per = np.argmax(np.dot(self.w_per, vec))
            y_h_pa = np.argmax(np.dot(self.w_pa, vec))
            y_h_svm = np.argmax(np.dot(self.w_svm, vec))
            print("perceptron: {}, svm: {}, pa: {}".format(y_h_per, y_h_svm, y_h_pa))

    def testStatistic(self, test_label):
        t1 = 0
        t2 = 0
        t3 = 0
        for t in range(self.data_count):
            vec = np.array(self.test_data[t]).astype(float)
            a1 = self.testPerceptrom(test_label[t], vec)
            a2 = self.testPA(test_label[t], vec)
            a3 = self.testSVM(test_label[t], vec)
            t1 += a1
            t2 += a2
            t3 += a3
        t1 /= self.data_count
        t2 /= self.data_count
        t3 /= self.data_count
        return t1, t2, t3

    def testPerceptrom(self, y, vec):
        y_hat = np.argmax(np.dot(self.w_per, vec))
        if y == y_hat:
            return 1
        return 0

    def testPA(self, y, vec):
        y_hat = np.argmax(np.dot(self.w_pa, vec))
        if y == y_hat:
            return 1
        return 0

    def testSVM(self, y, vec):
        y_hat = np.argmax(np.dot(self.w_svm, vec))
        if y == y_hat:
            return 1
        return 0
