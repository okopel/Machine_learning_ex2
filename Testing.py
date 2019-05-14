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

    def test(self):
        t1 = 0
        t2 = 0
        t3 = 0
        for t in range(self.data_count):
            vec = np.array(self.test_data[t]).astype(float)
            a1 = self.testPerceptrom(self.test_label[t], vec)
            a2 = self.testPA(self.test_label[t], vec)
            a3 = self.testSVM(self.test_label[t], vec)
            t1 += a1
            t2 += a2
            t3 += a3
            print("perseptron: {}, svm: {}, pa: {}".format(a1, a3, a2))
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
