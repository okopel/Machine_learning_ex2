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

    def getWPA(self):
        return self.w_pa

    def getWPER(self):
        return self.w_per

    def getWSVM(self):
        return self.w_svm

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

    @staticmethod
    def testerSubmit(tester, test):
        for t in range(len(test)):
            vec = np.array(test[t]).astype(float)
            pa = np.zeros(3)
            svm = np.zeros(3)
            per = np.zeros(3)
            for i in range(5):
                paRes = np.argmax(np.dot(tester[i].getWPA(), vec))
                if paRes == 0:
                    pa[0] += 1
                elif paRes == 1:
                    pa[1] += 1
                elif paRes == 2:
                    pa[2] += 1
                svmRes = np.argmax(np.dot(tester[i].getWSVM(), vec))
                if svmRes == 0:
                    svm[0] += 1
                elif svmRes == 1:
                    svm[1] += 1
                elif svmRes == 2:
                    svm[2] += 1
                perRes = np.argmax(np.dot(tester[i].getWPER(), vec))
                if perRes == 0:
                    per[0] += 1
                elif perRes == 1:
                    per[1] += 1
                elif perRes == 2:
                    per[2] += 1
            print("perceptron: {}, svm: {}, pa: {}".format(np.argmax(per), np.argmax(svm), np.argmax(pa)))
