"""
Ori Kopel 205533151 kopelor
Shlomo Rabinovich 308432517 rabinos6
"""
import sys

import numpy as np
from scipy import stats

import Testing
import Training


def main():
    # get the parameter from CMD
    if len(sys.argv) < 4:
        print("ERROR!!")
        return
    data_t = sys.argv[1]
    data_label = sys.argv[2]
    test_data = sys.argv[3]
    test_label = sys.argv[4]
    data_train, data_label, test_train, test_label = orderDate(data_t, data_label, test_data, test_label)
    trainer = Training.Training(data_t, data_label, 3, 0.2, 0.25, 50)
    w_per, w_pa, w_svm = trainer.train()
    tester = Testing.Testing(test_data, test_label, w_per, w_pa, w_svm)
    t1, t2, t3 = tester.test()
    print("per:", t1, " pa:", t2, " svm:", t3)


main()


def orderDate(data_train, data_label, test_train, test_label):
    data_train = np.genfromtxt(data_train, delimiter=',', dtype="|U5")
    data_label = np.genfromtxt(data_label, delimiter=",")
    test_train = np.genfromtxt(test_train, delimiter=",", dtype="|U5")
    test_label = np.genfromtxt(test_label, delimiter=',')

    data_train = one_hot(data_train, ['M', 'F', 'I'])
    test_train = one_hot(test_train, ['M', 'F', 'I'])

    data_train = MinMax_normalize(data_train)
    test_train = MinMax_normalize(test_train)

    return data_train, data_label, test_train, test_label


# todo!
def Z_normalize(arrOfParams):
    return stats.zscore(arrOfParams)


def MinMax_normalize(arrOfParams):
    # (x-min)/(max-min)
    for i in range(len(arrOfParams[0])):
        minArg = float(min(arrOfParams[:, i]))
        maxArg = float(max(arrOfParams[:, i]))
        if minArg == maxArg:  # todo
            return 1
        for j in range(len(arrOfParams)):
            old = arrOfParams[j, i]
            arrOfParams[j, i] = (float(arrOfParams[j, i]) - minArg) / (maxArg - minArg)
        return arrOfParams


# take one box with char/type args and seperete it to bits
# when the index of this char in arrOfype is 1 and the other is 0
# notice that the original box has to be in the 1st col
def one_hot(arrOfData, arrOfTypes):
    colToAdd = len(arrOfTypes) - 1
    # adding col of zeros
    for i in range(colToAdd):
        arrOfData = np.c_[np.zeros(len(arrOfData)), arrOfData]
    for i in range(len(arrOfData)):
        for j in range(len(arrOfTypes)):
            if arrOfData[i][colToAdd] == arrOfTypes[j]:
                # delete the original sign
                arrOfData[i][colToAdd] = float(0)
                # light the true bit
                arrOfData[i][j] = float(1)
    return arrOfData
