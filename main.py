"""
Ori Kopel 205533151 kopelor
Shlomo Rabinovich 308432517 rabinos
"""
import sys

import numpy as np
from scipy import stats

import perceptron


# todo!
def Z_normalize(arrOfParams):
    # return (arrOfParams - np.mean) / np.std(np.divide())
    print("norm\n")
    print(stats.zscore(arrOfParams))
    return stats.zscore(arrOfParams)


def MinMax_normalize(arrOfParams):
    # (x-min)/(max-min)
    for i in range(len(arrOfParams[0])):
        minArg = float(min(arrOfParams[:, i]))
        maxArg = float(max(arrOfParams[:, i]))
        if minArg == maxArg:  # todo
            return
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


def main():
    # get the parameter from CMD
    if len(sys.argv) < 4:
        print("ERROR!!")
        return

    # read the training set
    X = np.genfromtxt(sys.argv[1], delimiter=',', dtype="|U5")
    Y = np.genfromtxt(sys.argv[2], delimiter=",")
    x_test = np.genfromtxt(sys.argv[3], delimiter=",", dtype="|U5")
    y_test = np.genfromtxt(sys.argv[4], delimiter=',')
    # convert the first col to one hot (00..00100..) at the class place
    X = one_hot(X, ['M', 'F', 'I'])
    x_test = one_hot(x_test, ['M', 'F', 'I'])
    # X = Z_normalize(X)
    # normalize all the args to args between 0 to 1
    X = MinMax_normalize(X)
    x_test = MinMax_normalize(x_test)
    X, Y = perceptron.shuffle2np(X, Y)
    perceptronVec = perceptron.perceptron(X, Y)
    # svmVec = svm.svm(X, Y)
    # paVec = Passive_Aggressive.pa(X, Y)
    # printTest([perceptronVec, svmVec, paVec], ["perceptron", "svn", "pa"], X, Y)
    print("suc:", testing(perceptronVec, x_test, y_test))


def printTest(wArr, nameArr, X, Y):
    res = np.array(len(X), len(wArr))
    for i in range(len(X)):
        for w in range(len(wArr)):
            res[i][w] = testing(wArr[w], X, Y)
    print(res)


def testing(w, X_train, Y_train):
    m = 0
    n = len(X_train)
    # check all the training set after our training
    for t in range(0, n):
        vec = np.array(X_train[t]).astype(np.float64)
        y_hat = np.argmax(np.dot(w, vec))
        if Y_train[t] == y_hat:
            m += 1
    return m / n


main()
