import sys

import numpy as np
from scipy import stats


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
    # the parameter
    if len(sys.argv) != 4:
        print("ERROR!!")
        return

    # read the training set
    Y = np.genfromtxt(sys.argv[2], delimiter=",")
    X = np.genfromtxt(sys.argv[1], delimiter=',', dtype="|U5")
    # convert the first col to one hot (00..00100..) at the class place
    X = one_hot(X, ['M', 'F', 'I'])
    # X = Z_normalize(X)
    # normalize al the args to args between 0 to 1
    X = MinMax_normalize(X)
    print(X)
    # print(X, Y)
    # perceptron(X, Y)


main()
