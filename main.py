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


def one_hot(arrOfData, arrOfTypes):
    colToAdd = len(arrOfTypes) - 1
    # adding col of zeros
    for i in range(colToAdd):
        arrOfData = np.c_[np.zeros(len(arrOfData)), arrOfData]
    for i in range(len(arrOfData)):
        for j in range(len(arrOfTypes)):
            if arrOfData[i][colToAdd] == arrOfTypes[j]:
                arrOfData[i][colToAdd] = float(0)
                arrOfData[i][j] = float(1)
    return arrOfData


# if __name__ == '__main__':
def main():
    # todo argv[0]
    # read the training set
    Y = np.genfromtxt("train_y.txt", delimiter=",")
    X = np.genfromtxt("train_x.txt", delimiter=',', dtype="|U5")
    # convert the first col to one hot (00..00100..) at the class place
    X = one_hot(X, ['M', 'F', 'I'])
    # X = Z_normalize(X)
    # normalize al the args to args between 0 to 1
    X = MinMax_normalize(X)
    print(X)
    # print(X, Y)
    # perceptron(X, Y)


main()
