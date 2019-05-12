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


# if __name__ == '__main__':
def main():
    # todo argv[0]
    # read the training set
    X = np.genfromtxt("train_x.txt", delimiter=',', dtype="|U5")
    X[X == 'M'] = 0
    X[X == 'F'] = 1
    X[X == 'I'] = 2
    Y = np.genfromtxt("train_y.txt", delimiter=",")
    print(X, Y)
    # X = Z_normalize(X)
    X = MinMax_normalize(X)
    print(X, Y)
    # perceptron(X, Y)


main()
