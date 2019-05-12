import numpy as np
import scipy as scipy

from perceptron import perceptron


def Z_normalize(arrOfParams):
    # return (arrOfParams - np.mean) / np.std(np.divide())
    return scipy.stats.mstats.zscore(arrOfParams)


# if __name__ == '__main__':
def main():
    # todo argv[0]
    # , dtype="|U5"
    X = np.genfromtxt("train_x.txt", delimiter=',', dtype="|U5")
    X[X == 'M'] = 0
    X[X == 'F'] = 1
    X[X == 'I'] = 2
    Y = np.genfromtxt("train_y.txt", delimiter=",", dtype="|U5")
    print(X, Y)
    X = Z_normalize(X)
    print("after norm\n")
    print(X, Y)
    perceptron(X, Y)


main()
