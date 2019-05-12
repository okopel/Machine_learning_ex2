"""
Ori Kopel 205533151 kopelor
Shlomo Rabinovich 308432517 rabinos
"""

import numpy as np
from sklearn.utils import shuffle


def perceptron(X_train: np, Y_train: np) -> object:
    # num of args in the training set
    m = len(X_train)
    # num of args in one example
    d = len(X_train[0])
    # the change parameter
    eta = 0.1
    # vector of 2D (0,0,0,0...,0),(0,0,0,0...,0),(0,0,0,0...,0)..
    w = np.zeros((3, d))

    # <<Perceptron>>
    # multi class
    # num of iterations
    epochs = 20
    for e in range(epochs):
        # take one random example from the training set
        X_train, Y_train = shuffle(X_train, Y_train)
        # x= out example while y is the class
    for x, y in zip(X_train, Y_train):
        x = x.astype(np.float)
        # predict
        # y_hat take the index of the max arg of multiply between x and w
        # w = hypnotize
        y_hat = np.argmax(np.dot(w, x))
        # update
        # check if y(the real class) != our class calculate
        if y != y_hat:
            # add little change to all the parameters in the wrong hypnotize to make more distance
            # w[y, :] += eta * x
            etax = eta * np.array(x)
            w[int(y)] += etax
            # make the true hypnotize closer to us
            # w[y_hat, :] -= eta * x
            w[int(y_hat)] -= etax
    return w
