from random import shuffle

import numpy as np


def perceptron(X_train, Y_train):
    # num of args in the traning set
    m = len(X_train)
    # num of args in one example
    d = len(X_train[0])
    # the change parameter
    eta = 0.1
    # vector of 2D (0,0,0,0...,0),(0,0,0,0...,0),(0,0,0,0...,0)..
    w = np.zeros((d,))

    # Perceptron
    # multi class
    # num of iterations
    epochs = 20
    for e in range(epochs):
        # take one random example from the training set
        X_train, Y_train = shuffle(X_train, Y_train, random_state=1)
        # x= out example while y is the class
        for x, y in zip(X_train, Y_train):
            # predict
            # y_hat take the index of the max arg of multiply between x and w
            # w = hypnotize
            y_hat = np.argmax(np.dot(w, x))
            # update
            # check if y(the real class) != our class calculate
            if y != y_hat:
                # add little change to all the parameters in the wrong hypnotize to make more distance
                w[y, :] += eta * x
                # make the true hypnotize closer to us
                w[y_hat, :] -= eta * x
    # save our new hypnotize
    w_perceptron = w

    # testing
    m_perceptron = 0
    # check all the training set after our training
    for t in range(0, m):

        # y_hat = np.sign(np.dot(w_perceptron, X_train[t]))
        y_hat = np.argmןמ(np.dot(w_perceptron, X_train[t]))
        if Y_train[t] != y_hat:
            # error++
            m_perceptron += 1
    print("perceptron err =", float(m_perceptron) / m)
