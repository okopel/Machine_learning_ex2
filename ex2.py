"""
Ori Kopel 205533151 kopelor
Shlomo Rabinovich 308432517 rabinos6
"""
import sys

import numpy as np

import Testing
import Training
import Utils

if __name__ == '__main__':
    # get the parameter from CMD
    if len(sys.argv) < 4:
        print("ERROR!!")
    data_t = sys.argv[1]
    data_label = sys.argv[2]
    test_data = sys.argv[3]
    clssesNum = 3
    lamda = 0.2
    eta = 0.25
    epochs = 10
    tool = Utils.Utils(data_t, data_label, test_data)
    data_train, data_label, test_data = tool.orderData()
    trainer = Training.Training(data_train, data_label, clssesNum, lamda, eta, epochs)
    w_per, w_pa, w_svm = trainer.train()
    tester = Testing.Testing(test_data, w_per, w_pa, w_svm)
    if len(sys.argv) == 5:
        t1, t2, t3 = tester.testStatistic(np.genfromtxt(sys.argv[4], delimiter=','))
        print("succeeds rate: per:", t1, " pa:", t2, " svm:", t3)
    else:
        tester.test()
