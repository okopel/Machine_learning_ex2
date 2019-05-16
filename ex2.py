"""
Ori Kopel 205533151 kopelor
Shlomo Rabinovich 308432517 rabinos6
"""
import sys

import matplotlib.pyplot as plt
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
    lamda = 0.15
    etaPer = 0.01
    etaSvm = 0.05
    etaPer = 0.0000001
    etaSvm = 0.0000001
    # etaa = [0, 0.01, 0.05, 0.1, 0.2, 0.3]
    # lamdaa = [0, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    epochs = 400
    # epochsa = [100, 200, 400, 550, 700]
    succRateinPA = []
    succRateinSVM = []
    succRateinPER = []
    plt.ylabel("success rate")
    plt.xlabel("iteration number")
    plt.title("success rate by epochs={}  without shuffle".format(epochs))
    data_train, data_label, test_data = Utils.Utils(data_t, data_label, test_data).orderData()
    for i in range(3):

        #        etaPer = i
        #       print("etaPer:", etaPer)
        w_per, w_pa, w_svm = Training.Training(data_train, data_label, clssesNum, lamda, etaPer, etaSvm, epochs).train()
        tester = Testing.Testing(test_data, w_per, w_pa, w_svm)
        if len(sys.argv) == 5:  # debug mode
            t1, t2, t3 = tester.testStatistic(np.genfromtxt(sys.argv[4], delimiter=','))
            succRateinPER.append(t1)
            succRateinPA.append(t2)
            succRateinSVM.append(t3)
            # print("succeeds rate: svm:", t3)
            print("succeeds rate: per:", t1, " pa:", t2, " svm:", t3)
        else:  # testing mode
            tester.test()
    iteration = [1, 2, 3]
    plt.plot(iteration, succRateinSVM, label="SVM")
    plt.plot(iteration, succRateinPA, label="PA")
    plt.plot(iteration, succRateinPER, label="Perceptron")
    plt.legend()
    plt.show()
