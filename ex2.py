"""
Ori Kopel 205533151 kopelor
Shlomo Rabinovich 308432517 rabinos6
"""
import sys

import matplotlib.pyplot as plt

import Testing
import Training
import Utils

if __name__ == '__main__':
    # get the parameter from CMD
    if len(sys.argv) < 3:
        print("ERROR!!")
    data_train = sys.argv[1]
    data_label = sys.argv[2]
    test_data = None
    if len(sys.argv) == 4:
        test_data = sys.argv[3]

    clssesNum = 3
    lamda = 0.15
    etaPer = 0.01
    etaSvm = 0.2  # 0.05
    epochs = 70
    succRateinPA = []
    succRateinSVM = []
    succRateinPER = []
    plt.ylabel("Success rate")
    plt.xlabel("epochs arg")
    plt.title("success rate")
    data_train, data_label, test_data = Utils.Utils(data_train, data_label, test_data).orderData()
    iteration = []
    testArgs = [10, 50, 100, 200, 300, 400, 500, 600, 700]
    for e in testArgs:
        epochs = e
        for i in range(1):
            iteration.append(i)
            w_per, w_pa, w_svm = Training.Training(data_train, data_label, clssesNum, lamda, etaPer, etaSvm,
                                                   epochs).train(
                i)
            tester = Testing.Testing(data_train, data_label, w_per, w_pa, w_svm)
            if len(sys.argv) == 3:  # debug mode
                t1, t2, t3 = tester.testStatistic(i)
                succRateinPER.append(t1)
                succRateinPA.append(t2)
                succRateinSVM.append(t3)
                print("succeeds rate: per:", t1, " pa:", t2, " svm:", t3)
            else:  # testing mode
                tester.test(test_data)

    plt.plot(testArgs, succRateinSVM, label="SVM")
    plt.plot(testArgs, succRateinPA, label="PA")
    plt.plot(testArgs, succRateinPER, label="Perceptron")
    plt.legend()
    plt.show()
