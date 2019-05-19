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
    epochsPA = 80
    epochsSVM = 3
    epochPER = 180
    succRateinPA = []
    succRateinSVM = []
    succRateinPER = []
    plt.ylabel("Success rate")
    plt.xlabel("type")
    plt.title("success rate")
    iteration = []
    dt = data_train
    dl = data_label
    ts = test_data
    dt2 = data_train
    dl2 = data_label
    ts2 = test_data
    testArgs = [0, 1, 2, 3, 4]
    for i in testArgs:
        data_train, data_label, test_data = Utils.Utils(dt, dl, ts).orderData(3)
        i = int(i)
        for e in range(1):
            iteration.append(i)
            w_per, w_pa, w_svm = Training.Training(data_train, data_label, clssesNum, lamda, etaPer, etaSvm,
                                                   epochsPA, epochsSVM, epochPER).train(i)

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
