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
    dt = sys.argv[1]  # data_train
    dl = sys.argv[2]  # data label
    td = None  # test data
    if len(sys.argv) == 4:
        td = sys.argv[3]
    params = {
        "clssesNum": 3,
        "lamda": 0.15,
        "etaPer": 0.01,
        "etaSvm": 0.2,  # 0.05
        "epochsPA": 80,
        "epochsSVM": 3,
        "epochPER": 180
    }
    succRateinPA = []
    succRateinSVM = []
    succRateinPER = []
    plt.ylabel("Success rate")
    plt.xlabel("type")
    plt.title("Report")
    iteration = []
    dt, dl, td = Utils.Utils(dt, dl, td).orderData(3)
    seper = 5
    testArgs = range(seper)
    w_per = [[], [], [], [], []]
    w_pa = [[], [], [], [], []]
    w_svm = [[], [], [], [], []]
    tester = [Training, Training, Training, Training, Training]
    for tr in [0]:
        for i in testArgs:
            i = int(i)
            iteration.append(i)
            w_per[i], w_pa[i], w_svm[i] = Training.Training(dt, dl, params).train(i)
            tester[i] = Testing.Testing(dt, dl, w_per[i], w_pa[i], w_svm[i])
            if len(sys.argv) == 3:  # debug mode
                t1, t2, t3 = tester[i].testStatistic(i)
                succRateinPER.append(t1)
                succRateinPA.append(t2)
                succRateinSVM.append(t3)
                print("succeeds rate: per:", t1, " pa:", t2, " svm:", t3)

    if len(sys.argv) > 3:  # submit mode
        Testing.Testing.testerSubmit(tester, td)
        # tester.test(td)
    else:
        plt.plot(testArgs, succRateinSVM, label="SVM {}".format(tr))
        succRateinSVM = []
        plt.plot(testArgs, succRateinPA, label="PA {} ".format(tr))
        succRateinPA = []
        plt.plot(testArgs, succRateinPER, label="Perceptron {}".format(tr))
        succRateinPER = []
        plt.legend()
        plt.show()
