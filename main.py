"""
Ori Kopel 205533151 kopelor
Shlomo Rabinovich 308432517 rabinos6
"""
import sys

import Testing
import Training
import Utiles


def main():
    # get the parameter from CMD
    if len(sys.argv) < 4:
        print("ERROR!!")
        return
    data_t = sys.argv[1]
    data_label = sys.argv[2]
    test_data = sys.argv[3]
    test_label = sys.argv[4]
    tool = Utiles.Utiles(data_t, data_label, test_data, test_label)
    data_train, data_label, test_train, test_label = tool.orderData()
    trainer = Training.Training(data_t, data_label, 3, 0.2, 0.25, 50)
    w_per, w_pa, w_svm = trainer.train()
    tester = Testing.Testing(test_data, test_label, w_per, w_pa, w_svm)
    t1, t2, t3 = tester.test()
    print("succsess: per:", t1, " pa:", t2, " svm:", t3)


main()
