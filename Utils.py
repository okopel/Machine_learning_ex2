"""
Ori Kopel 205533151 kopelor
Shlomo Rabinovich 308432517 rabinos6
"""
import numpy as np
from scipy import stats


class Utils:
    def __init__(self, data_t, data_label, test_data):
        self.data_train = data_t
        self.data_label = data_label
        self.test_data = test_data

    def orderData(self, typeOfNormal):
        self.data_train = np.genfromtxt(self.data_train, delimiter=',', dtype="|U5")
        self.data_label = np.genfromtxt(self.data_label, delimiter=",", dtype='>i4')
        self.data_train = self.one_hot(self.data_train, ['M', 'F', 'I'])
        self.data_train = self.Z_normalize(self.data_train, typeOfNormal)
        # self.data_train = self.MinMax_normalize(self.data_train)

        if self.test_data is not None:
            self.test_data = np.genfromtxt(self.test_data, delimiter=",", dtype="|U5")
            self.test_data = self.one_hot(self.test_data, ['M', 'F', 'I'])
            self.test_data = self.Z_normalize(self.test_data, typeOfNormal)
            # self.test_data = self.MinMax_normalize(self.test_data)
        return self.data_train, self.data_label, self.test_data

    @staticmethod
    def Z_normalize(arrOfParams, type):
        if type == 0:
            return stats.zscore(arrOfParams.astype(float))
        if type == 1:
            return stats.zscore(arrOfParams.astype(float), ddof=1)
        if type == 2:
            return stats.zscore(arrOfParams.astype(float), axis=1)
        if type == 3:
            return stats.zscore(arrOfParams.astype(float), ddof=1, axis=1)
        if type == 4:
            return stats.mstats.zscore(arrOfParams.astype(float))
        if type == 5:
            return stats.mstats.zscore(arrOfParams.astype(float), ddof=1)
        if type == 6:
            return stats.mstats.zscore(arrOfParams.astype(float), ddof=1, axis=1)
        if type == 7:
            return stats.mstats.zscore(arrOfParams.astype(float), axis=1)

    @staticmethod
    def MinMax_normalize(arrOfParams):
        # (x-min)/(max-min)
        for i in range(len(arrOfParams[0])):
            minArg = float(min(arrOfParams[:, i]))
            maxArg = float(max(arrOfParams[:, i]))

            for j in range(len(arrOfParams)):
                if minArg == maxArg:
                    arrOfParams[j, i] = 0
                else:
                    arrOfParams[j, i] = (float(arrOfParams[j, i]) - minArg) / (maxArg - minArg)
        return arrOfParams

    # take one box with char/type args and separate it to bits
    # when the index of this char in arrOfypes is 1 and the other is 0
    # notice that the original box has to be in the 1st col
    @staticmethod
    def one_hot(arrOfData, arrOfTypes):
        colToAdd = len(arrOfTypes) - 1
        # adding col of zeros
        for i in range(colToAdd):
            arrOfData = np.c_[np.zeros(len(arrOfData)), arrOfData]
        for i in range(len(arrOfData)):
            for j in range(len(arrOfTypes)):
                if arrOfData[i][colToAdd] == arrOfTypes[j]:
                    # delete the original sign
                    arrOfData[i][colToAdd] = float(0)
                    # light the true bit
                    arrOfData[i][j] = float(1 / len(arrOfData))  # todo change to 1/3?
        return arrOfData
