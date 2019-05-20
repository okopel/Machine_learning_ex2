"""
Ori Kopel 205533151 kopelor
Shlomo Rabinovich 308432517 rabinos6
"""
import numpy as np
from scipy import stats


class Utils:
    def __init__(self, dt, dl, td):
        self.dt = dt
        self.dl = dl
        self.td = td

    def orderData(self, typeOfNormal):
        self.dt = np.genfromtxt(self.dt, delimiter=',', dtype="|U5")
        self.dl = np.genfromtxt(self.dl, delimiter=",", dtype='>i4')
        self.dt = self.one_hot(self.dt, ['M', 'F', 'I'])
        self.dt = self.Z_normalize(self.dt, typeOfNormal)
        # self.data_train = self.MinMax_normalize(self.data_train)

        if self.td is not None:
            self.td = np.genfromtxt(self.td, delimiter=",", dtype="|U5")
            self.td = self.one_hot(self.td, ['M', 'F', 'I'])
            self.td = self.Z_normalize(self.td, typeOfNormal)
            # self.test_data = self.MinMax_normalize(self.test_data)
        return self.dt, self.dl, self.td

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
