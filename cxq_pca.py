#coding=utf-8

import sys
import numpy

reload(sys)
sys.setdefaultencoding("utf-8")

class PCA:

    def __init__(self):
        self.thre = 0.9

    def process(self, trainX, testX):
        # meanVals = numpy.mean(trainX, axis=0)
        # DataAdjust = trainX - meanVals  # 减去平均值
        covMat = numpy.cov(numpy.transpose(trainX))

        eigVals, eigVects = numpy.linalg.eig(covMat)  # 计算特征值和特征向量

        eigValInd = numpy.argsort(eigVals)[::-1]
        # print eigVals
        # print eigVects
        sum = 0.0
        for ele in eigVals:
            sum += abs(ele)
        cur = 0.0
        for i in eigValInd:
            print eigVals[i]/sum
            cur += eigVals[i]/sum
            if cur > 0.99:
                break
        print cur

        eigValInd = eigValInd[:i+1]  # 保留最大的前K个特征值

        redEigVects = eigVects[:, eigValInd]  # 对应的特征向量
        lowDDataMat = numpy.dot(trainX, redEigVects)  # 将数据转换到低维新空间

        testX = numpy.dot(testX, redEigVects)
        return lowDDataMat, testX