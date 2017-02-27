#coding=utf-8
import sys
import numpy
import time
import xlrd

reload(sys)
sys.setdefaultencoding("utf-8")

class Tool:

    def __init__(self):
        # 不同波段的weight
        self.freq = [2.5, 6, 9, 11.5, 15.5, 24.5, 36, 46]
        self.width = [3, 4, 2, 3, 5, 13, 10, 10]
        self.period = 150
        self.percent = 0.3
        self.hr_map = dict()
        self.eeg_map = dict()
        self.data1 = []
        self.data2 = []
        self.data3 = []


    def diff_EEG_GF(self, data):
        self.eeg_map = dict()
        res = []
        mat = numpy.mat(data)
        # 13 - 20  标志位 21
        index = 0
        size = len(data)
        # 处理同一阶段
        while index + self.period - 1 < size:
            end = index + self.period
            k = self.period*self.percent;
            temp = 0.0
            temp2 = 0.0
            li = []
            for i in range(1, 9):
                avg1 = numpy.average(mat[index:index+k, i])
                temp = temp + avg1 * self.freq[i - 1] * self.width[i - 1]
                temp2 = temp2 + avg1 * self.width[i - 1]
            y1 = temp/temp2
            for i in range(1, 9):
                avg1 = numpy.average(mat[end-k:end, i])
                temp = temp + avg1 * self.freq[i - 1] * self.width[i - 1]
                temp2 = temp2 + avg1 * self.width[i - 1]
            y2 = temp/temp2
            li.append(y1 - y2)  # GF 重心频率
            li.append(numpy.average(mat[index:end, 9]))
            res.append(li)
            index += int(self.period / 4)
        return res

    def readHeartRate(self, filename):
        self.hr_map = dict()
        with open(filename.encode("gbk"), 'r') as fin:
            for line in fin:
                frags = line.strip().split(",")
                a = time.strptime(frags[0], "%Y-%m-%d %H:%M:%S")
                format_time = time.strftime("%H%M%S", a)
                self.hr_map[int(format_time)] = 60000.0/int(frags[1])
        return self.hr_map

    def readEEG(self, filename):
        data = xlrd.open_workbook(filename.encode("gbk"))
        table = data.sheets()[1]
        rows = table.nrows
        #get the break point
        a = 0
        b = 0
        for i in range(2, rows):
            if table.row_values(i)[21] != table.row_values(i - 1)[21]:
                if a == 0:
                    a = i
                else:
                    b = i - 1
                    break
        #stage 1
        self.data1 = []
        #stage 2
        self.data2 = []
        #stage 3
        self.data3 = []
        for i in range(2, a+1):
            t = int(table.row_values(i)[22])
            if t in self.hr_map:
                temp = []
                temp.append(t)
                temp += table.row_values(i)[5:13]
                temp.append(self.hr_map[t])
                self.data1.append(temp)
        for i in range(a+1, b):
            t = int(table.row_values(i)[22])
            if t in self.hr_map:
                temp = []
                temp.append(t)
                temp += table.row_values(i)[5:13]
                temp.append(self.hr_map[t])
                self.data2.append(temp)
        for i in range(b, rows):
            t = int(table.row_values(i)[22])
            if t in self.hr_map:
                temp = []
                temp.append(t)
                temp += table.row_values(i)[5:13]
                temp.append(self.hr_map[t])
                self.data3.append(temp)

if  __name__ == "__main__":
    eeg_file = ["lz-调节2hz_20161031_163934_ASIC_EEG.xlsx","cxq-调节2hz_20161102_104323_ASIC_EEG.xlsx","wnb-调节2hz_20161101_102759_ASIC_EEG.xlsx"]
    hr_file = ["22-lz调节2hz-SPO-10-31_17-3.txt", "24-cxq调节-SPO-11-02_10-58.txt", "23-wnb调节2hz-SPO-11-01_10-44.txt"]
    instance = Tool()
    res = []
    for i in range(len(eeg_file)):
        instance.readHeartRate(hr_file[i])
        instance.readEEG(eeg_file[i])
        res += instance.diff_EEG_GF(instance.data1)
    mat = numpy.mat(res)
    cov_mat = numpy.cov(numpy.transpose(mat))
    print cov_mat[0][1] / (numpy.std(mat[:, 0]) * numpy.std(mat[:, 1]))

    res = []
    for i in range(len(eeg_file)):
        instance.readHeartRate(hr_file[i])
        instance.readEEG(eeg_file[i])
        res += instance.diff_EEG_GF(instance.data2)
    mat = numpy.mat(res)
    cov_mat = numpy.cov(numpy.transpose(mat))
    print cov_mat[0][1] / (numpy.std(mat[:, 0]) * numpy.std(mat[:, 1]))

    res = []
    for i in range(len(eeg_file)):
        instance.readHeartRate(hr_file[i])
        instance.readEEG(eeg_file[i])
        res += instance.diff_EEG_GF(instance.data3)
    mat = numpy.mat(res)
    cov_mat = numpy.cov(numpy.transpose(mat))
    print cov_mat[0][1] / (numpy.std(mat[:, 0]) * numpy.std(mat[:, 1]))