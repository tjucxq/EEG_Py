#coding=utf-8
import sys
import sklearn
import os
import xlrd
import numpy
import math
import copy
from ExtractFeature import ExtractFeature
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals import joblib
from sklearn.svm import SVR
from cxq_pca import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDRegressor
import matplotlib
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing

reload(sys)
sys.setdefaultencoding("utf-8")

delta = 2.5 #1-3Hz
theta = 6 #4-7Hz
alpha1 = 9 #8-9Hz
alpha2 = 11.5 #10-12Hz
beta1 = 15.5  #13-17Hz
beta2 = 24.5  #18-30Hz
gamma1 = 36 #31-40Hz
gamma2 = 46 #41-50Hz
period = 10 #窗口
#不同波段的weight
freq = [2.5, 6, 9, 11.5, 15.5, 24.5, 36, 46]
width=[3, 4, 2, 3, 5, 13, 10, 10]
k = 1.5

#生成特征
def GetFeature(filename):
    global k
    trainX = []
    trainY = []
    data = xlrd.open_workbook(filename)
    table = data.sheets()[1]

    #tranform sheet to numpy array
    inputdata = Sheet2Array(table)
    # print type(inputdata)
    # print inputdata[0]
    # print numpy.shape(inputdata)

    rows = len(inputdata)
    # print rows

    #深拷贝
    # inputdata2 = copy.deepcopy(inputdata)

    #进行Z_Score变换
    Z_Score(0, rows-1, inputdata)

    # 进行数据标准化
    # inputdata = Normalize(inputdata)

    a = 0
    b = 0
    for i in range(1, rows):
        if inputdata[i, 21] != inputdata[i-1, 21]:
            if a == 0:
                a = i
            else:
                b = i-1
                break

    x, y = StaticFeature(0, a-1, inputdata, -1) # 平静为-1， 昏睡状态为1
    # m, n = StaticFeature(0, a-1, inputdata2, -1)
    # for j in range(len(m)):
    #     e = m[j]
    #     for i in range(len(e)):
    #         e[i] = math.log(e[i])
    #     x[j] += e
    trainX += x
    trainY += y
    # x1, y1 = DataPreprocess(x,y,k)
    # trainX += x1
    # trainY += y1
    x, y = StaticFeature(b+1, rows-1, inputdata, 1)
    # m, n = StaticFeature(b+1, rows-1, inputdata2, 1)
    # for j in range(len(m)):
    #     e = m[j]
    #     for i in range(len(e)):
    #         e[i] = math.log(e[i])
    #     x[j] += e
    trainX += x
    trainY += y
    # x1, y1 = DataPreprocess(x,y,k)
    # trainX += x1
    # trainY += y1

    return trainX, trainY

#统计特征
def StaticFeature(a,b,inputdata,label):
    trainX = []
    trainY = []
    index = a
    # 13 - 20  标志位 21
    # 处理同一阶段
    while index + period -1 <= b:
        li = []
        end = index + period
        temp = 0.0
        temp2 = 0.0

        # attention and meditation feature
        # li.append(numpy.average(table.col_values(2)[index:end]))
        # li.append(numpy.average(table.col_values(3)[index:end]))
        # li.append(numpy.std(table.col_values(2)[index:end]))
        # li.append(numpy.std(table.col_values(3)[index:end]))
        for i in range(13, 21):
            lst = inputdata[index:end, i]
            avg = numpy.average(lst)
            v = numpy.std(lst)
            max_t = max(lst)
            min_t = min(lst)
            diff = max_t - min_t
            avg1 = numpy.average(inputdata[index:end, i - 8])
            temp = temp + avg1 * freq[i - 13] * width[i - 13]
            temp2 = temp2 + avg1 * width[i - 13]
            # 每种波段有5个特征
            li.append(avg)
            li.append(v)
            li.append(max_t)
            li.append(min_t)
            li.append(diff)
        # 特征组合
        li.append(li[0] + li[5])  # delta + theta
        li.append(li[0] / li[10])  # delta/low_alpha
        li.append(li[0] / li[15])  # delta/high_alpha
        li.append(li[0] / li[20])  # delta/low_beta
        li.append(li[0] / li[25])  # delta/high_beta
        li.append(li[5] / li[20])  # theta/low_beta
        li.append(li[5] / li[25])  # theta/high_beta
        li.append(li[10] / li[20])  # low_alpha/low_beta
        li.append(li[10] / li[25])  # low_alpha/high_beta
        li.append(temp / temp2)  # GF 重心频率
        # l = len(li)
        # for i in range(l):
        #     li.append(math.exp(li[i]))

        trainX.append(li)
        trainY.append(label)
        index += int(period / 4)+1
    return trainX, trainY

#将excel转成ndarray
def Sheet2Array(table):
    num_rows = table.nrows - 1
    num_cells = table.ncols
    inputData = numpy.empty([num_rows, num_cells])
    curr_row = -1
    while curr_row < num_rows:  # for each row
        curr_row += 1
        row = table.row(curr_row)
        if curr_row > 0:  # don't want the first row because those are labels
            for col_ind, el in enumerate(row):
                inputData[curr_row - 1, col_ind] = el.value
    return inputData

#对数据进行标准化处理
def Z_Score(a, b, inputdata):
    for i in range(13,21):
        l = inputdata[a:b+1, i]
        avg = numpy.average(l)
        std = numpy.std(l)
        for j in range(a, b+1):
            inputdata[j, i] = (inputdata[j, i]-avg)/std

#对数据进行正则化处理
def Normalize(inputdata):
    X_normalized = preprocessing.normalize(inputdata, norm='l2')
    return X_normalized

#训练模型
def BuildModel(trainX,trainY):
    #clf = SVC(C=0.8, kernel='rbf', degree=2, gamma=1.0, coef0=0.0, shrinking=True,
    #          probability=False, tol=0.0001, cache_size=200, class_weight=None, verbose=False,
    #          max_iter=-1, decision_function_shape=None, random_state=None)
    clf = RandomForestClassifier(n_estimators = 500, criterion = 'gini', max_depth = 30,
                                min_samples_split = 10, min_samples_leaf = 1, min_weight_fraction_leaf = 0.0,
                                max_features = 0.8, max_leaf_nodes = None, bootstrap = True, oob_score = False,
                                n_jobs = 4, random_state = None, verbose = 0, warm_start = False, class_weight = None)
    # clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=3, min_samples_split=20, min_samples_leaf=2,
    #                              min_weight_fraction_leaf=0.0, max_features=0.8, random_state=None, max_leaf_nodes=None,
    #                              class_weight=None, presort=False)
    # clf = GradientBoostingClassifier(loss='exponential', learning_rate=0.1, n_estimators=300, subsample=0.8,
    #                                  min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=5,
    #                                  init=None, random_state=None, max_features=0.7, verbose=0, max_leaf_nodes=None,
    #                                  warm_start=False, presort='auto')
    # clf = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=300, subsample=0.8, min_samples_split=2,
    #                                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=6, init=None,
    #                                 random_state=None, max_features=0.8, alpha=0.45, verbose=0, max_leaf_nodes=None,
    #                                 warm_start=False, presort='auto')
    # clf = SVR(kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=0.001, C=1.5, epsilon=0.2, shrinking=True, cache_size=200,
    #     verbose=False, max_iter=3000)
    #
    # clf = LogisticRegression(penalty='l2', dual=False, tol=0.0005, C=1.5, fit_intercept=True,intercept_scaling=1,
    #                          class_weight=None, random_state=None, solver='liblinear',max_iter=1000, multi_class='ovr',
    #                          verbose=0, warm_start=False, n_jobs=4)

    # clf = SGDRegressor(loss='epsilon_insensitive', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, n_iter=1000,
    #                    shuffle=True, verbose=0, epsilon=0.1, random_state=None, learning_rate='invscaling', eta0=0.01,
    #                    power_t=0.25, warm_start=False, average=False)

    # clf = MLPClassifier(hidden_layer_sizes=(130,), activation='relu', solver='lbfgs', alpha=0.0001,
    #                                      batch_size='auto', learning_rate='adaptive', learning_rate_init=0.001,
    #                                      power_t=0.5, max_iter=500, shuffle=True, random_state=None, tol=0.0001,
    #                                      verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    #                                      early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
    #                                      epsilon=1e-08)

    # 数据集划分
    # X_train, X_test, y_train, y_test = cross_validation.train_test_split(trainX, trainY, test_size=0.3, random_state=0)
    # print "x_test len is " + str(len(X_test))

    print "train is beginning"

    clf.fit(trainX,trainY)
    print "train is done"

    # predictY = clf.predict(X_test)

    return clf

#保存特征文件
def dumpFeature(trainX,trainY):
    out = file("feature_lz.txt", "w")
    for i in range(len(trainX)):
        o = str(trainY[i]) + "\t" + "\t".join([str(ele) for ele in trainX[i]])
        out.write(o + "\n")
    out.close()

# 加载data目录，读excel，统计特征
def LoadTrainData():
    path = "train_data/"
    files = os.listdir(path)
    trainX = []
    trainY = []
    for f in files:
        if f.find("LZ") == -1 and f.find("lz") == -1:
            continue
        print f.decode("gbk").encode("utf-8")
        f = (path + f).decode("gbk")
        x, y = GetFeature(f)
        trainX += x
        trainY += y
    return trainX,trainY

def DataPreprocess(trainX, trainY, k):
    tt = []
    for lst in trainX:
        tt.append(lst[0])
    avg_delta = numpy.average(tt)
    std_delta = numpy.std(tt)
    X = []
    Y = []
    for i in range(len(trainX)):
        lst = trainX[i]
        if lst[0] >= avg_delta - k*std_delta and lst[0] <= avg_delta + k*std_delta:
            X.append(lst)
            Y.append(trainY[i])
    print "k is : "+str(k)
    print "Size before processing is : "+str(len(trainX)) + " Size after processing is : "+str(len(X)) + \
          " Confidence level is : "+str(round(len(X)*1.0/len(trainX),2))
    return X, Y

def CrossValidation(path):
    path = path.encode("gbk")
    files = os.listdir(path)

    out = file(path+"rf_Zscore_1222.txt","w")
    a = 0
    avg_prediction = 0.0
    size = 0
    while a < len(files):
        trainX = []
        trainY = []
        testX = []
        testY = []
        valiName = files[a]
        if len(valiName) < 30 or valiName.find("xlsx") < 0:
            a += 1
            continue
        size += 1
        f = (path + valiName).decode("gbk")

        x, y = GetFeature(f)
        testX += x
        testY += y
        j = 0
        while j < len(files):
            if a == j:
                j += 1
                continue
            name = files[j]
            if len(name) < 30 or name.find("xlsx") < 0:
                j += 1
                continue
            f = (path + name).decode("gbk")
            x, y = GetFeature(f)
            trainX += x
            trainY += y
            j += 1
        # pca = PCA()
        # trainX, testX = pca.process(trainX, testX)
        print len(trainX)
        print "file name is : "+valiName.decode("gbk").encode("utf-8")
        out.write("file name is : "+valiName.decode("gbk").encode("utf-8")+"\n")
        clf = BuildModel(trainX, trainY)
        predictY = clf.predict(testX)
        # 判断准确率
        right = 0
        for i in range(len(predictY)):
            if testY[i] == 1 and predictY[i] > 0:
                right += 1
            if testY[i] == -1 and predictY[i] <= 0:
                right += 1
        avg_prediction += right * 1.0 / len(predictY)
        print "precision is : " + str(right * 1.0 / len(predictY))
        out.write("precision is : " + str(right * 1.0 / len(predictY))+"\n")
        a += 1
    out.write("\navg_prediction is : "+str(avg_prediction/size))
    out.close()
    return trainX, trainY

def Train(path):
    path = path.encode("gbk")
    files = os.listdir(path)
    trainX = []
    trainY = []
    a = 0
    while a < len(files):
        valiName = files[a]
        if len(valiName) < 30 or valiName.find("xlsx") < 0:
            a += 1
            continue
        f = (path + valiName).decode("gbk")
        x, y = GetFeature(f)
        trainX += x
        trainY += y
        print len(trainX)
        a += 1
    clf = BuildModel(trainX, trainY)
    path += "model/"
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
    joblib.dump(clf, path+"rf.model")

def Test(path):
    path = path.encode("gbk")
    clf = joblib.load(path+"model/rf.model")
    path += "new_add/"
    files = os.listdir(path)
    a = 0
    avg_prediction = 0.0
    size = 0
    out = file(path + "rf_Zscore.txt", "w")
    while a < len(files):
        valiName = files[a]
        if len(valiName) < 30 or valiName.find("xlsx") < 0:
            a += 1
            continue
        size += 1
        f = (path + valiName).decode("gbk")
        x, y = GetFeature(f)
        print "file name is : " + valiName.decode("gbk").encode("utf-8")
        out.write("file name is : " + valiName.decode("gbk").encode("utf-8") + "\n")
        predictY = clf.predict(x)
        right = 0
        for i in range(len(predictY)):
            if y[i] == 1 and predictY[i] > 0:
                right += 1
            if y[i] == -1 and predictY[i] <= 0:
                right += 1
        avg_prediction += right * 1.0 / len(predictY)
        print "precision is : " + str(right * 1.0 / len(predictY))
        out.write("precision is : " + str(right * 1.0 / len(predictY)) + "\n")
        a += 1
    out.write("\navg_prediction is : " + str(avg_prediction / size))
    out.close()

#使用PCA对原始8个波段数据进行降维处理
def ApplyPCA(a,b,table,label):
    trainX = []
    trainY = []
    for i in range(a,b+1):
        trainX.append(table.row_values(i)[13:21])
        trainY.append(label)
    pca = PCA()
    pca.process(trainX,trainY)
    return trainX, trainY

#加载测试数据
def LoadTestData():
    path = "test_data_version2/"
    files = os.listdir(path)
    testX = []
    testY = []
    for f in files:
        # if f.find(u"单点".encode("gbk"))>-1:
        if f.find("lz")==-1:
            continue
        #     f = (path + f).decode("gbk")
        #     a, b = GetFeature(f)
        #     testX += a
        #     testY += b
        #     continue
        print f.decode("gbk").encode("utf-8")
        f = (path + f).decode("gbk")
        x, y = GetFeature(f)
        testX += x
        testY += y
    return testX, testY

if __name__ == "__main__":

    # lst = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    lst = [1.2, 1.3, 1.4, 1.5, 1.6]
    # lst = ["cxq", "hyy", "hzj", "lz", "wnb"]
    # lst = ["lz"]
    # GetFeature("./cxq-调节2hz_20161102_104323_ASIC_EEG.xlsx".decode("utf-8").encode("gbk"))

    for ele in lst:
        # path = "E:/计算所/EEG/Matlab Code/EEG_repository/EEG/Data/全部数据/原始excel/"
        path = "E:/计算所/EEG/Matlab Code/EEG_repository/EEG/Data/全部数据/原始excel/理想数据k=1~2/"
        # path = "理想数据/"
        # path = "./"
        path += "k="+str(ele)+"/"
        # path += ele +"/"
        CrossValidation(path)

        #训练模型，保存模型
        # Train(path)

        #用已有模型，预测新来数据
        # Test(path)