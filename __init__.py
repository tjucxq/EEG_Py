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

half_num = 0

#生成特征
def GetFeature(filename):
    global k
    trainX = []
    trainY = []
    data = xlrd.open_workbook(filename)
    table = data.sheet_by_name("第一组")

    #tranform sheet to numpy array
    inputdata = Sheet2Array(table)
    # print type(inputdata)
    # print inputdata[0]
    # print numpy.shape(inputdata)

    rows = len(inputdata)
    # print rows

    #深拷贝
    # inputdata2 = copy.deepcopy(inputdata)

    a = 0
    b = 0
    for i in range(1, rows):
        if inputdata[i, 21] != inputdata[i-1, 21]:
            if a == 0:
                a = i
            else:
                b = i-1
                break

    # 进行Z_Score变换
    # Z_Score(0, rows-1, inputdata)

    # 进行数据标准化
    # inputdata = Normalize(inputdata)

    # 进行max-min标准化
    # Max_min(0, rows-1, inputdata)

    x, y = StaticFeature(0, a-1, inputdata, -1) # 平静为-1， 昏睡状态为1
    # Z_Score(0,len(x)-1,x)    # 对提取特征进行标准化
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
    # Z_Score(0, len(x) - 1, numpy.asarray(x))   # 对提取的特征进行标准化
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
            avg_diff = 0.0
            for j in range(index, end-1):
                avg_diff += numpy.abs(inputdata[j+1][i]-inputdata[j][i])
            # 每种波段有6个特征
            li.append(avg)
            li.append(v)
            li.append(max_t)
            li.append(min_t)
            li.append(diff)
            li.append(avg_diff/period)
        # 特征组合
        li.append(li[0] + li[6])  # delta + theta
        li.append(li[0]/li[6])    # delta / theta
        li.append(li[0] / li[12])  # delta/low_alpha
        li.append(li[0] / li[18])  # delta/high_alpha
        li.append(li[0] / li[24])  # delta/low_beta
        li.append(li[0] / li[20])  # delta/high_beta
        li.append(li[6] / li[24])  # theta/low_beta
        li.append(li[6] / li[30])  # theta/high_beta
        li.append(li[12] / li[24])  # low_alpha/low_beta
        li.append(li[12] / li[30])  # low_alpha/high_beta
        li.append(temp / temp2)  # GF 重心频率

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
    X_normalized = preprocessing.normalize(inputdata[:,13:21], norm='l2')
    for i in range(len(inputdata)):
        for j in range(13,21):
            inputdata[i,j] = X_normalized[i][j-13]
    return inputdata

#对数据进行max-min处理
def Max_min(a, b, inputdata):
    for i in range(13,21):
        l = inputdata[a:b+1, i]
        max = numpy.max(l)
        min = numpy.min(l)
        for j in range(a, b+1):
            inputdata[j, i] = (inputdata[j, i]-min)*1.0/(max - min)

#训练模型
def BuildModel(trainX,trainY, half_num):
    #clf = SVC(C=0.8, kernel='rbf', degree=2, gamma=1.0, coef0=0.0, shrinking=True,
    #          probability=False, tol=0.0001, cache_size=200, class_weight=None, verbose=False,
    #          max_iter=-1, decision_function_shape=None, random_state=None)
    clf1 = RandomForestClassifier(n_estimators = 500, criterion = 'gini', max_depth = 30,
                                min_samples_split = 10, min_samples_leaf = 1, min_weight_fraction_leaf = 0.0,
                                max_features = 0.8, max_leaf_nodes = None, bootstrap = True, oob_score = False,
                                n_jobs = 12, random_state = None, verbose = 0, warm_start = False, class_weight = None)
    # clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=3, min_samples_split=20, min_samples_leaf=2,
    #                              min_weight_fraction_leaf=0.0, max_features=0.8, random_state=None, max_leaf_nodes=None,
    #                              class_weight=None, presort=False)
    # clf2 = GradientBoostingClassifier(loss='exponential', learning_rate=0.1, n_estimators=300, subsample=0.8,
    #                                  min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=5,
    #                                  init=None, random_state=None, max_features=0.7, verbose=0, max_leaf_nodes=None,
    #                                  warm_start=False, presort='auto')
    clf2 = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=300, subsample=0.8, min_samples_split=2,
                                    min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=6, init=None,
                                    random_state=None, max_features=0.8, alpha=0.45, verbose=0, max_leaf_nodes=None,
                                    warm_start=False, presort='auto')
    clf3 = SVR(kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=0.001, C=1.5, epsilon=0.2, shrinking=True, cache_size=200,
        verbose=False, max_iter=3000)
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
    clf1.fit(trainX,trainY)
    # clf2.fit(trainX,trainY)
    # clf3.fit(trainX,trainY)

    #模型融合策略
    # ml_lst = []
    # ml_lst.append(clf1)
    # ml_lst.append(clf2)
    # ml_lst.append(clf3)
    # res = Ensemble_Train(ml_lst, trainX, trainY, half_num)

    print "train is done"

    return [clf1, clf2, clf3]
    # return res

def Ensemble_Train(ml_lst,trainX,trainY,half_num):
    print "train is beginning"
    clf1 = ml_lst[0]
    clf2 = ml_lst[1]
    clf3 = ml_lst[2]
    clf1.fit(trainX[:len(trainX)/2], trainY[:len(trainX)/2])
    clf2.fit(trainX[:len(trainX)/2], trainY[:len(trainX)/2])
    clf3.fit(trainX[:len(trainX)/2], trainY[:len(trainX)/2])
    print "train is done"

    sec_trainX = []
    sec_trainY = trainY[len(trainX)/2:]
    for i in range(len(ml_lst)):
        clf = ml_lst[i]
        # if i < 2:
        #     yy = clf.predict_proba(trainX[len(trainX)/2:])
        # else:
        #     yy = clf.predict(trainX[len(trainX)/2:])
        yy = clf.predict(trainX[len(trainX) / 2:])
        if i == 0:
            for j in range(len(yy)):
                sec_trainX.append([])
                sec_trainX[j].append(yy[j])
            continue
        for j in range(len(yy)):
            sec_trainX[j].append(yy[j])

    ensemble_clf = LogisticRegression(penalty='l2', dual=False, tol=0.0005, C=1.5, fit_intercept=True, intercept_scaling=1,
                             class_weight=None, random_state=None, solver='liblinear',max_iter=100, multi_class='ovr',
                             verbose=0, warm_start=False, n_jobs=4)
    print sec_trainX
    print numpy.asarray(sec_trainX)
    ensemble_clf.fit(numpy.asarray(sec_trainX),sec_trainY)
    ml_lst.append(ensemble_clf)
    return ml_lst

def Ensemble_Test(ml_lst,testX):
    sec_testX = []
    for i in range(len(ml_lst)-1):
        clf = ml_lst[i]
        # if i < 2:
        #     predictY = clf.predict_proba(testX)
        # else:
        #     predictY = clf.predict(testX)
        predictY = clf.predict(testX)
        if i == 0:
            for j in range(len(predictY)):
                sec_testX.append([])
                sec_testX[j].append(predictY[j])
            continue
        for j in range(len(predictY)):
            sec_testX[j].append(predictY[j])
    predictY = ml_lst[-1].predict(sec_testX)
    return predictY

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


def CrossValidation(path):
    global half_num
    model_name = ["rf.model", "gbrt.model", "svm.model"]
    path = path.encode("gbk")
    files = os.listdir(path)

    out1 = file(path+"rf_0405.txt","w")
    # out2 = file(path+"gbrt_0405.txt","w")
    # out3 = file(path+"svm_0405.txt","w")
    # out = [out1, out2, out3]
    out = [out1]
    a = 0
    avg_prediction = [0.0, 0.0, 0.0]
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
            # 判断一半的数据集
            if size == 15:
                half_num = len(trainY)

            # pca = PCA()
        # trainX, testX = pca.process(trainX, testX)
        print len(trainX)
        print "file name is : "+valiName.decode("gbk").encode("utf-8")

        clf = BuildModel(trainX, trainY, half_num)

        #三个模型
        for m in range(len(out)):
            out[m].write("file name is : " + valiName.decode("gbk").encode("utf-8") + "\n")
            #单模型
            predictY = clf[m].predict(testX)
            #多模型融合
            # predictY = Ensemble_Test(clf,testX)
            # 判断准确率
            TP = 0
            TN = 0
            Positive = 0
            Negative = 0
            for i in range(len(predictY)):
                if testY[i] == 1:
                    Positive += 1
                    if predictY[i] > 0.5:
                        TP += 1
                else:
                    Negative += 1
                    if predictY[i] <= 0.5:
                        TN += 1
            right = TP +TN
            avg_prediction[m] += right * 1.0 / len(predictY)
            print model_name[m] + " precision is : " + str(right * 1.0 / len(predictY))
            out[m].write("TP is : "+str(TP)+" TN is : "+str(TN)+" Positive is : "+str(Positive)+" Negative is : "+str(Negative)+" precision is : " + str(right * 1.0 / len(predictY))+"\n")
        a += 1

    for i in range(len(out)):
        out[i].write("\navg_prediction is : "+str(avg_prediction[i]/size)+ " size is : "+str(size))
        out[i].close()

        path += "model/"
        isExists = os.path.exists(path)
        if not isExists:
            os.makedirs(path)
        joblib.dump(clf[i], path + model_name[i])

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

def Test(path,path2):
    path = path.encode("gbk")
    path2 = path2.encode("gbk")
    clf = joblib.load(path2+"model/rf.model")
    # path += "new_add/"
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
    lst = [1.3, 1.4, 1.5, 1.6, 1.7]
    # lst = [1.3]
    # lst = ["cxq", "hyy", "hzj", "lz", "wnb"]
    # lst = ["lz"]
    # GetFeature("./cxq-调节2hz_20161102_104323_ASIC_EEG.xlsx".decode("utf-8").encode("gbk"))

    for ele in lst:
        # path = "E:/计算所/EEG/Matlab Code/EEG_repository/EEG/Data/全部数据/原始excel/调节6Hz(K=1~2)/"
        path = "E:/计算所/EEG/Matlab Code/EEG_repository/EEG/Data/全部数据/原始excel/理想数据k=1~2/"

        # path = "理想数据/"
        # path = "./"
        path += "k="+str(ele)+"/"
        # path2 += "k=" + str(ele) + "/"
        # path += ele +"/"
        CrossValidation(path)

        # 训练模型，保存模型
        # Train(path)

        # 用已有模型，预测新来数据
        # Test(path,path2)