from com.cn.CFEEG import mergeFile

resultDict=mergeFile('E:\脑电材料\数据采集\EEG-master\Data\全部数据\原始excel\平静','第一组',row=(0,),col=(13,21),target_col=21,window_size=10,step=3)
# print(resultDict)
import numpy as np
print(np.shape(resultDict['train_data']))

from sklearn import cross_validation

X_train,X_test,y_train,y_test=cross_validation.train_test_split(resultDict['train_data'],resultDict['target'],test_size=0.2,random_state=0)
from sklearn.ensemble import RandomForestClassifier
# max_depth=15, min_samples_leaf=1,max_features=0.7
clf = RandomForestClassifier(n_estimators = 50,criterion='gini',min_samples_split=50,max_depth=10, min_samples_leaf=1,max_features=0.7,n_jobs=3 )

#训练模型
s = clf.fit(X_train , y_train)

#评估模型准确率
r = clf.score(X_test , y_test)
print('随机森林的准确率为:',r)