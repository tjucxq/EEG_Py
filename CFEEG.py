import xlsxwriter
import xlrd
import sys
import os

import numpy as np

from collections import defaultdict
'''
    [Delta,Theta,Low Alpha,High Alpha,Low Beta,High Beta,Low Gamma,Mid Gamma]

'''

wave_dict={0:'Delta',1:'Theta',2:'Low Alpha',3:'High Alpha',4:'Low Beta',5:'High Beta',6:'Low Gamma',7:'Mid Gamma'}

'''
    row=(start,end)
    col=(start,end)
'''
def read_data(path,sheet,*,row,col,target_col):

    result=defaultdict()

    data=xlrd.open_workbook(path)
    work_sheet=data.sheet_by_name(sheet)

    row_start=row[0]
    if len(row)==1:
        row_end=work_sheet.nrows-1
    else:
        row_end=row[1]

    col_start=col[0]
    if len(col)==1:
        col_end=work_sheet.ncols-1
    else:
        col_end=col[1]


    header=work_sheet.row_values(row_start)[col_start:col_end]
    result['header']=header

    size_row=row_end-row_start
    size_col=col_end-col_start

    dataSet=np.zeros((size_row,size_col))

    for i in range(1,size_row+1):
        dataSet[i-1,:]=work_sheet.row_values(i)[col_start:col_end]

    result['train_data']=dataSet

    target=work_sheet.col_values(target_col,row_start+1,row_end+1)
    result['target']=target

    return result

'''
    resultDict:defaultdict
    condition:List<Tuple>
'''

def data_processing(resultDict,condition):

    result=defaultdict()
    size=len(condition)

    header=[]
    for i in range(size):
        if len(condition[i])==1:
            header.append(wave_dict[condition[i][0]])
        else:
            tmp=wave_dict[condition[i][0]]+'/'+wave_dict[condition[i][1]]
            header.append(tmp)
    result['header']=header

    target=resultDict['target']
    result['target']=target

    data = resultDict['train_data']
    shape = np.shape(data)

    dataSet=np.zeros((shape[0],size),dtype=float)

    for i in range(size):
        if len(condition[i])==1:
            dataSet[:,i]=data[:,condition[i][0]]
        else:
            dataSet[:,i]=data[:,condition[i][0]]/data[:,condition[i][1]]
    result['train_data']=dataSet

    return result

'''
    抽取取调节前后的数据
'''

from com.cn.SW_EEG import separate
def split_data(resultDcit):
    result=defaultdict()
    target=resultDcit['target']
    [F,S]=separate(target)
    # print(F,'----indexF')
    dataSet=resultDcit['train_data']
    data=np.r_[dataSet[0:F+1],dataSet[S+1:]]
    F_target=target[0:F+1]
    S_target=[(i+1) for i in target[S+1:]]
    new_target=F_target+S_target
    result['header']=resultDcit['header']
    result['train_data']=data
    result['target']=new_target
    return result

'''
滑动窗口滑动数据
'''

def mergeSlipper(resultDict,windows_size,step):
    result=defaultdict()

    t=resultDict['target']
    [F,S]=separate(t)

    dataSet=resultDict['train_data']
    Fdata=dataSet[0:F+1]
    Sdata=dataSet[S+1:]

    FAdata=slipper(Fdata,windows_size,step)
    SAdata=slipper(Sdata,windows_size,step)
    new_data=np.r_[FAdata,SAdata]

    Fr,Fc=np.shape(FAdata)
    Sr,Sc=np.shape(SAdata)

    target=[0 for i in range(Fr)]+[1 for i in range(Sr)]

    h=resultDict['header']
    header=[]
    for i in h:
        th='std of '+i
        header.append(th)
    for i in h:
        th='median of '+i
        header.append(th)
    for i in h:
        th='mean of '+i
        header.append(th)
    for i in h:
        th='min of '+i
        header.append(th)
    for i in h:
        th='max of '+i
        header.append(th)

    result['header']=header
    result['train_data']=new_data
    result['target']=target
    return result

def slipper(dataSet,windows_size,step):

    r,c=np.shape(dataSet)

    index=0
    temp1=np.std(dataSet[index:windows_size,:],axis=0)
    temp2=np.median(dataSet[index:windows_size,:],axis=0)
    temp3=np.mean(dataSet[index:windows_size,:],axis=0)
    temp4=np.min(dataSet[index:windows_size,:],axis=0)
    temp5=np.max(dataSet[index:windows_size,:],axis=0)

    temp=np.r_[temp1,temp2,temp3,temp4,temp5]
    # print(np.shape(temp.T))
    index+=step

    while index+windows_size<r:
        tempC=np.r_[np.std(dataSet[index:index+windows_size,:],axis=0),np.median(dataSet[index:index+windows_size,:],axis=0),np.mean(dataSet[index:index+windows_size,:],axis=0),
                    np.min(dataSet[index:index + windows_size, :], axis=0),np.max(dataSet[index:index+windows_size,:],axis=0)]
        temp=np.c_[temp,tempC]
        index+=step

    return temp.T

'''
    合并多个文件组成一个大的数据集
'''
def mergeData(paths,sheets,rows,cols,target_cols):

    resultCollect=[]

    resultDict=defaultdict()


    for i in range(len(paths)):
        result=read_data(paths[i],sheets[i],row=rows[i],col=cols[i],target_col=target_cols[i])
        resultCollect.append(split_data(result))

    header=resultCollect[i]['header']

    temp=resultCollect[0]['train_data']
    target=[]
    for i in range(1,len(resultCollect)):
        temp=np.r_[temp,resultCollect[i]['train_data']]

    for result in resultCollect:
        target=target+result['target']

    resultDict['header']=header
    resultDict['target']=target
    resultDict['train_data']=temp
    return resultDict



def compute_corrcoef(resultDict):

    result=defaultdict()
    data=resultDict['train_data']
    dataSet=np.corrcoef(data.T)

    result['corrcoef']=dataSet

    header=resultDict['header']
    result['header']=header

    return result


def write_data(resultDict,sheet,path):

    corrcoef_matrix=resultDict['corrcoef']
    header=resultDict['header']

    workbook=xlsxwriter.Workbook(path)
    work_sheet=workbook.add_worksheet(sheet)
    work_sheet.write_row('A1',header)

    shape=np.shape(corrcoef_matrix)

    for i in range(shape[0]):
        position='A'+str(i+2)
        work_sheet.write_row(position,corrcoef_matrix[i,:])

    workbook.close()


import os
def mergeFile(path,sheet,*,row,col,target_col,window_size,step):

    paths=os.listdir(path)
    result=defaultdict()
    result['target']=[]
    result['train_data']=None

    for filename in paths:
        print(os.path.join(path,filename),'开始处理！！！')

        dataset1=read_data(os.path.join(path,filename),sheet,row=row,col=col,target_col=target_col)
        new_data=mergeSlipper(dataset1,window_size,step)
        result['header']=new_data['header']
        result['target']=result['target']+new_data['target']
        if result['train_data']==None:
            result['train_data']=new_data['train_data']
        else:
            result['train_data']=np.r_[result['train_data'],new_data['train_data']]

        print(os.path.join(path,filename),'处理成功！！！')

    return result


























