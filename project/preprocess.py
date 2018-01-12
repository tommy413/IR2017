import csv
import os
import numpy as np
from keras.utils import to_categorical

def split_data(X_data,Y_data,split_ratio):
    indices = np.arange(X_data.shape[0])  
    np.random.shuffle(indices) 

    num_validation_sample = int(split_ratio * X_data.shape[0] )

    ix_train = indices[num_validation_sample:]
    ix_val = indices[:num_validation_sample]

    return (X_data[ix_train],Y_data[ix_train]),(X_data[ix_val],Y_data[ix_val])

def padding(arr,maxlen):
    for line in arr:
        while len(line) < maxlen:
            line.append(0)
        if len(line) > maxlen:
            line = line[:maxlen]
    return arr

def get_data(data_name,split = True, to_onehot = True):
    data_path = os.path.join("dataset",data_name)

    dict_file = open(os.path.join(data_path,"dictionary.csv"),'r')
    data_file = open(os.path.join(data_path,"traindata.csv"),'r')
    test_file = open(os.path.join(data_path,"testdata.csv"),'r')

    dict_wf = csv.reader(dict_file)
    data_wf = csv.reader(data_file)
    test_wf = csv.reader(test_file)

    dictionary = dict()
    for line in list(dict_wf)[1:]:
        dictionary[int(line[0])] = line[1]
    wordsize = len(dictionary.keys())

    X = []
    Y = []
    test_X = []
    test_Y = []
    maxlen = 0
    for line in list(data_wf)[1:]:
        docid = int(line[0])
        label = int(line[1])
        sen = [int(x) for x in line[2:]]
        maxlen = max(maxlen,len(sen))
        X.append(sen)
        Y.append(label)

    for line in list(test_wf)[1:]:
        docid = int(line[0])
        label = int(line[1])
        sen = [int(x) for x in line[2:]]
        test_X.append(sen)
        test_Y.append([docid,label])

    X = padding(X,maxlen)
    test_X = padding(test_X,maxlen)

    X = np.array(X)
    test_X = np.array(test_X)
    Y = np.array(Y)
    if to_onehot:
        Y = to_categorical(Y,num_classes=13)

    if split :
        (X,Y),(Xval,Yval) = split_data(X,Y,0.2)

        return X,Y,Xval,Yval,test_X,test_Y,dictionary,maxlen,wordsize
    else :
        return X,Y,test_X,test_Y,dictionary,maxlen,wordsize
