import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pickle
#import tensorflow as tf
from random import shuffle

labels = {'hard' :1 , 'easy' : 0}
#100 - 115 hard
#120 - 135 hard
# 140 - 185 easy

#300 - 320 easy
#320 -340 easy
#340 - 360 hard


#200 - 220 easy
#220 - 240 easy
#240 - 260 hard

last_col = 6000

def get_label(x) :
    if (x >=140 and x< 180) or (x >=200 and x<240) or (x>=300 and x<340) :
        return labels ['easy']
    return labels['hard']

def read_data(dir , file_path):
    data = pd.read_csv( dir +file_path , header = None, sep='\t')
    df =  data[3:17].T
   # print np.shape(df)
    return df[1:last_col+1]

def save_pickle(dir , files , pklfile):
    datas = []
    y = []
    for x in files :
        file = str(x) + '.csv'
        try :
            open(dir + file, 'r')
        except :
            continue
        temp = read_data(dir , file)
        if np.shape(temp)[0] < last_col :
            print file
            continue
        datas.append(temp)
        y.append(get_label(x))
    z = pd.concat(datas)
    print np.shape(z.values)
    values = z.values
    file_count = len(y)
    print file_count *last_col , np.shape(values)
    X = np.reshape(values, (file_count, last_col, 14))
    Y = np.reshape(y,(file_count,1))
    print np.shape(X) , np.shape(Y)

    with open(pklfile +'.pkl', 'wb') as outp :
        pickle.dump(X, outp,pickle.HIGHEST_PROTOCOL)
        pickle.dump(Y ,outp,pickle.HIGHEST_PROTOCOL)

def open_pickle(pklfile)    :
    with open(pklfile +'.pkl', 'rb') as outp :
        X = pickle.load(outp)
        Y = pickle.load(outp)
    count  = 0
    for a ,b in zip(X,Y) :
        if b == 0 :
            count +=1
    print count
    return X, Y

#train

files = [x for x in range(100,180)]
shuffle(files)
print files
save_pickle('../train_100/' ,files , 'train')
X , Y = open_pickle('train')
print np.shape(X), np.shape(Y)

files = [200, 201, 202 , 203 , 220 , 221 , 241, 260 , 261]
shuffle(files)
print files
save_pickle('../val_200/' ,files , 'val')
X , Y = open_pickle('val')
print np.shape(X), np.shape(Y)

files = [300 , 301 , 320 , 321 , 340 , 341, 360 , 361]
shuffle(files)
print files
save_pickle('../test_300/' ,files , 'test')
X , Y = open_pickle('test')
print np.shape(X), np.shape(Y)



