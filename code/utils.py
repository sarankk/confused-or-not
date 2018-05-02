import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pickle

def sample_data():
    #printing filenames within the directory
    # print(os.listdir("eeg_brain"))

    #getting data
    demographic_info = pd.read_csv("data/demographic_info.csv")
    data_eeg = pd.read_csv("data/EEG_data.csv")

    #casting subject ids and ground truth labels to int
    data_eeg['SubjectID'] = data_eeg['SubjectID'].astype(int)
    data_eeg['VideoID'] = data_eeg['VideoID'].astype(int)
    data_eeg['predefinedlabel'] = data_eeg['predefinedlabel'].astype(int)
    data_eeg['user-definedlabeln'] = data_eeg['user-definedlabeln'].astype(int)

    # data_eeg.info() #getting details about the data
    feature_names = ['Attention', 'Mediation', 'Raw', 'Delta',
                'Theta', 'Alpha1', 'Alpha2', 'Beta1', 'Beta2', 'Gamma1', 'Gamma2']

    data = data_eeg.query('(SubjectID != 6) & (SubjectID != 3 | VideoID !=3)') #removing subject 6, data corrupted

    # print data.iloc[:, 2:].describe() #gives all statistics of data like mean max min std 25% 50% 75% count

    #getting features and labels for training
    x_features = data[feature_names]
    y_labels = data['user-definedlabeln']

    #splitting data after normalizing 
    X, X_test, Y, Y_test = train_test_split(x_features, y_labels, test_size=0.1, random_state=42)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=42)
    
    #getting data from pandas object
    X_train = X_train.values
    X_val = X_val.values
    X_test = X_test.values
    Y_train = Y_train.values
    Y_val = Y_val.values
    Y_test = Y_test.values

    return X_train, X_val, X_test, Y_train, Y_val, Y_test

def open_pickle(pklfile)    :
    with open(pklfile +'.pkl', 'rb') as outp :
        X = pickle.load(outp)
        Y = pickle.load(outp)

    return X, Y

    

