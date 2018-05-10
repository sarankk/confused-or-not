import utils
import CNN_model
import DilatedCNN_model
import baseline_models
import windowing

import numpy as np
from numpy import inf

def main():
    #getting data
    # X_train, X_val, X_test, Y_trian, Y_val, Y_test = utils.sample_data()
    # X_train, Y_train = utils.open_pickle('train')
    '''
    count = 0
    acount =0
    for a , b in zip(X_train, Y_train):
    	if b == 0 :
    		count+=1
    	else :
    		acount+=1
    print(acount, count)
    '''
    # X_val, Y_val = utils.open_pickle('val')
    # X_test, Y_test = utils.open_pickle('test')
 

    # print(X_train.shape)
    # print(X_val.shape)
    # print(X_test.shape)

    #trying CNN model
    # trained_cnn_model = CNN_model.get_cnn_model(X_train, X_val, X_test, Y_train, Y_val, Y_test)

    #trying Dilated CNN model
    # dilated_cnn_model = DilatedCNN_model.get_Dcnn_model(X_train, X_val, X_test, Y_train, Y_val, Y_test)


    #getting data
    X_train, Y_train, X_val, Y_val, X_val_sub, Y_val_sub = windowing.main()

    X_train = np.nan_to_num(X_train)
    X_val = np.nan_to_num(X_val)
    X_val_sub = np.nan_to_num(X_val_sub)
    Y_train = np.nan_to_num(Y_train)
    Y_val = np.nan_to_num(Y_val)
    Y_val_sub = np.nan_to_num(Y_val_sub)

    X_train[X_train == -inf] = 0
    X_val[X_val == -inf] = 0
    Y_train[Y_train == -inf] = 0
    Y_val[Y_val == -inf] = 0
    X_val_sub[X_val_sub == -inf] = 0
    Y_val_sub[Y_val_sub == -inf] = 0
    #trying one of baseline models
    trained_baseline_model = baseline_models.svm_rbf(X_train, X_val, X_val_sub, Y_train, Y_val, Y_val_sub)
    trained_baseline_model = baseline_models.svm_sig(X_train, X_val, X_val_sub, Y_train, Y_val, Y_val_sub)
    trained_baseline_model = baseline_models.knn(X_train, X_val, X_val_sub, Y_train, Y_val, Y_val_sub)
    trained_baseline_model = baseline_models.decisionTree(X_train, X_val, X_val_sub, Y_train, Y_val, Y_val_sub)
    trained_baseline_model = baseline_models.randomForest(X_train, X_val, X_val_sub, Y_train, Y_val, Y_val_sub)
    trained_baseline_model = baseline_models.logRegression(X_train, X_val, X_val_sub, Y_train, Y_val, Y_val_sub)

    #trying one of baseline models
    # trained_baseline_model = baseline_models.xgboost(X_train, X_val, X_val_sub, Y_train, Y_val, Y_val_sub)
    
if __name__ == '__main__':
    main()

