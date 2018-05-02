import utils
import CNN_model
import DilatedCNN_model
import baseline_models

X_train = []
Y_train = []
X_test = []
Y_test = []

def main():
    #getting data
    # X_train, X_val, X_test, Y_trian, Y_val, Y_test = utils.sample_data()
    X_train, Y_train = utils.open_pickle('train')
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
    X_val, Y_val = utils.open_pickle('val')
    X_test, Y_test = utils.open_pickle('test')
 
 	
    # print(X_train.shape)
    # print(X_val.shape)
    # print(X_test.shape)

    #trying CNN model
    trained_cnn_model = CNN_model.get_cnn_model(X_train, X_val, X_test, Y_train, Y_val, Y_test)

    #trying Dilated CNN model
    # dilated_cnn_model = DilatedCNN_model.get_Dcnn_model(X_train, X_val, X_test, Y_train, Y_val, Y_test)

    #trying one of baseline models
    # trained_baseline_model = baseline_models.xgboost(X_train, X_val, Y_train, Y_val)
    
if __name__ == '__main__':
    main()

