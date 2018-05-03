import CNN_model
import CNN_model_dilated
import pickle

X_train = []
Y_train = []
X_test = []
Y_test = []

def open_pickle(pklfile)    :
    with open(pklfile +'.pkl', 'rb') as outp :
        X = pickle.load(outp)
        Y = pickle.load(outp)
    return X, Y

def main():
    #getting data
    X_train, Y_train = open_pickle('train')
    X_val, Y_val = open_pickle('test')
    X_test, Y_test = open_pickle('val')
    '''
    X_test = X_train[:8]
    X_train = X_train[8:]
    Y_test = Y_train[:8]
    Y_train = Y_train[8:]
    '''

    trained_cnn_model = CNN_model_dilated.get_cnn_model(X_train, X_val, X_test, Y_train, Y_val, Y_test)

if __name__ == '__main__':
    main()

