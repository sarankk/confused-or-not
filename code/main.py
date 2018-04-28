import utils
import CNN_Model

X_train = []
Y_train = []
X_test = []
Y_test = []

def main():
    #data cleaning 
    X_train, X_val, X_test, Y_train, Y_val, Y_test = utils.sample_data()

    # print(X_train.shape, X_test.shape)

    trained_cnn_model = CNN_Model.get_cnn_model(X_train, X_val, X_test, Y_train, Y_val, Y_test)
    
if __name__ == '__main__':
    main()

