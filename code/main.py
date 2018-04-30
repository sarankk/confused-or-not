import utils
import CNN_Model
import baseline_models

X_train = []
Y_train = []
X_test = []
Y_test = []

def main():
    #data cleaning 
    X_train, X_val, X_test, Y_train, Y_val, Y_test = utils.sample_data()

    #trying CNN model
    # trained_cnn_model = CNN_Model.get_cnn_model(X_train, X_val, X_test, Y_train, Y_val, Y_test)

    #trying one of baseline models
    trained_baseline_model = baseline_models.xgboost(X_train, X_val, Y_train, Y_val)
    
if __name__ == '__main__':
    main()

