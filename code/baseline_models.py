from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import accuracy_score


#various models
def svm_linear(X_train, X_val, Y_train, Y_val):

	model = SVC(kernel='linear', random_state=None)
	evaluate_model(model, X_train, X_val, Y_train, Y_val)

def svm_rbf(X_train, X_val, Y_train, Y_val):

	model = SVC(kernel='rbf', random_state=None)
	evaluate_model(model, X_train, X_val, Y_train, Y_val)

def svm_sig(X_train, X_val, Y_train, Y_val):

	model = SVC(kernel='sigmoid', random_state=None)
	evaluate_model(model, X_train, X_val, Y_train, Y_val)

def knn(X_train, X_val, Y_train, Y_val):

	model = KNeighborsClassifier(n_neighbors=5)
	evaluate_model(model, X_train, X_val, Y_train, Y_val)

def decisionTree(X_train, X_val, Y_train, Y_val):

	model = DecisionTreeClassifier(max_depth=2)
	evaluate_model(model, X_train, X_val, Y_train, Y_val)

def randomForest(X_train, X_val, Y_train, Y_val):

	model = RandomForestClassifier(n_estimators=4, max_depth=5)
	evaluate_model(model, X_train, X_val, Y_train, Y_val)

def logRegression(X_train, X_val, Y_train, Y_val):

	model = LogisticRegression(penalty='l2', random_state=None, max_iter=100)
	evaluate_model(model, X_train, X_val, Y_train, Y_val)

def xgboost(X_train, X_val, Y_train, Y_val):

	model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=10)
	data_matrix = xgb.DMatrix(data=X_train, label=Y_train)

	params = {}
	# Perform cross-validation: cv_results
	cv_results = xgb.cv(dtrain=data_matrix, params=params, nfold=3, num_boost_round=5, metrics="error", as_pandas=True)

	# Print the accuracy
	print("Test Accuracy: {:.5f}".format(((1-cv_results["test-error-mean"]).iloc[-1])))


#evalutating models
def evaluate_model(model, X_train, X_val, Y_train, Y_val):

	kf = KFold(n_splits=5, random_state=None)

	total_score = 0.0

	for train_index, test_index in kf.split(X_train):

		print("Train:", train_index, "Test:", test_index)
		X_batch_train, X_batch_test = X_train[train_index], X_train[test_index]
		Y_batch_train, Y_batch_test = Y_train[train_index], Y_train[test_index]

		model.fit(X_batch_train, Y_batch_train)
		predicted = model.predict(X_batch_test)
		total_score += accuracy_score(Y_batch_test, predicted)

	total_score = total_score / 5
	print('Train accuracy by SVM linear kernel classifier:', total_score)

	#validation score
	predicted = model.predict(X_val)
	print('Validation accuracy for baseline classifier:', accuracy_score(Y_val, predicted))

	return model





