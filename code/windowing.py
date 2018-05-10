import pickle
import utils
import numpy as np

from scipy.signal import argrelmax

##############################################################################################################

def get_spikes(temp_mean_Peaks, temp_num_Peaks, temp_q25_Peaks, temp_q50_Peaks, temp_q75_Peaks, data, index):

	x = data
	t = argrelmax(x)

	spikes = x[t]
	sortedSpikes = np.sort(spikes)

	numPeak = len(spikes)
	meanPeak = np.nan_to_num(np.mean(spikes))

	if len(sortedSpikes) != 0:
		peak25 = np.percentile(sortedSpikes, 25)
		peak50 = np.percentile(sortedSpikes, 50)
		peak75 = np.percentile(sortedSpikes, 75)
	else:
		peak25 = 0
		peak50 = 0
		peak75 = 0

	temp_mean_Peaks[0, index] = meanPeak 
	temp_num_Peaks[0, index] = numPeak
	temp_q25_Peaks[0, index] = peak25
	temp_q50_Peaks[0, index] = peak50
	temp_q75_Peaks[0, index] = peak75

def get_EEG_features(eeg_data_point, y_value):

	meanEEG = np.zeros((1,14))
	minEEG = np.zeros((1, 14))
	maxEEG = np.zeros((1, 14))
	stdEEG = np.zeros((1, 14))
	meanPeaks = np.zeros((1, 14))
	numPeaks = np.zeros((1, 14))
	q25Peaks = np.zeros((1, 14))
	q50Peaks = np.zeros((1, 14))
	q75Peaks = np.zeros((1, 14))
	yTruth = np.zeros(1)

	sizeData = len(eeg_data_point)
	i = 0
	
	while i < sizeData:

		current_window = eeg_data_point[i:i+200, :]

		temp_mean_EEG = np.mean(current_window, axis=0)
		temp_mean_EEG = np.reshape(temp_mean_EEG, (1, 14))
		temp_min_EEG = np.min(current_window, axis=0)
		temp_min_EEG = np.reshape(temp_min_EEG, (1, 14))
		temp_max_EEG = np.min(current_window, axis=0)
		temp_max_EEG = np.reshape(temp_max_EEG, (1, 14))
		temp_std_EEG = np.std(current_window, axis=0, dtype=np.float32)
		temp_std_EEG = np.reshape(temp_std_EEG, (1, 14))

		temp_mean_Peaks = np.zeros((1, 14))
		temp_num_Peaks = np.zeros((1, 14))
		temp_q25_Peaks = np.zeros((1, 14))
		temp_q50_Peaks = np.zeros((1, 14))
		temp_q75_Peaks = np.zeros((1, 14))

		for j in range(0, 14):
			get_spikes(temp_mean_Peaks, temp_num_Peaks, temp_q25_Peaks, temp_q50_Peaks, temp_q75_Peaks, current_window[:, j], j)

		meanEEG = np.vstack((meanEEG, temp_mean_EEG))
		minEEG = np.vstack((minEEG, temp_min_EEG))
		maxEEG = np.vstack((maxEEG, temp_max_EEG))
		stdEEG = np.vstack((stdEEG, temp_std_EEG))
		q25Peaks = np.vstack((q25Peaks, temp_q25_Peaks))
		q50Peaks = np.vstack((q50Peaks, temp_q50_Peaks))
		q75Peaks = np.vstack((q75Peaks, temp_q75_Peaks))
		meanPeaks = np.vstack((meanPeaks, temp_mean_Peaks))
		numPeaks = np.vstack((numPeaks, temp_num_Peaks))
		yTruth = np.vstack((yTruth, y_value))

		i = i + 100

	return meanEEG, minEEG, maxEEG, stdEEG, meanPeaks, numPeaks, q25Peaks, q50Peaks, q75Peaks, yTruth



##############################################################################################################

def get_all_subject_data():

	overall_train_meanEEG = np.zeros((1,14))
	overall_train_minEEG = np.zeros((1, 14))
	overall_train_maxEEG = np.zeros((1, 14))
	overall_train_stdEEG = np.zeros((1, 14))
	overall_train_meanPeaks = np.zeros((1, 14))
	overall_train_numPeaks = np.zeros((1, 14))
	overall_train_q25Peaks = np.zeros((1, 14))
	overall_train_q50Peaks = np.zeros((1, 14))
	overall_train_q75Peaks = np.zeros((1, 14))
	overall_train_yTruth = np.zeros(1)


	overall_test_meanEEG = np.zeros((1,14))
	overall_test_minEEG = np.zeros((1, 14))
	overall_test_maxEEG = np.zeros((1, 14))
	overall_test_stdEEG = np.zeros((1, 14))
	overall_test_meanPeaks = np.zeros((1, 14))
	overall_test_numPeaks = np.zeros((1, 14))
	overall_test_q25Peaks = np.zeros((1, 14))
	overall_test_q50Peaks = np.zeros((1, 14))
	overall_test_q75Peaks = np.zeros((1, 14))
	overall_test_yTruth = np.zeros(1)

	overall_test_sub_meanEEG = np.zeros((1,14))
	overall_test_sub_minEEG = np.zeros((1, 14))
	overall_test_sub_maxEEG = np.zeros((1, 14))
	overall_test_sub_stdEEG = np.zeros((1, 14))
	overall_test_sub_meanPeaks = np.zeros((1, 14))
	overall_test_sub_numPeaks = np.zeros((1, 14))
	overall_test_sub_q25Peaks = np.zeros((1, 14))
	overall_test_sub_q50Peaks = np.zeros((1, 14))
	overall_test_sub_q75Peaks = np.zeros((1, 14))
	overall_test_sub_yTruth = np.zeros(1)

	X_train, Y_train = utils.open_pickle('eng_train')
	X_val, Y_val = utils.open_pickle('eng_val')
	X_test, Y_test = utils.open_pickle('eng_test')
	X_sub_test, Y_sub_test = utils.open_pickle('sub_test')

	for i in range(0, len(X_train)):
		data_point = X_train[i]
		y_val = Y_train[i]

		meanEEG, minEEG, maxEEG, stdEEG, meanPeaks, numPeaks, q25Peaks, q50Peaks, q75Peaks, yTruth = get_EEG_features(data_point, y_val)
		# print(meanEEG.shape)
		# print(q50Peaks.shape)

		overall_train_meanEEG = np.vstack((overall_train_meanEEG, meanEEG))
		overall_train_minEEG = np.vstack((overall_train_minEEG, minEEG))
		overall_train_maxEEG = np.vstack((overall_train_maxEEG, maxEEG))
		overall_train_stdEEG = np.vstack((overall_train_stdEEG, stdEEG))
		overall_train_q25Peaks = np.vstack((overall_train_q25Peaks, q25Peaks))
		overall_train_q50Peaks = np.vstack((overall_train_q50Peaks, q50Peaks))
		overall_train_q75Peaks = np.vstack((overall_train_q75Peaks, q75Peaks))
		overall_train_meanPeaks = np.vstack((overall_train_meanPeaks, meanPeaks))
		overall_train_numPeaks = np.vstack((overall_train_numPeaks, numPeaks))
		overall_train_yTruth = np.vstack((overall_train_yTruth, yTruth))

	for i in range(0, len(X_val)):
		data_point = X_val[i]
		y_val = Y_val[i]

		meanEEG, minEEG, maxEEG, stdEEG, meanPeaks, numPeaks, q25Peaks, q50Peaks, q75Peaks, yTruth = get_EEG_features(data_point, y_val)
		# print(meanEEG.shape)
		# print(q50Peaks.shape)

		overall_train_meanEEG = np.vstack((overall_train_meanEEG, meanEEG))
		overall_train_minEEG = np.vstack((overall_train_minEEG, minEEG))
		overall_train_maxEEG = np.vstack((overall_train_maxEEG, maxEEG))
		overall_train_stdEEG = np.vstack((overall_train_stdEEG, stdEEG))
		overall_train_q25Peaks = np.vstack((overall_train_q25Peaks, q25Peaks))
		overall_train_q50Peaks = np.vstack((overall_train_q50Peaks, q50Peaks))
		overall_train_q75Peaks = np.vstack((overall_train_q75Peaks, q75Peaks))
		overall_train_meanPeaks = np.vstack((overall_train_meanPeaks, meanPeaks))
		overall_train_numPeaks = np.vstack((overall_train_numPeaks, numPeaks))
		overall_train_yTruth = np.vstack((overall_train_yTruth, yTruth))

	for i in range(0, len(X_test)):
		data_point = X_test[i]
		y_val = Y_test[i]

		meanEEG, minEEG, maxEEG, stdEEG, meanPeaks, numPeaks, q25Peaks, q50Peaks, q75Peaks, yTruth = get_EEG_features(data_point, y_val)
		# print(meanEEG.shape)
		# print(q50Peaks.shape)

		overall_test_meanEEG = np.vstack((overall_test_meanEEG, meanEEG))
		overall_test_minEEG = np.vstack((overall_test_minEEG, minEEG))
		overall_test_maxEEG = np.vstack((overall_test_maxEEG, maxEEG))
		overall_test_stdEEG = np.vstack((overall_test_stdEEG, stdEEG))
		overall_test_q25Peaks = np.vstack((overall_test_q25Peaks, q25Peaks))
		overall_test_q50Peaks = np.vstack((overall_test_q50Peaks, q50Peaks))
		overall_test_q75Peaks = np.vstack((overall_test_q75Peaks, q75Peaks))
		overall_test_meanPeaks = np.vstack((overall_test_meanPeaks, meanPeaks))
		overall_test_numPeaks = np.vstack((overall_test_numPeaks, numPeaks))
		overall_test_yTruth = np.vstack((overall_test_yTruth, yTruth))

	for i in range(0, len(X_sub_test)):
		data_point = X_sub_test[i]
		y_val = Y_sub_test[i]

		meanEEG, minEEG, maxEEG, stdEEG, meanPeaks, numPeaks, q25Peaks, q50Peaks, q75Peaks, yTruth = get_EEG_features(data_point, y_val)
		# print(meanEEG.shape)
		# print(q50Peaks.shape)

		overall_test_sub_meanEEG = np.vstack((overall_test_sub_meanEEG, meanEEG))
		overall_test_sub_minEEG = np.vstack((overall_test_sub_minEEG, minEEG))
		overall_test_sub_maxEEG = np.vstack((overall_test_sub_maxEEG, maxEEG))
		overall_test_sub_stdEEG = np.vstack((overall_test_sub_stdEEG, stdEEG))
		overall_test_sub_q25Peaks = np.vstack((overall_test_sub_q25Peaks, q25Peaks))
		overall_test_sub_q50Peaks = np.vstack((overall_test_sub_q50Peaks, q50Peaks))
		overall_test_sub_q75Peaks = np.vstack((overall_test_sub_q75Peaks, q75Peaks))
		overall_test_sub_meanPeaks = np.vstack((overall_test_sub_meanPeaks, meanPeaks))
		overall_test_sub_numPeaks = np.vstack((overall_test_sub_numPeaks, numPeaks))
		overall_test_sub_yTruth = np.vstack((overall_test_sub_yTruth, yTruth))


	# overall_train = np.dstack((overall_train_q75Peaks, overall_train_numPeaks, overall_train_meanPeaks, overall_train_stdEEG, overall_train_maxEEG, overall_train_minEEG))
	# overall_test = np.dstack((overall_test_q75Peaks, overall_test_numPeaks, overall_test_meanPeaks, overall_test_stdEEG, overall_test_maxEEG, overall_test_minEEG))


	return overall_train_maxEEG, overall_train_yTruth, overall_test_maxEEG, overall_test_yTruth, overall_test_sub_maxEEG, overall_test_sub_yTruth

##############################################################################################################

def main():
	X_train, Y_train, X_val, Y_val, X_val_sub, Y_val_sub = get_all_subject_data()

	return X_train, Y_train, X_val, Y_val, X_val_sub, Y_val_sub
