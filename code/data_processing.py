import numpy as np
import csv
from numpy import array

Y = []
dataset = []
fy = open('yLabels.csv', 'a')
writer = csv.writer(fy)

##########################################################################################

def get_features_GSR(actualValues, sizeData):

	currRow = actualValues[0]
	sampleTime = time.strptime(currRow[0], '%H:%M:%S.%f')
	sampleTime_secs = time_in_seconds(sampleTime)
	i = 0

	while i < sizeData:

		currRow = actualValues[i]
		t = time.strptime(currRow[0], '%H:%M:%S.%f')
		t_secs = time_in_seconds(t)

		if t_secs - sampleTime_secs <= 10:
			if t_secs - sampleTime_secs == 5:
				next_index = i
				temp_sample_time = t_secs
			i += 1
		else:

			# if (t_secs >= 41090 and t_secs <= 41840) or (sampleTime_secs >= 41090 and sampleTime_secs <= 41840):
			# 	writer.writerow('1')
			# else:
			writer.writerow('0')

			i = next_index
			sampleTime_secs = temp_sample_time

def get_all_participants_data():
	suffix = '.csv'
	channel_range = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

	for i in range(1, 2):

		f = open(str(i)+suffix)
		fReader = csv.reader(f)

		j = 1
		for row in fReader:
			if j in channel_range:
				print(j)
				dataset.append(row)
			j = j + 1

	print(len(dataset))
	a = array(dataset)
	print(a[0][0].shape)
		# dataNoLabels = tempDataset[8:]
		# dataset.append(dataNoLabels)

##########################################################################################

if __name__ == '__main__':

	get_all_participants_data()


	# get_features_GSR(actualValues, sizeData)

	fy.close()
