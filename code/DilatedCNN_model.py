import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.data as data_utils

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
import sys
import copy

import config

#2 conv layers, 2 fc layers
class Dcnn_model(nn.Module):
	def __init__(self):
		super(Dcnn_model, self).__init__()
		self.layer1 = nn.Sequential(
		    nn.Conv1d(3, 16, kernel_size=2, stride=2, dilation=2, padding=1),
		    nn.BatchNorm1d(16),
		    nn.ReLU(),
		    nn.MaxPool1d(2))
		self.layer2 = nn.Sequential(
		    nn.Conv1d(16, 8, kernel_size=2, stride=2, dilation=2, padding=1),
		    nn.BatchNorm1d(8),
		    nn.ReLU())
		self.layer3 = nn.Sequential(
		    nn.Conv1d(8, 8, kernel_size=2, stride=2, dilation=2, padding=1),
		    nn.BatchNorm1d(8),
		    nn.ReLU())
		self.fc1 = nn.Linear(8, 32)
		self.fc2 = nn.Linear(32, 2)
        
	def forward(self, x):
	    out = self.layer1(x)
	    # print(out.size())	
	    out = self.layer2(out)
	    # print(out.size())
	    out = self.layer3(out)
	    # print(out.size())
	    out = out.view(out.size(0), -1)
	    # print(out.size())
	    out = self.fc1(out)
	    # print(out.size())
	    out = self.fc2(out)
	    # print(out.size())
	    return out

def train_model(dataloaders, model, criterion, optimizer, dataset_sizes):

	num_epochs = config.NUM_EPOCHS

	since = time.time()

	t_losses = []
	v_losses = []
	t_accs = []
	v_accs = []

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		# Each epoch has a training and validation phase
		for phase in ['train', 'val']:

			if phase == 'train':
				model.train()  # Set model to training mode
			else:
				model.eval()  # Set model to evaluate mode

			running_loss = 0.0
			running_corrects = 0

			# Iterate over data.
			for i_batch, sampled_batch in enumerate(dataloaders[phase]):

				# get the inputs
				inputs = sampled_batch[0]
				labels = sampled_batch[1]
				check_labels = labels

				# wrap them in Variable
				inputs = Variable(inputs)
				labels = Variable(labels)

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward
				outputs = model(inputs)
				_, preds = torch.max(outputs.data, 1)
				print(outputs)
				print(labels)

				loss = criterion(outputs, labels)

				# backward + optimize only if in training phase
				if phase == 'train':
				  loss.backward()
				  optimizer.step()

				# statistics
				running_loss += loss.data[0] * inputs.size(0)
				running_corrects += (preds == check_labels).sum()

			epoch_loss = running_loss / float(dataset_sizes[phase])
			epoch_acc = running_corrects / float(dataset_sizes[phase])

			if phase == 'train':
				t_losses.append(epoch_loss)
				train_losses = np.array(t_losses)
				t_accs.append(epoch_acc)
				train_accs = np.array(t_accs)
			if phase == 'val':
				v_losses.append(epoch_loss)
				val_losses = np.array(v_losses)
				v_accs.append(epoch_acc)
				val_accs = np.array(v_accs)
					
			print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

	plt.figure(1)
	plt.clf()
	plt.plot(np.arange(epoch+1)+1, train_losses, color='r', label='Train')
	plt.plot(np.arange(epoch+1)+1, val_losses, color='b', label='Val')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	plt.title('CNN - Loss')
	plt.savefig(config.PLOT_CNN_LOSS)

	plt.figure(2)
	plt.clf()
	plt.plot(np.arange(epoch+1)+1, train_accs, color='r', label='Train')
	plt.plot(np.arange(epoch+1)+1, val_accs, color='b', label='Val')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.title('CNN - Accuracy')
	plt.savefig(config.PLOT_CNN_ACC)

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

def get_Dcnn_model(X_train, X_val, X_test, Y_train, Y_val, Y_test):

	dataset_sizes = {}
	dataloaders = {}

	model = Dcnn_model()
	print(Y_train.size())

	# Loss and Optimizer
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

	size1, len1 = X_test.shape
	size2, len2 = X_train.shape
	size3, len3 = X_val.shape

	dataset_sizes['train'] = len(X_train)
	dataset_sizes['val'] = len(X_test) 
	dataset_sizes['test'] = len(X_val)

	X_train = np.reshape(X_train, (size2, 1, len2))
	X_val = np.reshape(X_val, (size3, 1, len3))
	X_test = np.reshape(X_test, (size1, 1, len1))

	X_train = torch.FloatTensor(X_train)
	Y_train = torch.LongTensor(Y_train)
	X_test = torch.FloatTensor(X_test)
	Y_test = torch.LongTensor(Y_test)
	X_val = torch.FloatTensor(X_val)
	Y_val = torch.LongTensor(Y_val)

	print(X_train.shape)
	print(Y_train.shape)

	train_data = data_utils.TensorDataset(X_train, Y_train)
	dataloaders['train'] = data_utils.DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)

	val_data = data_utils.TensorDataset(X_val, Y_val)
	dataloaders['val'] = data_utils.DataLoader(val_data, batch_size=config.BATCH_SIZE, shuffle=True)

	train_model(dataloaders, model, criterion, optimizer, dataset_sizes)

	#testing with test sample
	model.eval()
	inputs = Variable(X_test)
	labels = Variable(Y_test)
	outputs = model(inputs)
	_, preds = torch.max(outputs.data, 1)
	loss = criterion(outputs, labels)
	test_loss = loss.data[0] * inputs.size(0)
	corrects = (preds == Y_test).sum()
	test_acc = corrects / float(dataset_sizes['test'])
	print('{} Loss: {:.4f} Acc: {:.4f}'.format('test', test_loss, test_acc))

	return model



