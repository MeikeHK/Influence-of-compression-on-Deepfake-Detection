from XceptionNet import Xception
import torch.optim as optim
import torch.nn as nn
import os
import cv2
import torch
import numpy as np
import json
import random
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

XceptionNet = Xception().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(XceptionNet.parameters(), lr=0.0001, momentum=0.9)

#Load model from git, note that no optimizer and such have been saved in this case
#checkpoint = torch.load('./XceptionNet.pkl', map_location=torch.device('cpu'))
#XceptionNet.load_state_dict(checkpoint)

#Load saved (best) model to continue training
checkpoint = torch.load('XceptionNet/XceptionNet_best.pkl')
XceptionNet.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch_n = checkpoint['epoch']
loss = checkpoint['loss']

train_set = './training_set/training_set' #on Lisa
with open (train_set + '/labels_shorter.json') as json_file:
	labels = json.load(json_file)

	#shuffling the labels dict
	labels_list = list(labels.items())
	random.shuffle(labels_list)
	labels = dict(labels_list)

	best_model = XceptionNet.state_dict()
	best_loss = loss

	for epoch in range(epoch_n, epoch_n + 5):  # loop over the dataset multiple times
		total_len_train = 0
		total_correct_train = 0

		print('Starting Epoch:', epoch+1)

		running_loss = 0.0
		count = 0

		for x in range(len(labels)):
			if x % 4 != 0:
				continue;
			else:
				input_frames = []
				y = []
				#work in batches of 4 videos
				for i in range(4):
					if x+i < len(list(labels)):
						new_frames = []
						file = list(labels)[x+i]
						cap= cv2.VideoCapture(train_set + '/' + file)
						z=0
						while(cap.isOpened()):
							ret, frame = cap.read()
							if ret == False:
								break
							desired_frame = cv2.resize(frame, (299,299), interpolation = cv2.INTER_AREA)
							new_frames.append(desired_frame)
							z+=1
							#only read the first 100 frames
							if z == 100:
								break
						cap.release()
						cv2.destroyAllWindows()
						input_frames.extend(new_frames)

						label = labels[file]
						if labels[file] == 1:
							y_new = [1] * len(new_frames)
						elif labels[file]== 0:
							y_new = [0] * len(new_frames)
						y.extend(y_new)

				input_frames = np.array(input_frames)
				input_frames = np.transpose(input_frames, (0,3,1,2))
				input_frames = torch.FloatTensor(input_frames).to(device)
				y = torch.tensor(y, dtype=torch.long).to(device)

				total_len_train += len(y)

				#shuffling the loaded frames
				indices = np.arange(y.shape[0])
				np.random.shuffle(indices)
				y = y[indices]
				input_frames = input_frames[indices]

				#trains in batches
				batch_size = math.floor(len(y)/16)
				for j in range(16):
					# zero the parameter gradients
					optimizer.zero_grad()

					# forward + backward + optimize
					outputs = XceptionNet(input_frames[(j*batch_size):((j+1)*batch_size)])
					_, predictions = torch.max(outputs.data, 1)
					total_correct_train += (y[(j*batch_size):((j+1)*batch_size)] == predictions).sum().item()
					loss = criterion(outputs, y[(j*batch_size):((j+1)*batch_size)])
					loss.backward()
					optimizer.step()

					# print statistics
					running_loss += loss.item()

					if j % 4 == 0:
						count += 1
						if count % 100 == 99: #print statistics
							accuracy_last = total_correct_train / total_len_train
							print('[%d, %5d] loss: %.3f' %(epoch + 1, count + 1, running_loss / 100), 'and accuracy: ', accuracy_last)
							total_len_train = 0
							total_correct_train = 0
							running_loss = 0.0
					if loss < best_loss: #Save new model if it is better than the currently saved model
						best_loss = loss
						best_model = XceptionNet.state_dict()


		val_set = './validation_set' #on Lisa
		with open (val_set + '/labels_shorter.json') as json_file_val:
			labels_val = json.load(json_file_val)
			total_correct = 0
			total_len = 0

			XceptionNet.load_state_dict(best_model)

			for x in range( len(labels_val)):
				file = list(labels_val)[x]
				input_frames = []
				cap= cv2.VideoCapture(val_set + '/' + file)
				i=0
				while(cap.isOpened()):
					ret, frame = cap.read()
					if ret == False:
						break
					desired_frame = cv2.resize(frame, (299,299), interpolation = cv2.INTER_AREA)
					input_frames.append(desired_frame)
					i+=1
				cap.release()
				cv2.destroyAllWindows()
				input_frames = np.array(input_frames)
				input_frames = np.transpose(input_frames, (0,3,1,2))

				label = labels_val[file]
				if labels_val[file] == 1:
					y_val = torch.ones([input_frames.shape[0]], dtype=torch.long).to(device)
				elif labels_val[file]== 0:
					y_val = torch.zeros([input_frames.shape[0]], dtype=torch.long).to(device)
				total_len += input_frames.shape[0]

				input_frames = torch.FloatTensor(input_frames).to(device)
				batch_size = math.floor(len(y_val)/40)
				for h in range(40):

					output_val = XceptionNet(input_frames[(h*batch_size):((h+1)*batch_size)])
					_, predictions = torch.max(output_val.data, 1)
					total_correct += (y_val[(h*batch_size):((h+1)*batch_size)] == predictions).sum().item()


			accuracy = total_correct / total_len
			print('The accuracy on validations set is: ', accuracy)
			y_val = []
			labels_val = {}

	XceptionNet.load_state_dict(best_model)

	#save the best model
	torch.save({'epoch': epoch, 'model_state_dict': best_model, 'optimizer_state_dict': optimizer.state_dict(), 'loss': best_loss}, 'XceptionNet/XceptionNet_best.pkl')
