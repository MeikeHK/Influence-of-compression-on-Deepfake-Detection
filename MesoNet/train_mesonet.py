from MesoNet import Meso4, MesoInception4
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

MesoNet = Meso4().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(MesoNet.parameters(), lr=0.00001, momentum=0.9)

#Load the uploaded model, no optimizer settings and such are included.
#checkpoint = torch.load('./MesoNet.pkl', map_location=torch.device('cpu'))
#MesoNet.load_state_dict(checkpoint)

#Load the previous model.
checkpoint = torch.load('MesoNet/MesoNet_best.pkl')
MesoNet.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch_n = checkpoint['epoch']
loss = checkpoint['loss']

train_set = './training_set/training_set'
with open (train_set + '/labels_shorter.json') as json_file:
	labels = json.load(json_file)

	#shuffling the labels dict
	labels_list = list(labels.items())
	random.shuffle(labels_list)
	labels = dict(labels_list)

	#Setting comparison values of previously best model.
	best_model = MesoNet.state_dict()
	best_loss = loss

	for epoch in range(epoch_n, epoch_n + 9):  # loop over the dataset multiple times
		total_len_train = 0
		total_correct_train = 0
		running_loss = 0.0
		count = 0

		#loop over all files
		for x in range(len(labels)):
			if x % 4 != 0:
				continue
			else:
				input_frames = []
				y = []
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
							desired_frame = cv2.resize(frame, (256,256), interpolation = cv2.INTER_AREA)
							new_frames.append(desired_frame)
							z+=1
							#Only read the first 100 frames
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

				total_len_train += y.shape[0]

				#shuffling the loaded frames
				indices = np.arange(y.shape[0])
				np.random.shuffle(indices)
				y = y[indices]
				input_frames = input_frames[indices]

				batch_size = math.floor(len(y)/4)
				for j in range(4):
					# zero the parameter gradients
					optimizer.zero_grad()

					# forward + backward + optimize
					outputs = MesoNet(input_frames[(j*batch_size):((j+1)*batch_size)])
					_, predictions = torch.max(outputs.data, 1)
					total_correct_train += (y[(j*batch_size):((j+1)*batch_size)] == predictions).sum().item()
					loss = criterion(outputs, y[(j*batch_size):((j+1)*batch_size)])
					loss.backward()
					optimizer.step()

					running_loss += loss.item()

					count += 1
					if count % 100 == 99: #print the statistics
						accuracy_last = total_correct_train / total_len_train
						print('[%d, %5d] loss: %.3f' %(epoch + 1, count + 1, running_loss / 100), 'and accuracy: ', accuracy_last)
						total_len_train = 0
						total_correct_train = 0
						running_loss = 0.0
					if loss.item() < best_loss: #Save model if it is better than the previously saved model
						best_loss = loss.item()
						best_model = MesoNet.state_dict()

		#validation
		val_set = './validation_set' #on Lisa
		with open (val_set + '/labels_shorter.json') as json_file_val:
			labels_val = json.load(json_file_val)
			total_correct = 0
			total_len = 0

			MesoNet.load_state_dict(best_model)

			for x in range( len(labels_val)):
				file = list(labels_val)[x]
				input_frames = []
				cap= cv2.VideoCapture(val_set + '/' + file)
				i=0
				while(cap.isOpened()):
					ret, frame = cap.read()
					if ret == False:
						break
					desired_frame = cv2.resize(frame, (256,256), interpolation = cv2.INTER_AREA)
					input_frames.append(desired_frame)
					i+=1
				cap.release()
				cv2.destroyAllWindows()
				input_frames = np.array(input_frames)
				input_frames = np.transpose(input_frames, (0,3,1,2))

				if labels_val[file] == 1:
					y_val = torch.ones([input_frames.shape[0]], dtype=torch.long).to(device)
				elif labels_val[file]== 0:
					y_val = torch.zeros([input_frames.shape[0]], dtype=torch.long).to(device)
				total_len += len(y_val)

				input_frames = torch.FloatTensor(input_frames).to(device)

				output_val = MesoNet(input_frames)
				_, predictions = torch.max(output_val.data, 1)
				total_correct += (y_val == predictions).sum().item()

			accuracy = total_correct / total_len
			print('The accuracy on validations set is: ', accuracy)

	MesoNet.load_state_dict(best_model)

	#Save best model in .pkl file, so it can be used again later.
	torch.save({'epoch': epoch, 'model_state_dict': best_model, 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}, 'MesoNet/MesoNet_best.pkl')
