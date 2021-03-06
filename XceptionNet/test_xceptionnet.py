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

#run on GPU is possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#set initial lists
correct_subparts = [0,0,0,0,0]
len_per_set = [0,0,0,0,0]
detected_df = 0
detected_non_df = 0
total_df = 0
total_non_df = 0

#Testomg
test_set = './test_set'
with open (test_set + '/labels_shorter.json') as json_file_test:

	#load the best model
	XceptionNet = Xception().to(device)

	checkpoint = torch.load('./XceptionNet/XceptionNet.pkl', map_location=torch.device('cpu'))
	XceptionNet.load_state_dict(checkpoint)
	XceptionNet.eval()

	labels_test = json.load(json_file_test)

	for x in range(len(labels_test)):
		file = list(labels_test)[x]

		input_frames = []
		y = []
		predictions = []

		#takes all frames from the video
		cap= cv2.VideoCapture(test_set + '/' + file)
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

		#constructs the correct x input
		input_frames = np.array(input_frames)
		input_frames = np.transpose(input_frames, (0,3,1,2))
		input_frames = torch.FloatTensor(input_frames).to(device)

		#constucts tensor of the known ouputs
		label = labels_test[file]
		if labels_test[file] == 1:
			y_test = torch.ones([input_frames.shape[0]], dtype=torch.long).to(device)
			total_df += len(y_test)
		elif labels_test[file]== 0:
			y_test = torch.zeros([input_frames.shape[0]], dtype=torch.long).to(device)
			total_non_df += len(y_test)

		#Create 30 batches of frames
		batch_size = math.floor(len(y_test)/30)
		for j in range(30):

			#gets predictions for all frames
			output_val = XceptionNet(input_frames[(j*batch_size):((j+1)*batch_size)])
			_, predictions = torch.max(output_val.data, 1)
			detected_df += torch.sum(predictions)
			detected_non_df += len(predictions) - torch.sum(predictions)

			#checks how many predictions were correct, and stores this.
			index = int(file.split('_')[0])
			correct_subparts[index] = correct_subparts[index] + (y_test[(j*batch_size):((j+1)*batch_size)] == predictions).sum().item()
			len_per_set[index] = len_per_set[index] + batch_size

	#print the statistics, eerste twee kunnen weg als het programmatje goed genoeg is :)
	print('correct per set: ', correct_subparts)
	print('total amount of frames: ', len_per_set)
	print('The accuracy on test set is: ', sum(correct_subparts) / sum(len_per_set))
	print('accuracies per set:', np.divide(correct_subparts, len_per_set))
	print('deepfakes detected', detected_df, 'of', total_df)
	print('original detected', detected_non_df, 'of', total_non_df)
