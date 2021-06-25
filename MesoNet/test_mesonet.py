from MesoNet import Meso4, MesoInception4
import torch.optim as optim
import torch.nn as nn
import os
import cv2
import torch
import numpy as np
import json
import random

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

#testing
test_set = './test_set'
with open (test_set + '/labels_shorter.json') as json_file_test:

	#load the best model
	MesoNet = Meso4().to(device)
	checkpoint = torch.load('./MesoNet.pkl', map_location=torch.device('cpu'))
	MesoNet.load_state_dict(checkpoint)
	MesoNet.eval()

	labels_test = json.load(json_file_test)

	for x in range(len(labels_test)):
		file = list(labels_test)[x]

		index = int(file.split('_')[0])
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

		#constucts tensor of the known ouputs, also updates variables of interest
		label = labels_test[file]
		if labels_test[file] == 1:
			y_test = torch.ones([input_frames.shape[0]], dtype=torch.long).to(device)
			total_df += len(y_test)
			len_per_set[index] = len_per_set[index] + len(y_test)
		elif labels_test[file]== 0:
			y_test = torch.zeros([input_frames.shape[0]], dtype=torch.long).to(device)
			total_non_df += len(y_test)
			len_per_set[index] = len_per_set[index] + len(y_test)

		#gets predictions for all frames
		output_val = MesoNet(input_frames)
		_, predictions = torch.max(output_val.data, 1)
		detected_df += torch.sum(predictions)
		detected_non_df += len(predictions) - torch.sum(predictions)

		#checks how many predictions were correct, and stores this.
		correct_subparts[index] = correct_subparts[index] + (y_test == predictions).sum().item()

	#print the statistics, eerste twee kunnen weg als het programmatje goed genoeg is :)
	print('correct per set: ', correct_subparts)
	print('total amount of frames: ', len_per_set)
	print('The accuracy on test set is: ', sum(correct_subparts) / sum(len_per_set))
	print('accuracies per set:', np.divide(correct_subparts, len_per_set))
	print('deepfakes in set', total_df, 'of which', detected_df, 'were detected')
	print('originals in set', total_non_df, 'of which', detected_non_df, 'were detected')
