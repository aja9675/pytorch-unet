#!/usr/bin/env python3

'''
This script tracks bees using a custom NN and <unknown> tracking algorithm
'''

import sys
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model import ResNetUNet
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import cv2
from copy import deepcopy
from icecream import ic
# Custom helpers
import helper
# My custom dataset
from bee_dataloader import BeePointDataset
import pickle
import time
import argparse


np.set_printoptions(linewidth=240)

# For inference timing of different components
ENABLE_TIMING = False


def setup_model_dataloader(args, batch_size):
	num_class = 1
	model = ResNetUNet(num_class)
	device = torch.device("cuda")
	model_file = os.path.join(args.model_dir, 'best_model.pth')
	model.load_state_dict(torch.load(model_file, map_location="cuda:0"))
	model.to(device)
	# Set model to the evaluation mode
	model.eval()
	return model, device


def track_bees(args):

	# Handle file not found
	if not os.path.exists(args.video_file_in):
		sys.exit("Error. File not found. (-2)")

	model, device = setup_model_dataloader(args, batch_size=1)

	vid = cv2.VideoCapture(args.video_file_in)
	ic("Processing ", args.video_file_in)

	# Get some metadata
	num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
	width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps =  vid.get(cv2.CAP_PROP_FPS)
	ic("Num frames: ", num_frames)
	ic("Width x Height: ", (width, height))
	# I don't think fps is accurate!
	ic("Source FPS: ", fps)
	if args.fps != 0:
		frame_duration = 1000 // int(args.fps) # ms
	else:
		frame_duration = 1000 // int(fps) # ms


	# Don't actually need the dataset. Just need the input dimensions
	bee_ds = BeePointDataset(root_dir="/dev/null")
	input_size = bee_ds.input_size
	ic(bee_ds.input_size)


	for frame_num in range(num_frames):

		# Read frame from video
		ret, frame = vid.read()
		if frame is None:
			break

		# Rotate if rotation is set
		if args.rotate == -90:
			frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
		elif args.rotate == 90:
			frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
		# Resize to network input size
		frame = cv2.resize(frame, (input_size[1], input_size[0]))

		if 0:
			cv2.imshow("Video", frame)
			key = cv2.waitKey(30)
			if key == 'q' or key == 27:
				break

		# Convert to tensor and forward pass
		tensor_frame = TF.to_tensor(frame)
		tensor_frame.to(device)
		tensor_frame = torch.unsqueeze(tensor_frame, 0)
		pred = helper.model_forward(model, tensor_frame, device, ENABLE_TIMING)

		# Get prediction centroids (predicted points)
		pred_pts = helper.get_centroids(pred)

		if len(pred_pts) > 0:
			if 0: # Draw GT
				for point in points:
					cv2.circle(frame, tuple((point[1],point[0])), 5, (0,255,0), cv2.FILLED)
			for pred_pts in pred_pts:
				cv2.circle(frame, tuple((pred_pts[1],pred_pts[0])), 5, (0,0,255), cv2.FILLED)

		# Show results
		helper.show_image("Predictions", frame, delay=frame_duration)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--video_file_in', default='/data/datasets/bees/bee_videos/best_bee_data/20180522_173523.mp4', type=str, help='Video file to process')
	parser.add_argument('--data_dir', default='/data/datasets/bees/ak_bees/images', type=str, help='Dataset dir')
	parser.add_argument('--model_dir', default='./results/latest', type=str, help='results/<datetime> dir')
	parser.add_argument('--debug', action='store_true', default=False, help='Enable debugging prints and vis')
	parser.add_argument('--fps', type=int, default=0, help='Enable debugging prints and vis')
	parser.add_argument('-r', '--rotate', type=int, default=0, help='Rotate [-90,90')
	parser.set_defaults(func=track_bees)

	args = parser.parse_args()
	args.func(args)

