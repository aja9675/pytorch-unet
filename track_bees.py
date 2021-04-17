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
# Visualization color
VIS_COLOR = (0,0,255)

'''
Draw the 'tails' of the tracking system
point_pairs is an array of arrays of points. Each entry represents a unique instance.
'''
def draw_tracking_lines(image, point_pairs):
	# For each set of points
	for pairs in point_pairs:
		for pts in pairs:
			# Draw a line from the previous to the current detected point
			cv2.line(image, tuple((pts[0][1], pts[0][0])), tuple((pts[1][1], pts[1][0])), VIS_COLOR, 2)
	return image

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

	# Vars for tracking
	prev_pts = np.array([])
	running_pairs = []

	# Vars for profiling
	total_exec_time_accumulator = 0
	exec_time_accumulator = 0
	num_exec_frames = 0

	for frame_num in range(num_frames):

		# Read frame from video
		ret, frame = vid.read()
		if frame is None:
			break

		if args.enable_timing:
			start_time = time.time()

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
		num_bees = len(pred_pts)

		# Since what we really care about is # bees, stop time profiling here
		# The rest is for visualization
		if args.enable_timing and frame_num > 0:
			iter_time = time.time() - start_time
			total_exec_time_accumulator += iter_time
			exec_time_accumulator += iter_time
			num_exec_frames += 1
			if frame_num % int(fps) == 0:
				avg_exec_time = exec_time_accumulator / num_exec_frames
				ic("Avg exec time: ", avg_exec_time)
				exec_time_accumulator = 0
				num_exec_frames = 0

		# Use bipartite graph minimum weight matching to associate detections
		if len(pred_pts) > 0 and len(prev_pts) > 0 and not args.disable_tracking:
			pairs = helper.calculate_pairs(prev_pts, pred_pts, args.threshold)
			# Extract actual points based on indices in original arrays
			point_pairs = []
			for pair in pairs:
				point_pairs.append((prev_pts[pair[1]], pred_pts[pair[0]]))

			running_pairs.append(point_pairs)
			if len(running_pairs) > args.tracker_frame_len:
				running_pairs.pop(0)

			# Draw the tracking lines
			frame = draw_tracking_lines(frame, running_pairs)
		elif len(running_pairs) > 0:
			running_pairs.pop(0)

		if len(pred_pts) > 0:
			for pred_pt in pred_pts:
				cv2.circle(frame, tuple((pred_pt[1],pred_pt[0])), 5, VIS_COLOR, cv2.FILLED)

		# Draw # of bees in text on bottom left of image
		num_bees_text = "# Bees: %i" % num_bees
		bottom_left = (0, frame.shape[0]-20)
		frame = cv2.putText(frame, num_bees_text, org=bottom_left, fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                   fontScale=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

		# Show results
		helper.show_image("Predictions", frame, delay=frame_duration)

		prev_pts = deepcopy(pred_pts)

	# Calculate the average processing fps. num_frames-1 b/c we didnt' count the first frame initialization delay
	avg_exec_time = total_exec_time_accumulator / (num_frames-1)
	measured_fps = 1 / avg_exec_time
	ic("Overall avg exec time: ", avg_exec_time)
	ic("Overall FPS: ", measured_fps)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--video_file_in', default='/data/datasets/bees/bee_videos/best_bee_data/20180522_173523.mp4', type=str, help='Video file to process')
	parser.add_argument('--data_dir', default='/data/datasets/bees/ak_bees/images', type=str, help='Dataset dir')
	parser.add_argument('--model_dir', default='./results/latest', type=str, help='results/<datetime> dir')
	parser.add_argument('--fps', type=int, default=0, help='Enable debugging prints and vis')
	parser.add_argument('-r', '--rotate', type=int, default=0, help='Rotate [-90,90')
	parser.add_argument('--disable_tracking', action='store_true', default=False, help='Disable tracking')
	parser.add_argument('-t', '--threshold', type=int, default=50, help='Distance threshold for tracking')
	parser.add_argument('--tracker_frame_len', type=int, default=15, help='Number of frames to show tracking visualization')
	parser.add_argument('--enable_timing', action='store_true', default=False, help='Enables performance timing')
	parser.set_defaults(func=track_bees)

	args = parser.parse_args()
	args.func(args)

