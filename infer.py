#!/usr/bin/env python3

'''
This script either runs inference and visualizes results. Or runs evaluations on the test set.

The following will just run inference and show GT (Green) and predictions (Red).
./infer.py infer

Running the following will visualize results:
infer.py --debug test

Visualization legend:
Blue - GT True Positives
Yellow - Pred True Positives
Green - GT False Negatives
Red - Pred False Positives
'''

import sys
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model import ResNetUNet
import torch.nn.functional as F
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

	# Setup dataset
	# Need to be careful here. This isn't perfect.
	# I'm assuming that the dataset isn't changing between training and inference time
	bee_ds = BeePointDataset(root_dir=args.data_dir)
	if 1:
		dbfile = open(os.path.join(args.model_dir, 'test_ds.pkl'), 'rb')
		test_ds = pickle.load(dbfile)
		#test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=1)
		test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=helper.bee_collate_fn)
	else:
		# Just use the defaults in the bee_ds
		test_loader = DataLoader(bee_ds, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=helper.bee_collate_fn)

	return model, test_loader, device





def inference(args):

	model, dataloder, device = setup_model_dataloader(args, batch_size=1)

	for _ in range(len(dataloder)):
		# Because we have variable length points coming from Dataloader, there's some extra overhead
		# in managing the input and GT data. See collate_fn().
		inputs, mask, points = next(iter(dataloder))
		inputs = torch.unsqueeze(inputs[0], 0)
		points = points[0]
		pred = helper.model_forward(model, inputs, device, ENABLE_TIMING)

		# For visualization, convert to viewable image
		input_img = inputs.cpu().numpy().squeeze().transpose(1,2,0)
		input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
		input_img = helper.normalize_uint8(input_img)
		#helper.show_image("input_img", input_img)

		# Normalize for viewing
		pred_norm = helper.normalize_uint8(pred)
		#pred_norm = pred * 255
		#helper.show_image("pred", pred_norm)
		cv2.imshow("pred", pred_norm)
		#print(np.max(pred_norm))

		start_time = time.time()
		centroids = helper.get_centroids(pred)
		if ENABLE_TIMING:
			print("get_centroids time: %s s" % (time.time() - start_time))
			# get_centroids time: 0.009763717651367188 s

		# Convert pred to color
		pred_norm_color = cv2.cvtColor(pred_norm, cv2.COLOR_GRAY2BGR)
		color_mask = deepcopy(pred_norm_color)
		# Color it by zeroing specific channels
		color_mask[:,:,[0,2]] = 0 # green
		#color_mask[:,:,[0,1]] = 0 # red

		# Create a colored overlay
		overlay = cv2.addWeighted(input_img, 0.5, color_mask, 0.5, 0.0, dtype=cv2.CV_8UC3)
		#helper.show_image("Heatmap Overlay", overlay)
		cv2.imshow("Heatmap Overlay", overlay)

		#stacked = np.hstack((pred_norm_color, overlay))
		#helper.show_image("pred", stacked)

		draw = True
		if draw and len(centroids) > 0:
			if 1: # Draw GT
				for point in points:
					cv2.circle(input_img, tuple((point[1],point[0])), 5, (0,255,0), cv2.FILLED)
			for centroid in centroids:
				cv2.circle(input_img, tuple((centroid[1],centroid[0])), 5, (0,0,255), cv2.FILLED)
			helper.show_image("Predictions", input_img)


def test(args):
	model, dataloder, device = setup_model_dataloader(args, batch_size=1)

	num_tp = 0
	num_fp = 0
	num_fn = 0

	for i in range(len(dataloder)):
		inputs, mask, gt_points = next(iter(dataloder))
		inputs = torch.unsqueeze(inputs[0], 0)
		gt_points = gt_points[0]
		# Forward pass
		pred = helper.model_forward(model, inputs, device, ENABLE_TIMING)
		# Get centroids from resulting heatmap
		pred_pts = helper.get_centroids(pred)

		# Compare pred pts to GT
		pairs = helper.calculate_pairs(pred_pts, gt_points, args.threshold)

		if len(pairs) > 0:
			# Calculate stats on the predictions
			sample_tp, sample_fp, sample_fn = helper.calculate_stats(pred_pts, gt_points, pairs)
			num_tp += sample_tp
			num_fp += sample_fp
			num_fn += sample_fn
			if args.debug:
				ic("TP: ", sample_tp)
				ic("FP: ", sample_fp)
				ic("FN: ", sample_fn)
		elif args.debug:
			print("No matches found")

		if args.debug:
			# For visualization, convert to viewable image
			input_img = inputs.cpu().numpy().squeeze().transpose(1,2,0)
			input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
			input_img = helper.normalize_uint8(input_img)
			#helper.show_image("input_img", input_img)

			# Draw GT in green
			for gt_pt in gt_points:
				cv2.circle(input_img, tuple((gt_pt[1],gt_pt[0])), 5, (0,255,0), cv2.FILLED)
			# Draw all preds in red
			for pred_pt in pred_pts:
				cv2.circle(input_img, tuple((pred_pt[1],pred_pt[0])), 5, (0,0,255), cv2.FILLED)
			# Draw matched preds in yellow, and matched GTs in blue.
			# This will overwrite the red spots for good matches.
			# Note that pairs looks like: [(0, 2), (2, 1), (3, 3), (4, 4), (5, 0)]
			# Where each entry is (gt_idx, pred_idx)
			for pair in pairs:
				gt_pt = gt_points[pair[0]]
				pred_pt = pred_pts[pair[1]]
				cv2.circle(input_img, tuple((gt_pt[1],gt_pt[0])), 5, (255,0,0), cv2.FILLED)
				cv2.circle(input_img, tuple((pred_pt[1],pred_pt[0])), 5, (0,255,255), cv2.FILLED)

			cv2.namedWindow("input_img")
			helper.show_image("input_img", input_img)
			print()



	ic("Confusion matrix:")
	conf_mat_id = np.array([["TP", "FP"],["FN", "TN"]])
	ic(conf_mat_id)
	conf_mat = np.array([[num_tp, num_fp],[num_fn,0]])
	ic(conf_mat)
	precision = num_tp / (num_tp + num_fp)
	recall = num_tp / (num_tp + num_fn)
	f1 = 2* precision * recall / (precision + recall)
	ic("Precision: ", precision)
	ic("Recall: ", recall)
	ic("F1 Score: ", f1)


	model_dir = os.path.abspath(args.model_dir)
	result_file = os.path.join(model_dir, "results.txt")
	with open(result_file, "w") as f:
		f.write("Confusion matrix:\n")
		f.writelines(str(conf_mat_id) + '\n')
		f.writelines(str(conf_mat) + '\n')
		f.writelines("Precision: %f\n" % precision)
		f.writelines("Recall: %f\n" % recall)
		f.writelines("F1 Score: %f\n" % f1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/data/datasets/bees/ak_bees/images', type=str, help='Dataset dir')
    parser.add_argument('--model_dir', default='./results/latest', type=str, help='results/<datetime> dir')
    parser.add_argument('--debug', action='store_true', default=False, help='Enable debugging prints and vis')
    parser.add_argument('-t', '--threshold', type=int, default=15, help='Pixel distance threshold for matching')

    subparsers = parser.add_subparsers()
    parser_infer = subparsers.add_parser('infer')
    parser_infer.set_defaults(func=inference)
    parser_test = subparsers.add_parser('test')
    parser_test.set_defaults(func=test)

    args = parser.parse_args()
    args.func(args)

