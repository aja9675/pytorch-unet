#!/usr/bin/env python3

import sys
import os
import torch
import math
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
from bee_dataloader import BeePointDataset, visualize_bee_points
import pickle
import time
import argparse

# For clustering & centroid detection
# Mean shift?
#from sklearn.cluster import MeanShift, estimate_bandwidth
from scipy import ndimage as ndi
from skimage.feature import peak_local_max


# For inference timing of different components
ENABLE_TIMING = False


def normalize_uint8(img):
	return cv2.normalize(src=img, dst=None, 
		alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

'''
Calcuate centroids from my heatmap
skimage peak_local_max() is doing the work here. It works surprisingly well.
'''
def get_centroids(pred):
	im = np.float32(pred)
	# image_max is the dilation of im with a 20*20 structuring element
	# It is used within peak_local_max function
	if 0:
		# Is this necessary???
		# Looks like a bug in https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_peak_local_max.html#sphx-glr-auto-examples-segmentation-plot-peak-local-max-py
		# because image_max isn't used??
		image_max = ndi.maximum_filter(im, size=20, mode='constant')

	# Comparison between image_max and im to find the coordinates of local maxima
	centroids = peak_local_max(im, min_distance=20)
	#ic(cluster_centers)

	draw = False
	if draw and len(centroids) > 0:
		pred_color = cv2.cvtColor(pred_norm, cv2.COLOR_GRAY2BGR)
		for centroid in centroids:
			cv2.circle(pred_color, tuple((centroid[1],centroid[0])), 5, (0,255,0), cv2.FILLED)
		helper.show_image("pred_color", pred_color)

	return centroids


def inference(args):

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
	bee_ds = BeePointDataset(root_dir='/data/datasets/bees/ak_bees/images/20180522_173523')
	if 1:
		dbfile = open(os.path.join(args.model_dir, 'test_ds.pkl'), 'rb')
		test_ds = pickle.load(dbfile)
		#test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=1)
		test_loader = DataLoader(test_ds, batch_size=1, shuffle=True, num_workers=1, collate_fn=helper.bee_collate_fn)
	else:
		# Just use the defaults
		test_loader = DataLoader(bee_ds, batch_size=1, shuffle=True, num_workers=1)

	for _ in range(len(test_ds)):
		# Because we have variable length points coming from Dataloader, there's some extra overhead
		# in managing the input and GT data. See collate_fn().
		inputs, mask, points = next(iter(test_loader))
		inputs = torch.unsqueeze(inputs[0], 0)
		points = points[0]
		inputs = inputs.to(device)

		# Convert to viewable image
		input_img = inputs.cpu().numpy().squeeze().transpose(1,2,0)
		input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
		input_img = normalize_uint8(input_img)
		#helper.show_image("input_img", input_img)

		start_time = time.time()
		pred = model(inputs)
		# I think sigmoid isn't included in the normal forward pass b/c of the custom BCE+Dice loss
		pred = F.sigmoid(pred)
		pred = pred.data.cpu().numpy()
		pred = pred.squeeze()
		if ENABLE_TIMING:
			print("model forward time: %s s" % (time.time() - start_time))
			# model forward time: 0.04796314239501953 s

		# Threshold heatmap to create binary prediction
		#pred = filter_pred(pred)

		# Normalize for viewing
		pred_norm = normalize_uint8(pred)
		#pred_norm = pred * 255
		#helper.show_image("pred", pred_norm)
		cv2.imshow("pred", pred_norm)
		#print(np.max(pred_norm))

		start_time = time.time()
		centroids = get_centroids(pred)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='/data/datasets/bees/ak_bees/images', type=str, help='Dataset dir')
    parser.add_argument('--model_dir', default='./results/latest', type=str, help='results/<datetime> dir')
    parser.set_defaults(func=inference)
    args = parser.parse_args()
    args.func(args)
