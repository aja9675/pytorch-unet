#!/usr/bin/env python3

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
ENABLE_TIMING = True


def normalize_uint8(img):
	return cv2.normalize(src=img, dst=None, 
		alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

def reverse_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)

    return inp

def filter_pred(pred):
	filtered = np.zeros_like(pred)
	filtered = np.where(pred > 5, 1, 0)
	return filtered


def mean_shift_centroids(pred):
	# Compute clustering with MeanShift

	if 0:
		# The following bandwidth can be automatically detected using
		bandwidth = estimate_bandwidth(pred, quantile=0.2, n_samples=500)
	else:
		bandwidth = 2

	#ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
	ms = MeanShift(bandwidth=bandwidth, bin_seeding=False)
	ms.fit(pred)
	labels = ms.labels_
	cluster_centers = ms.cluster_centers_

	labels_unique = np.unique(labels)
	n_clusters_ = len(labels_unique)

	print("number of estimated clusters : %d" % n_clusters_)

	if len(cluster_centers) > 0:
		pred_color = cv2.cvtColor(pred_norm, cv2.COLOR_GRAY2BGR)
		for centroid in cluster_centers:
			print(centroid)
			print(centroid.shape)
			cv2.circle(pred_color, centroid, 5, (0,255,0), cv2.FILLED)
		helper.show_image("pred_color", pred_color)


# Test function for playing around with various algorithms
def get_centroids_test(pred):

	# Try mean shift
	#centroids = mean_shift_centroids(pred)
	# Nope

	## Try determinant of hessian
	#from skimage.feature import hessian_matrix_det
	##hess = hessian_matrix_det(pred_norm, sigma=1, approximate=True)
	#hess = hessian_matrix_det(pred_norm, sigma=3, approximate=False)
	## Assume no local minima so I don't have to check fxx
	## Zero out everyone <= 0 (saddle points)
	##hess = np.where(hess <=0, 0, hess)
	## Set all points > 0 to 255
	#hess_norm = np.uint8(np.where(hess > 0, 255, 0))
	##hess_norm = cv2.normalize(src=hess, dst=None, \
	##	alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
	#helper.show_image("hess", hess_norm)

	#import scipy
	##ret = scipy.ndimage.filters.maximum_filter(pred, size=None, footprint=None, output=None, mode='reflect', cval=0.0, origin=0)
	#max_filtered = scipy.ndimage.filters.maximum_filter(pred, size=1)
	#max_filtered = normalize_uint8(max_filtered)
	#helper.show_image("max_filtered", max_filtered)
	return

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
		test_loader = DataLoader(test_ds, batch_size=1, shuffle=True, num_workers=1)
	else:
		# Just use the defaults
		test_loader = DataLoader(bee_ds, batch_size=1, shuffle=True, num_workers=1)

	for _ in range(len(test_ds)):
		inputs, labels = next(iter(test_loader))
		inputs = inputs.to(device)
		labels = labels.to(device)
		# Convert to viewable image
		input_img = inputs.cpu().numpy().squeeze().transpose(1,2,0)
		input_img = cv2.normalize(src=input_img, dst=None, \
			alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
		input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
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
		pred_norm = cv2.normalize(src=pred, dst=None, \
			alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
		#pred_norm = pred * 255
		#helper.show_image("pred", pred_norm)
		cv2.imshow("pred", pred_norm)
		print(np.max(pred_norm))

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
			for centroid in centroids:
				cv2.circle(input_img, tuple((centroid[1],centroid[0])), 5, (0,255,0), cv2.FILLED)
			helper.show_image("Predictions", input_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='/data/datasets/bees/ak_bees/images', type=str, help='Dataset dir')
    parser.add_argument('--model_dir', default='./results/latest', type=str, help='results/<datetime> dir')
    parser.set_defaults(func=inference)
    args = parser.parse_args()
    args.func(args)
