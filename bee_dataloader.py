#!/usr/bin/env python3

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import helper
import glob
from skimage import io, transform
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import torchvision.utils
import torchvision.transforms.functional as TF
import random
from copy import deepcopy
import cv2

# Display size for debugging
disp_size = (1920, 1080) # 1080p
#disp_size = (1280, 720) # 720p


class BeePointDataset(Dataset):
	"""Bee point dataset."""

	def __init__(self, root_dir, sigma=1.75, enable_augmentations=False):
		"""
		Args:
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.root_dir = root_dir
		self.enable_augmentations = enable_augmentations
		# Glob all the label files in root_dir
		self.labels_file_list = []
		if 0:
			# This takes forever!!! (documentation warns against using ** in large directories)
			self.labels_file_list.extend(glob.glob(os.path.join(root_dir, "/**/*.labels"), recursive=True))
		else:
			# This is uglier but much much faster
			for root, dirs, files in os.walk(root_dir):
				for file in files:
					if file.endswith(".labels"):
						self.labels_file_list.append(os.path.join(root, file))
		#self.input_size = (720, 1280) # (rows, cols) - 720p
		# Hacking this so the autoencoder spatial dimensions line up cleanly
		# Both dimensions need to be evenly divisible by 32
		#self.input_size = (736, 1280) # (rows, cols) - 720p
		# 720p had very good results but was probably unnecessarily large
		# Trying to downsize and maintain aspect ration the best I can
		self.input_size = (544, 960) # (rows, cols)
		#self.input_size = (126, 224)
		self.sigma = sigma

	def __len__(self):
		return len(self.labels_file_list)

	def __read_points(self, file_path):
		points = []
		with open(file_path, 'r') as f:
			lines = f.readlines()
		for line in lines:
			point_str = line.split(' ')
			# Note the reversal of points here to switch from OpenCV to numpy coordinate notation
			#point = (int(point_str[0]), int(point_str[1]))
			point = (int(point_str[1]), int(point_str[0]))
			points.append(point)
		points = np.array(points)
		return points

	# Need a custom resizer to handle image resizing and points
	def __resize(self, image, points):
		h, w = image.shape[:2]
		new_h, new_w = self.input_size
		#print(image.shape)
		#print(self.input_size)
		# Handle OpenCV row/col weirdness
		image = cv2.resize(image, (self.input_size[1], self.input_size[0]))
		#print("Orig points: ", points)
		points = points * [new_h / h, new_w / w]
		# Convert back to int
		points = np.int32(points)
		#print("New points: ", points)
		return image, points

	# Need to use a custom transform b/c of the segmentation mask
	def __transform(self, image, mask):
		# Assuming we're already resized at this point
		if self.enable_augmentations:
			# Random crop
			i, j, h, w = transforms.RandomCrop.get_params(
			    image, output_size=(512, 512))
			image = TF.crop(image, i, j, h, w)
			mask = TF.crop(mask, i, j, h, w)
			# Random horizontal flipping
			if random.random() > 0.5:
			    image = TF.hflip(image)
			    mask = TF.hflip(mask)
			# Random vertical flipping
			if random.random() > 0.5:
			    image = TF.vflip(image)
			    mask = TF.vflip(mask)

		# Note that to_tensor converts to float and normalizes 0-1
		image = TF.to_tensor(image)
		mask = TF.to_tensor(mask)

		return image, mask

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		# TODO - if jpg doesn't exist, check for png
		img_name = os.path.splitext(self.labels_file_list[idx])[0] + ".jpg"
		#print("Reading ", img_name)
		image = io.imread(img_name)
		points = self.__read_points(self.labels_file_list[idx])

		# Resize image and points
		image, points = self.__resize(image, points)

		# Create a 2D binary mask with background 0 and our points 1
		mask = np.zeros(image.shape[:2], dtype=np.float32)
		pts_x, pts_y = points.T
		mask[pts_x, pts_y] = 1.0
		if 1:
			# This assert will get triggered if you update input resolution and force
			# you to re-evaluate the Gaussian that represents GT instances
			assert self.input_size == (544, 960), \
				"You changed input resolution, is your Gaussian window and sigma still valid?"
			# With (736, 1280) a bee width is ~17 pix
			mask = cv2.GaussianBlur(mask, ksize=(13,13), sigmaX=self.sigma)
			if 0:
				show_mask = deepcopy(mask)
				show_mask = cv2.normalize(src=show_mask, dst=None, \
					alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
				helper.show_image("Gaussian mask", show_mask)

		image, mask = self.__transform(image, mask)

		sample = [image, mask, points]
		return sample


def visualize_bee_points(image, mask, points):
	# Need to transpose from (3, 720, 1280) tensor to (720, 1280, 3) image
	image = np.asarray(image).transpose(1,2,0)
	mask = np.asarray(mask).squeeze().transpose(0,1)
	#print(image.dtype)
	#print(image.shape)
	#print(mask.dtype)
	#print(mask.shape)
	# Convert back from 0-1.0 to 0-255
	image = cv2.normalize(src=image, dst=None, \
		alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

	#print("Shapes: %s, %s" % (image.shape, mask.shape))
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	mask = cv2.normalize(src=mask, dst=None, \
		alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
	if 0:
		# Dilate for visibility
		# You may or may not want to do this depending on your value of sigma
		mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
	mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
	#helper.show_image("img", image)
	cv2.imshow("img", image)
	helper.show_image("mask", mask)
	#helper.show_image('img mask', cv2.resize(np.hstack((image,mask)), disp_size))


if __name__ == "__main__":
	print("Testing BeePointDataset")

	#bee_ds = BeePointDataset(root_dir='/data/datasets/bees/ak_bees/images/20180522_173523', enable_augmentations=True)
	#bee_ds = BeePointDataset(root_dir='/data/datasets/bees/ak_bees/images/20180522_173523')
	bee_ds = BeePointDataset(root_dir='/data/datasets/bees/ak_bees/images')

	#for i in range(len(bee_ds)):
	for i in range(len(bee_ds)):
		idx = random.randrange(0, len(bee_ds))
		sample = bee_ds[idx]
		visualize_bee_points(*sample)

