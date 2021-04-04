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

import cv2

# Display size for debugging
disp_size = (1920, 1080) # 1080p
#disp_size = (1280, 720) # 720p

# Just for debugging
def show_image(window, img):
	cv2.imshow(window, img)

	# Escape key will exit program
	key = cv2.waitKey(0) & 0xFF
	if key == 27:
		sys.exit(0)


class BeePointDataset(Dataset):
	"""Bee point dataset."""

	def __init__(self, root_dir, enable_augmentations=False):
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
		self.labels_file_list.extend(glob.glob(os.path.join(root_dir, "*.labels")))
		self.input_size = (720, 1280) # (rows, cols) - 720p

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
		# Handle OpenCV row/col weirdness
		image = cv2.resize(image, (self.input_size[1], self.input_size[0]))
		points = points * [new_w / w, new_h / h]
		# Convert back to int
		points = np.int32(points)
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
		# Using from_numpy to keep dtype and not normalize
		mask = torch.from_numpy(mask)

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
		mask = np.zeros(image.shape[:2], dtype=np.uint8)
		pts_x, pts_y = points.T
		mask[pts_x, pts_y] = 1

		image, mask = self.__transform(image, mask)

		sample = [image, mask]

		return sample


def visualize_bee_points(image, mask):
	# Need to transpose from (3, 720, 1280) tensor to (720, 1280, 3) image
	image = np.asarray(image).transpose(1,2,0)
	mask = np.asarray(mask)
	#print(image.dtype)
	#print(image.shape)
	# Convert back from 0-1.0 to 0-255
	image = cv2.normalize(src=image, dst=None, \
		alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

	#print("Shapes: %s, %s" % (image.shape, mask.shape))
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	# Normalize mask, mult is safe b/c I only have 1 class
	mask *= 255
	# Dilate for visibility
	mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
	mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
	show_image('img', cv2.resize(np.hstack((image,mask)), disp_size))


if __name__ == "__main__":
	print("Testing BeePointDataset")

	#bee_ds = BeePointDataset(root_dir='/data/datasets/bees/ak_bees/images/20180522_173523', enable_augmentations=True)
	bee_ds = BeePointDataset(root_dir='/data/datasets/bees/ak_bees/images/20180522_173523')

	#for i in range(len(bee_ds)):
	for i in range(4):
		sample = bee_ds[i]
		visualize_bee_points(*sample)

