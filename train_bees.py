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
from collections import defaultdict
import torch.nn.functional as F
from loss import dice_loss
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import cv2
from model import ResNetUNet
from torchsummary import summary
import pickle
import argparse
from icecream import ic

# My custom dataset
from bee_dataloader import BeePointDataset, visualize_bee_points

from datetime import datetime
current_datetime = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


def calc_loss(pred, target, metrics, bce_weight=0.5):
	bce = F.binary_cross_entropy_with_logits(pred, target)

	pred = torch.sigmoid(pred)
	dice = dice_loss(pred, target)

	loss = bce * bce_weight + dice * (1 - bce_weight)

	metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
	metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
	metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

	return loss

def print_metrics(metrics, epoch_samples, phase):
	outputs = []
	for k in metrics.keys():
		outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

	print("{}: {}".format(phase, ", ".join(outputs)))

def train_model(device, model, optimizer, scheduler, dataloaders, num_epochs, out_dir, no_results):
	best_model_wts = copy.deepcopy(model.state_dict())
	best_loss = 1e10

	checkpoint_dir = os.path.join(out_dir, current_datetime)

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		since = time.time()

		# Each epoch has a training and validation phase
		for phase in ['train', 'val']:
			if phase == 'train':
				scheduler.step()
				for param_group in optimizer.param_groups:
					print("LR", param_group['lr'])

				model.train()  # Set model to training mode
			else:
				model.eval()   # Set model to evaluate mode

			metrics = defaultdict(float)
			epoch_samples = 0

			# Dataloader returns both mask and GT points
			# Currently, GT points are only used at test time for generating metrics
			for i, (inputs, labels, points) in enumerate(dataloaders[phase]):
				inputs = torch.stack(inputs).to(device)
				labels = torch.stack(labels).to(device)
				# points are not used!!

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward
				# track history if only in train
				with torch.set_grad_enabled(phase == 'train'):
					# Enable automatic mixed precision
					#with torch.cuda.amp.autocast():
					outputs = model(inputs)
					loss = calc_loss(outputs, labels, metrics)

					# backward + optimize only if in training phase
					if phase == 'train':
						loss.backward()
						optimizer.step()

				# statistics
				epoch_samples += inputs.size(0)

				if i % 100:
					print_metrics(metrics, epoch_samples, phase)

			print_metrics(metrics, epoch_samples, phase)
			epoch_loss = metrics['loss'] / epoch_samples

			# deep copy the model
			if phase == 'val' and epoch_loss < best_loss and not no_results:
				print("saving best model")
				best_loss = epoch_loss
				best_model_wts = copy.deepcopy(model.state_dict())

				print("saving best model to: ", checkpoint_dir)
				torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))

		#if epoch % 10 and True:
		if epoch % 1 == 0 and not no_results:
			checkpoint_filename = 'checkpoint_' + str(epoch) + '.pth'
			print("Saving epoch checkpoint: ", checkpoint_filename)
			torch.save(model.state_dict(), os.path.join(checkpoint_dir, checkpoint_filename))

		time_elapsed = time.time() - since
		print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

	print('Best val loss: {:4f}'.format(best_loss))

	# load best model weights
	model.load_state_dict(best_model_wts)
	return model

# Helper for pickling loaders, mostly used for testing on the right test data
def pickle_datasets(out_dir, train, val, test):

	train_path = os.path.join(out_dir, 'train_ds.pkl')
	dbfile = open(train_path, 'wb')
	pickle.dump(train, dbfile)
	dbfile.close()
	val_path = os.path.join(out_dir, 'val_ds.pkl')
	dbfile = open(val_path, 'wb')
	pickle.dump(val, dbfile)
	dbfile.close()
	test_path = os.path.join(out_dir, 'test_ds.pkl')
	dbfile = open(test_path, 'wb')
	pickle.dump(test, dbfile)
	dbfile.close()


def train_bees(args):
	batch_size = 4
	num_epochs = 50
	learning_rate = 1e-4

	if args.no_results:
		print("WARNING. --no_results specified. Not saving any results.")

	# Setup dataset
	bee_ds = BeePointDataset(root_dir=args.data_dir)
	# Split dataset
	dataset_len = len(bee_ds)
	print("Total dataset length: ", dataset_len)
	val_split = 0.20
	test_split = 0.10
	val_size = int(val_split * dataset_len)
	test_size = int(test_split * dataset_len)
	train_size = dataset_len - test_size - val_size
	print("Num Train/eval/test: %i/%i/%i" % (train_size, val_size, test_size))
	train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(bee_ds, [train_size, val_size, test_size])
	#print(train_dataset.indices)
	#print(val_dataset.indices)
	#print(test_dataset.indices)

	if not args.no_results:
		out_dir = os.path.abspath(args.out_dir)
		# Delete existing 'results/latest' symlink if it exists
		latest_symlink = os.path.join(out_dir, 'latest')
		if os.path.exists(latest_symlink):
			os.remove(latest_symlink)

		# Create a new results directiory
		out_dir = os.path.join(out_dir, current_datetime)
		if not os.path.exists(out_dir):
			os.mkdir(out_dir)
		os.symlink(out_dir, latest_symlink)

		# Pickle our dataset lists so we can resume from them later (namely for testing)
		pickle_datasets(out_dir, train_dataset, val_dataset, test_dataset)

	# Create dataloaders
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
								num_workers=10, collate_fn=helper.bee_collate_fn)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
								num_workers=10, collate_fn=helper.bee_collate_fn)
	#test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=True,
	#							num_workers=10, collate_fn=helper.bee_collate_fn)
	dataloaders = { 'train': train_loader, 'val': val_loader }

	if 0:
		# Verify dataloader is working
		for i_batch, sample_batched in enumerate(val_loader):
			for i in range(batch_size):
				visualize_bee_points(sample_batched[0][i], sample_batched[1][i])

	# Train

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print(device)

	num_class = 1
	model = ResNetUNet(num_class).to(device)

	# check keras-like model summary using torchsummary
	#summary(model, input_size=(3, 544, 960))

	# freeze backbone layers
	#for l in model.base_layers:
	#    for param in l.parameters():
	#        param.requires_grad = False

	optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

	exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

	model = train_model(device, model, optimizer_ft, exp_lr_scheduler, dataloaders, num_epochs, args.out_dir, args.no_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='/data/datasets/bees/ak_bees/images', type=str, help='Dataset dir')
    parser.add_argument('--out_dir', default='./results', type=str, help='Results dir (will create a new folder from current datetime')
    parser.add_argument('--no_results', action='store_true', default=False, help='Dont save results (for debugging)')
    parser.set_defaults(func=train_bees)
    args = parser.parse_args()
    args.func(args)


