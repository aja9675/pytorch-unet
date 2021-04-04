#!/usr/bin/env python3

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


class BeePointDataset(Dataset):
    """Bee point dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        # Glob all the label files in root_dir
        self.labels_file_list = []
        self.labels_file_list.extend(glob.glob(os.path.join(root_dir, "*.labels")))

    def __len__(self):
        return len(self.labels_file_list)

    def __read_points(self, file_path):
        points = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            point_str = line.split(' ')
            point = (int(point_str[0]), int(point_str[1]))
            points.append(point)
        return points

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # TODO - if jpg doesn't exist, check for png
        img_name = os.path.splitext(self.labels_file_list[idx])[0] + ".jpg"
        image = io.imread(img_name)
        points = self.__read_points(self.labels_file_list[idx])
        points = np.array(points)
        #points = points.astype('float').reshape(-1, 2)
        sample = {'image': image, 'points': points}

        if self.transform:
            sample = self.transform(sample)

        return sample


def show_landmarks(image, points):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(points[:, 0], points[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated


if __name__ == "__main__":
    print("Testing BeePointDataset")

    bee_ds = BeePointDataset(root_dir='/data/datasets/bees/ak_bees/images/20180522_173523')


    for i in range(len(bee_ds)):
        sample = bee_ds[i]

        print(i, sample['image'].shape, sample['points'].shape)

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        show_landmarks(**sample)

        if i == 3:
            plt.show()
            break
