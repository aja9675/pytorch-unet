'''
Miscellaneous helper functions
'''

import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

# For clustering & centroid detection
from scipy import ndimage as ndi
from skimage.feature import peak_local_max


def show_image(window, img, delay=0):
    cv2.imshow(window, img)

    # Escape key will exit program
    key = cv2.waitKey(delay) & 0xFF
    if key == 27:
        sys.exit(0)


def plot_img_array(img_array, ncol=3):
    nrow = len(img_array) // ncol

    f, plots = plt.subplots(nrow, ncol, sharex='all', sharey='all', figsize=(ncol * 4, nrow * 4))

    for i in range(len(img_array)):
        plots[i // ncol, i % ncol]
        plots[i // ncol, i % ncol].imshow(img_array[i])

from functools import reduce
def plot_side_by_side(img_arrays):
    flatten_list = reduce(lambda x,y: x+y, zip(*img_arrays))

    plot_img_array(np.array(flatten_list), ncol=len(img_arrays))

import itertools
def plot_errors(results_dict, title):
    markers = itertools.cycle(('+', 'x', 'o'))

    plt.title('{}'.format(title))

    for label, result in sorted(results_dict.items()):
        plt.plot(result, marker=next(markers), label=label)
        plt.ylabel('dice_coef')
        plt.xlabel('epoch')
        plt.legend(loc=3, bbox_to_anchor=(1, 0))

    plt.show()

def masks_to_colorimg(masks):
    colors = np.asarray([(201, 58, 64), (242, 207, 1), (0, 152, 75), (101, 172, 228),(56, 34, 132), (160, 194, 56)])

    colorimg = np.ones((masks.shape[1], masks.shape[2], 3), dtype=np.float32) * 255
    channels, height, width = masks.shape

    for y in range(height):
        for x in range(width):
            selected_colors = colors[masks[:,y,x] > 0.5]

            if len(selected_colors) > 0:
                colorimg[y,x,:] = np.mean(selected_colors, axis=0)

    return colorimg.astype(np.uint8)

def bee_collate_fn(batch):
    data = [item[0] for item in batch]
    mask = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    return [data, mask, labels]

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

def model_forward(model, batch_imgs, device, enable_timing):
    batch_imgs = batch_imgs.to(device)
    if enable_timing:
        start_time = time.time()
    pred = model(batch_imgs)
    # I think sigmoid isn't included in the normal forward pass b/c of the custom BCE+Dice loss
    pred = torch.sigmoid(pred)
    pred = pred.data.cpu().numpy()
    pred = pred.squeeze()
    if enable_timing:
        print("model forward time: %s s" % (time.time() - start_time))
        # model forward time: 0.04796314239501953 s
    return pred
