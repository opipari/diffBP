import os
import pickle

import numpy as np
import cv2
from skimage import io
import random


nb_samples = 0
mean_es = 0
std_es = 0


min_part = np.ones((3))*1000
max_part = -np.ones((3))*1000


h = 480
w = 640
mode = 'train'

data_path = "."

with open(os.path.join(mode+'.txt'), 'r') as f:
    sequence_paths = f.read().splitlines()


for seq_idx in range(len(sequence_paths)):
    chosen_path = sequence_paths[seq_idx]
    seq_len = len(os.listdir(os.path.join(data_path, chosen_path, 'color')))

    skeleton_path = os.path.join(data_path, chosen_path,
                                 'pose', 'skeleton.txt')
    skeleton_vals = np.loadtxt(skeleton_path)
    if skeleton_vals.size > 0:
        skel_order = skeleton_vals[:, 0]
        skel = skeleton_vals[:, 1:].reshape(-1, 21, 3)
        min_part = np.minimum(min_part, skel.min(axis=0).min(axis=0))
        max_part = np.maximum(max_part, skel.max(axis=0).max(axis=0))

    skeleton_path = os.path.join(data_path, chosen_path,
                                 'pose_cropped', 'skeleton_cropped.txt')
    skeleton_vals = np.loadtxt(skeleton_path)
    if skeleton_vals.size > 0:
        skel_order = skeleton_vals[:, 0]
        skel = skeleton_vals[:, 6:].reshape(-1, 21, 3)
        labels = skel
        labels[:, :, 2] = labels[:, :, 2]/1000
        labels[:, :, 0] = (labels[:, :, 0]-48)/96
        labels[:, :, 1] = (labels[:, :, 1]-48)/96

    for idx in range(seq_len):
        img_file = os.path.join(data_path, chosen_path,
                                'color', 'color_'+f'{idx:04}'+'.jpeg')

        #image = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
        #image = cv2.resize(image, (w, h))
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        depth_file = os.path.join(data_path, chosen_path,
                                  'depth_cropped', 'depth_'+f'{idx:04}'+'.png')
        depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        depth = depth - skeleton_vals[idx, 5]
        mean_es += np.mean(depth)
        std_es += np.std(depth)
        nb_samples += 1

        if nb_samples % 100 == 0:
            print(seq_idx, len(sequence_paths),
                  mean_es/nb_samples, std_es/nb_samples)

print(mean_es/nb_samples, std_es/nb_samples)
print(min_part)
print(max_part)
