import os
import cv2
import time
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt


# train/test split based on original F-PHAB dataset: https://github.com/guiggh/hand_pose_action#hand-pose-estimation
# cross-subject training, as described by Garcia-Hernando et al., is used in the DNBP project
train_shuffle = ['Subject_1', 'Subject_3', 'Subject_4']
test_shuffle = ['Subject_2', 'Subject_5', 'Subject_6']

root_data_file = './F-PHAB/Video_files/'
skeleton_root = './F-PHAB/Hand_pose_annotation_v1'

train = []
for subject_dr in train_shuffle:
    if subject_dr.startswith('.'):
        continue
    subject_dir = os.path.join(root_data_file, subject_dr)
    for action_dr in os.listdir(subject_dir):
        if action_dr.startswith('.'):
            continue
        action_dir = os.path.join(subject_dir, action_dr)
        for seq_dr in os.listdir(action_dir):
            if seq_dr.startswith('.'):
                continue
            seq_dir = os.path.join(action_dir, seq_dr)

            print(seq_dir)

            train.append(seq_dir[2:])


test = []
val = []
for subject_dr in test_shuffle:
    if subject_dr.startswith('.'):
        continue
    subject_dir = os.path.join(root_data_file, subject_dr)
    for action_dr in os.listdir(subject_dir):
        if action_dr.startswith('.'):
            continue
        action_dir = os.path.join(subject_dir, action_dr)
        for seq_dr in os.listdir(action_dir):
            if seq_dr.startswith('.'):
                continue
            seq_dir = os.path.join(action_dir, seq_dr)

            print(seq_dir)

            if random.random() < 0.9:
                test.append(seq_dir[2:])
            else:
                val.append(seq_dir[2:])

with open('train.txt', 'w') as f:
    for item in train:
        f.write(item+'\n')


with open('test.txt', 'w') as f:
    for item in test:
        f.write(item+'\n')

with open('val.txt', 'w') as f:
    for item in val:
        f.write(item+'\n')
