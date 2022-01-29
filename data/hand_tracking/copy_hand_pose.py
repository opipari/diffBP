import os
import cv2
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import minimum_filter, maximum_filter, median_filter, gaussian_filter


root_data_file = './F-PHAB/Video_files/'
skeleton_root = './F-PHAB/Hand_pose_annotation_v1'
for subject_dr in ['Subject_1', 'Subject_2', 'Subject_3', 'Subject_4', 'Subject_5', 'Subject_6']:
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
            if seq_dir in ['./F-PHAB/Video_files/Subject_3/drink_mug/2',
                           './F-PHAB/Video_files/Subject_6/open_soda_can/1',
                           './F-PHAB/Video_files/Subject_6/drink_mug/1']:
                continue
            print(seq_dir)

            pose = os.path.join(seq_dir, 'pose')
            if not os.path.exists(pose):
                os.makedirs(pose, exist_ok=True)

            skeleton_path = os.path.join(skeleton_root, seq_dir.replace(
                './F-PHAB/Video_files/', ''), 'skeleton.txt')
            shutil.copyfile(skeleton_path, os.path.join(pose, 'skeleton.txt'))

            pose_cropped = os.path.join(seq_dir, 'pose_cropped')
            if not os.path.exists(pose_cropped):
                os.makedirs(pose_cropped, exist_ok=True)

            skeleton_path = os.path.join(skeleton_root, seq_dir.replace(
                './F-PHAB/Video_files/', ''), 'skeleton_cropped.txt')
            shutil.copyfile(skeleton_path, os.path.join(
                pose_cropped, 'skeleton_cropped.txt'))
