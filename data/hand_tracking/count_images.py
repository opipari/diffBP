import os
import cv2
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt

from open3d import *
total = 0

root_data_file = './F-PHAB/Video_files/'
skeleton_root = './F-PHAB/Hand_pose_annotation_v1'
for subject_dr in ['Subject_2', 'Subject_5', 'Subject_6']:
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

            color = os.path.join(seq_dir, 'color')
            if os.path.isdir(color):
                total += len(os.listdir(color))
print(total)

#105459, 50403
