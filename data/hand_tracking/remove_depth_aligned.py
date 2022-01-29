import os
import cv2
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt

from open3d import *


# Display utilities
def visualize_joints_2d(ax, joints, joint_idxs=True, links=None, alpha=1):
    """Draw 2d skeleton on matplotlib axis"""
    if links is None:
        links = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
                 (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]
    # Scatter hand joints on image
    x = joints[:, 0]
    y = joints[:, 1]
    ax.scatter(x, y, 1, 'r')

    # Add idx labels to joints
    for row_idx, row in enumerate(joints):
        if joint_idxs:
            plt.annotate(str(row_idx), (row[0], row[1]))
    _draw2djoints(ax, joints, links, alpha=alpha)


def _draw2djoints(ax, annots, links, alpha=1):
    """Draw segments, one color per link"""
    colors = ['r', 'm', 'b', 'c', 'g']

    for finger_idx, finger_links in enumerate(links):
        for idx in range(len(finger_links) - 1):
            _draw2dseg(
                ax,
                annots,
                finger_links[idx],
                finger_links[idx + 1],
                c=colors[finger_idx],
                alpha=alpha)


def _draw2dseg(ax, annot, idx1, idx2, c='r', alpha=1):
    """Draw segment of given color"""
    ax.plot(
        [annot[idx1, 0], annot[idx2, 0]], [annot[idx1, 1], annot[idx2, 1]],
        c=c,
        alpha=alpha)


reorder_idx = np.array([
    0, 1, 6, 7, 8, 2, 9, 10, 11, 3, 12, 13, 14, 4, 15, 16, 17, 5, 18, 19,
    20
])


cam_extr = np.array(
    [[0.999988496304, -0.00468848412856, 0.000982563360594,
      25.7], [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
     [-0.000969709653873, 0.00274303671904, 0.99999576807,
      3.902], [0, 0, 0, 1]])
cam_intr = np.array([[1395.749023, 0, 935.732544],
                     [0, 1395.749268, 540.681030], [0, 0, 1]])

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

            print(seq_dir)

            depth_aligned = os.path.join(seq_dir, 'depth_aligned')
            if os.path.isdir(depth_aligned):
                shutil.rmtree(depth_aligned)
