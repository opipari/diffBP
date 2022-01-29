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


found = False

root_data_file = './F-PHAB/Video_files/'
skeleton_root = './F-PHAB/Hand_pose_annotation_v1'
for subject_dr in ['Subject_6']:
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

            if seq_dir == './F-PHAB/Video_files/Subject_6/handshake/1':
                found = True
            if not found:
                continue
            # skip files that caused error
            if seq_dir in ['./F-PHAB/Video_files/Subject_3/drink_mug/2',
                           './F-PHAB/Video_files/Subject_6/open_soda_can/1',
                           './F-PHAB/Video_files/Subject_6/drink_mug/1',
                           './F-PHAB/Video_files/Subject_5/give_card/3',
                           './F-PHAB/Video_files/Subject_5/handshake/4',
                           './F-PHAB/Video_files/Subject_5/receive_coin/1',
                           './F-PHAB/Video_files/Subject_6/handshake/6',
                           './F-PHAB/Video_files/Subject_6/handshake/5']:
                continue
            print(seq_dir)

            color_dir = os.path.join(seq_dir, 'color')
            color_cropped = os.path.join(seq_dir, 'color_cropped')
            if os.path.isdir(color_cropped):
                shutil.rmtree(color_cropped)
            if not os.path.exists(color_cropped):
                os.makedirs(color_cropped, exist_ok=True)

            skeleton_path = os.path.join(skeleton_root, seq_dir.replace(
                './F-PHAB/Video_files/', ''), 'skeleton.txt')

            skeleton_vals = np.loadtxt(skeleton_path)
            if skeleton_vals.size > 0:
                skel_order = skeleton_vals[:, 0]
                skel = skeleton_vals[:, 1:].reshape(
                    skeleton_vals.shape[0], 21, -1)
                skel = skel.astype(np.float32)

                centerlefttop = np.mean(skel, axis=1)
                centerlefttop[:, 0] -= 100
                centerlefttop[:, 1] += 100

                centerrightbottom = np.mean(skel, axis=1)
                centerrightbottom[:, 0] += 100
                centerrightbottom[:, 1] -= 100

                centerlefttop_camcoords = cam_extr.dot(
                    np.concatenate([centerlefttop, np.ones([centerlefttop.shape[0], 1])], 1).transpose()).transpose()[:, :3].astype(np.float32)
                centerlefttop_pixel = np.array(cam_intr).dot(
                    centerlefttop_camcoords.transpose()).transpose()
                centerlefttop_pixel = (
                    centerlefttop_pixel / centerlefttop_pixel[:, 2:])[:, :2]

                centerrightbottom_camcoords = cam_extr.dot(
                    np.concatenate([centerrightbottom, np.ones([centerrightbottom.shape[0], 1])], 1).transpose()).transpose()[:, :3].astype(np.float32)
                centerrightbottom_pixel = np.array(cam_intr).dot(
                    centerrightbottom_camcoords.transpose()).transpose()
                centerrightbottom_pixel = (
                    centerrightbottom_pixel / centerrightbottom_pixel[:, 2:])[:, :2]

                for idx in range(centerrightbottom_camcoords.shape[0]):
                    color = cv2.imread(os.path.join(
                        color_dir, 'color_{:04d}.jpeg'.format(int(skel_order[idx]))), -1)

                    new_xmin = max(centerlefttop_pixel[idx, 0], 0)
                    new_ymin = max(centerrightbottom_pixel[idx, 1], 0)
                    new_xmax = min(
                        centerrightbottom_pixel[idx, 0], color.shape[1]-1)
                    new_ymax = min(
                        centerlefttop_pixel[idx, 1], color.shape[0]-1)

                    crop = color[int(new_ymin):int(new_ymax),
                                 int(new_xmin):int(new_xmax)]
                    output_color = cv2.resize(
                        crop, (96, 96), interpolation=cv2.INTER_NEAREST)

                    cv2.imwrite(os.path.join(color_cropped, 'color_{:04d}.jpeg'.format(
                        int(skel_order[idx]))), output_color)
