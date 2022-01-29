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

            # skip files that caused error
            if seq_dir in ['./F-PHAB/Video_files/Subject_3/drink_mug/2',
                           './F-PHAB/Video_files/Subject_6/open_soda_can/1',
                           './F-PHAB/Video_files/Subject_6/drink_mug/1']:
                continue
            print(seq_dir)

            if os.path.isfile(os.path.join(skeleton_root, seq_dir.replace('./F-PHAB/Video_files/', ''), 'skeleton_cropped.txt')):
                os.remove(os.path.join(skeleton_root, seq_dir.replace(
                    './F-PHAB/Video_files/', ''), 'skeleton_cropped.txt'))

            depth_dir = os.path.join(seq_dir, 'depth')
            depth_cropped = os.path.join(seq_dir, 'depth_cropped')
            if os.path.isdir(depth_cropped):
                shutil.rmtree(depth_cropped)
            if os.path.isdir(os.path.join(seq_dir, 'pose_cropped')):
                shutil.rmtree(os.path.join(seq_dir, 'pose_cropped'))
            if not os.path.exists(depth_cropped):
                os.makedirs(depth_cropped, exist_ok=True)

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

                # Project skeleton to get center of hand and size
                dpt_intr = np.array([[475.065948, 0, 315.944855],
                                     [0, 475.065857, 245.287079], [0, 0, 1]])

                skel_hom2d = np.array(dpt_intr).dot(skel.transpose(
                    2, 0, 1).reshape(3, -1)).reshape(3, -1, 21).transpose(1, 2, 0)
                skel_proj = (skel_hom2d / skel_hom2d[:, :, 2:])[:, :, :2]

                centerlefttop_pixel = np.array(dpt_intr).dot(
                    centerlefttop.transpose(1, 0)).transpose(1, 0)
                centerlefttop_pixel = (
                    centerlefttop_pixel / centerlefttop_pixel[:, 2:])[:, :2]
                centerrightbottom_pixel = np.array(dpt_intr).dot(
                    centerrightbottom.transpose(1, 0)).transpose(1, 0)
                centerrightbottom_pixel = (
                    centerrightbottom_pixel / centerrightbottom_pixel[:, 2:])[:, :2]

                skel_cropped = []
                top_lefts = []
                orig_sizes = []
                mean_depths = []
                for idx in range(skel_proj.shape[0]):
                    skel_ = skel[idx]
                    proj_ = skel_proj[idx]
                    depth = cv2.imread(os.path.join(
                        depth_dir, 'depth_{:04d}.png'.format(int(skel_order[idx]))), -1)

                    new_xmin = max(centerlefttop_pixel[idx, 0], 0)
                    new_ymin = max(centerrightbottom_pixel[idx, 1], 0)
                    new_xmax = min(
                        centerrightbottom_pixel[idx, 0], depth.shape[1]-1)
                    new_ymax = min(
                        centerlefttop_pixel[idx, 1], depth.shape[0]-1)

                    mean_depth = np.mean(skel_, axis=0)[2]

                    crop = depth[int(new_ymin):int(new_ymax),
                                 int(new_xmin):int(new_xmax)]
                    top_lefts.append([new_xmin, new_ymin])
                    orig_sizes.append([new_xmax-new_xmin, new_ymax-new_ymin])
                    mean_depths.append(mean_depth)
                    output_depth = cv2.resize(
                        crop, (96, 96), interpolation=cv2.INTER_NEAREST)

                    depth_thresh = 150
                    output_depth[np.where(
                        output_depth >= mean_depth+depth_thresh)] = mean_depth
                    output_depth[np.where(
                        output_depth <= mean_depth-depth_thresh)] = mean_depth
                    #output_depth = output_depth - mean_depth

                    cv2.imwrite(os.path.join(depth_cropped, 'depth_{:04d}.png'.format(
                        int(skel_order[idx]))), output_depth)

                    label_xyz = np.ones((21, 3), dtype='float32')
                    label_xyz[:, 0] = (proj_[:, 0].copy() -
                                       top_lefts[-1][0])*96/orig_sizes[-1][0]
                    label_xyz[:, 1] = (proj_[:, 1].copy() -
                                       top_lefts[-1][1])*96/orig_sizes[-1][1]
                    label_xyz[:, 2] = skel_[:, 2].copy()

                    skel_cropped.append(label_xyz.reshape(1, 21, 3))

                    # import random
                    # if random.random()<0.01:
                    # 	fig, ax = plt.subplots()
                    # 	ax.imshow(output_depth)
                    # 	visualize_joints_2d(ax, label_xyz[reorder_idx])
                    # 	plt.show()

                skel_camcoords = np.concatenate(
                    skel_cropped).reshape(skel_order.shape[0], -1)
                skel = np.concatenate([skel_order.reshape(-1, 1),
                                       np.array(top_lefts).reshape(
                                           skel_order.shape[0], -1),
                                       np.array(orig_sizes).reshape(
                                           skel_order.shape[0], -1),
                                       np.array(mean_depths).reshape(
                                           skel_order.shape[0], -1),
                                       skel_camcoords], 1)
                np.savetxt(os.path.join(skeleton_root, seq_dir.replace('./F-PHAB/Video_files/', ''), 'skeleton_cropped.txt'),
                           skel,
                           '%04i '+'%.6f %.6f %.6f %.6f %.6f '+' '.join(['%.6f' for i in range(skel_camcoords.shape[1])]))
            else:
                shutil.copyfile(skeleton_path, os.path.join(skeleton_root, seq_dir.replace(
                    './F-PHAB/Video_files/', ''), 'skeleton_cropped.txt'))
