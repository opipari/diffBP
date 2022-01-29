
import os
import sys
import time
import shutil
import json

import numpy as np
from PIL import Image

from matplotlib import cm
import matplotlib.pyplot as plt
from skimage import io

from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torch.utils.data import Dataset, DataLoader


import pickle





thresh_amounts = list(range(0,81,5))








fig, ax = plt.subplots()
ax.set_xlim(0,80)
ax.set_ylim(0,100)


estim_errors = pickle.load(open('./experiments/hand_tracking/dnbp_models/epoch_16/evaluation_estim.pkl',"rb"))
track_errors = pickle.load(open('./experiments/hand_tracking/dnbp_models/epoch_16/evaluation_track.pkl',"rb"))

#			   [0  5   10   15   20   25   30  35  40  45    50  55    60    65    70    75    80]
baseline_prc = [0, 55, 65, 73.5, 80, 85.5, 89, 92, 94, 95.7, 97, 97.9, 98.1, 99, 99.5, 99.7, 99.9]
estim_prc = []
track_prc = []
for thresh in thresh_amounts:
	estim_prc.append(100*np.sum(estim_errors['errors']<thresh)/estim_errors['errors'].shape[0])
	track_prc.append(100*np.sum(track_errors['errors']<thresh)/track_errors['errors'].shape[0])


ax.plot(thresh_amounts, estim_prc, lw=2, marker='x', linestyle='--', label='DNBP (Estimation)', color='tab:orange')
ax.plot(thresh_amounts, track_prc, lw=2, marker='x', linestyle='-.', label='DNBP (Tracking)', color='tab:green')
ax.plot(thresh_amounts, baseline_prc, lw=2, marker='x', linestyle='-', label='Hernando et al. (Estimation)', color='tab:blue')

ax.set_title(r'Test Evaluation of DNBP on Hand Tracking')
ax.set_xlabel('Error Threshold (mm)')
ax.set_ylabel('Percentage of Frames with error < Threshold')
ax.set_xticks([0,10,20,30,40,50,60,70,80])
ax.set_xticklabels([0,10,20,30,40,50,60,70,80])
ax.set_yticks([0,10,20,30,40,50,60,70,80,90,100])
ax.set_yticklabels([0,10,20,30,40,50,60,70,80,90,100])
lgd=plt.legend(bbox_to_anchor=(1.05, 1), loc='best')
ax.grid(alpha=0.5, linestyle='dashed', linewidth=0.5)
fig.savefig('./experiments/hand_tracking/results/percent_hand.png', bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=100)

