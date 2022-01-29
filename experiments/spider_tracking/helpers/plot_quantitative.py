
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





noise_amounts = list(range(5,96,10))



lstm_mu = []
lstm_sig = []
for noise_amount in noise_amounts:
    #####################
    model_folder = "./experiments/spider_tracking/lstm_models"
    epoch_folder = os.path.join(model_folder, "epoch_199")
    ######################

    with open(os.path.join(epoch_folder, 'evaluation_errors_'+str(noise_amount)+'.pkl'),'rb') as f:
        errors = pickle.load(f)
    all_errors = np.concatenate([errors['dynamic_dynamic_noise'], errors['dynamic_static_noise']], axis=-1)
    
    joint_mus=[np.mean(all_errors[0]), np.mean(all_errors[1]), np.mean(all_errors[2]), np.mean(all_errors[3]), np.mean(all_errors[4]), np.mean(all_errors[5]), np.mean(all_errors[6]),  np.mean(all_errors)]
    joint_stds =[np.std(all_errors[0]), np.std(all_errors[1]), np.std(all_errors[2]), np.std(all_errors[3]), np.std(all_errors[4]), np.std(all_errors[5]), np.std(all_errors[6]), np.std(all_errors)]
    lstm_mu.append(joint_mus)
    lstm_sig.append(joint_stds)
lstm_mu = np.array(lstm_mu)
lstm_sig = np.array(lstm_sig)





fig, ax = plt.subplots()
ax.set_xlim(0,100)
ax.set_ylim(0,50)



particle_amounts=[200]
for num_p in particle_amounts:
	lbp_mu = []
	lbp_sig = []
	for noise_amount in noise_amounts:
		#####################
		model_folder = "./experiments/spider_tracking/dnbp_models/"
		epoch_folder = os.path.join(model_folder, "epoch_74")
		######################
		try:
			with open(os.path.join(epoch_folder, str(num_p)+'_particles', 'evaluation_errors_'+str(noise_amount)+'.pkl'),'rb') as f:
				errors = pickle.load(f)
			all_errors = np.concatenate([errors['dynamic_dynamic_noise'], errors['dynamic_static_noise']], axis=-1)

			joint_mus=[np.mean(all_errors[0]), np.mean(all_errors[1]), np.mean(all_errors[2]), np.mean(all_errors[3]), np.mean(all_errors[4]), np.mean(all_errors[5]), np.mean(all_errors[6]),  np.mean(all_errors)]
			joint_stds =[np.std(all_errors[0]), np.std(all_errors[1]), np.std(all_errors[2]), np.std(all_errors[3]), np.std(all_errors[4]), np.std(all_errors[5]), np.std(all_errors[6]), np.std(all_errors)]
		except:
			joint_mus=[0]*8
			joint_stds=[0]*8
		lbp_mu.append(joint_mus)
		lbp_sig.append(joint_stds)
	lbp_mu = np.array(lbp_mu)
	lbp_sig = np.array(lbp_sig)



	v=0.5
	if num_p<200:
		clr = cm.get_cmap(plt.get_cmap('YlOrRd'))(0.15+(v*0.85))[:3]
	elif num_p==200:
		clr = (255/255,50/255,160/255)
	else:
		clr = (255/255,0/255,255/255)

	ax.plot(noise_amounts, lbp_mu[:,7], lw=2, marker='o', label='DNBP', color=clr)
	ax.scatter(x=noise_amounts, y=lbp_mu[:,7], s=20, color=clr)


ax.plot(noise_amounts, lstm_mu[:,7], lw=2, marker='x', linestyle='--', label='LSTM', color='black')
ax.scatter(x=noise_amounts, y=lstm_mu[:,7], s=20, color='black')
ax.fill_between(noise_amounts, lbp_mu[:,7]+lbp_sig[:,7], lbp_mu[:,7]-lbp_sig[:,7], facecolor='pink', alpha=0.75)
ax.fill_between(noise_amounts, lstm_mu[:,7]+lstm_sig[:,7], lstm_mu[:,7]-lstm_sig[:,7], facecolor='gray', alpha=0.4)




ax.set_title(r'Test Evaluation of DNBP and LSTM on Spider Tracking')
ax.set_xlabel('Clutter Ratio')
ax.set_ylabel('Average Euclidean Error (Pixels)')
ax.set_xticks([0,10,20,30,40,50,60,70,80,90,100])
ax.set_xticklabels([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
lgd=plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(alpha=0.5, linestyle='dashed', linewidth=0.5)
fig.savefig('./experiments/spider_tracking/results/euclidean_error_noise_spider.png',bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=100)





