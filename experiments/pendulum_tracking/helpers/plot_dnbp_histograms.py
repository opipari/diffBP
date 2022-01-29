import os
import sys
import time
import shutil
import json
from tqdm import tqdm

import numpy as np
from PIL import Image


import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from skimage import io

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torch.utils.data import Dataset, DataLoader


import pickle

from diffBP.networks.dnbp_synthetic import factors, dnbp
from diffBP.datasets import articulated_toy_dataset, articulated_transforms
from diffBP.datasets import pendulum_plotting as pend_plot



training_folder = "./experiments/pendulum_tracking"
epoch = "epoch_74"



device = torch.device("cuda:0")

config_file = os.path.join(training_folder,"dnbp_config_file.json")
config = json.load(open(config_file, 'r'))


graph = torch.tensor(config["graph"])
edge_set = torch.tensor(config["edge_set"])


if "density_std" in config.keys():
    density_std = config["density_std"]
else:
    density_std = config["std"]


if config["precision"]=="float32":
    data_type = torch.float32
else:
    data_type = torch.double

bpn = dnbp.DNBP(graph, edge_set, config["inc_nghbrs"], 
                  particle_count=config["train_particle_count"],
                  shared_feats=config["shared_feats"],
                  enc_hidden_feats_tot=config["enc_hidden_feats_tot"],
                  enc_output_feats_tot=config["enc_output_feats_tot"],
                  multi_edge_samplers=config["multi_edge_samplers"],
                  std=config["std"], 
                  density_std=density_std, 
                  lambd=config["lambd"],
                  device=device,
                  precision=config["precision"])


epoch_folder = os.path.join(training_folder, "dnbp_models", epoch)
bpn.node_likelihoods.load_state_dict(torch.load(os.path.join(epoch_folder, "node_liks.pt"),map_location=device), strict=False)
bpn.likelihood_features.load_state_dict(torch.load(os.path.join(epoch_folder, "lik_feats.pt"),map_location=device), strict=False)
bpn.edge_densities.load_state_dict(torch.load(os.path.join(epoch_folder,"edge_dense.pt"),map_location=device), strict=False)
bpn.edge_samplers.load_state_dict(torch.load(os.path.join(epoch_folder,"edge_samps.pt"),map_location=device), strict=False)
bpn.time_samplers.load_state_dict(torch.load(os.path.join(epoch_folder,"time_samps.pt"),map_location=device), strict=False)




bin_size = 0.025
num_samples = 100000
root_dir = './data/pendulum/'
train_transforms = torchvision.transforms.Compose([articulated_transforms.ToTensor(),
															articulated_transforms.Resize(size=128)])

batch_size = 6
train_dataset = articulated_toy_dataset.ArticulatedToyDataset(root_dir, mode='Train', 
                                                              categories=["dynamic", "noise", "no_noise"],
                                                              window_size=20, data_max_length=20, 
                                                 transform=train_transforms)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0)




time_deltas_list=[]
for i in range(3):
    time_deltas_list.append(bpn.time_samplers[i](num_samples).view(num_samples,1,2).cpu().detach())

# (Bx(20-1)) x 3 x 2
time_deltas = torch.cat(time_deltas_list, dim=1)


fig, ax = plt.subplots(1, 3, figsize=(10,3))

for i in range(3):
    ax[i].set_xlim(-1,1)
    ax[i].set_ylim(-1,1)
    ax[i].scatter(x=time_deltas[:,i,0], y=time_deltas[:,i,1], s=1, c='r')
plt.savefig(os.path.join(training_folder, 'results', 'dnbp_time_samples.jpg'), dpi=fig.dpi)


# Convert samples into numpy coordinate frame
shifted_time_deltas = time_deltas.clone()
shifted_time_deltas[:,:,0] += 1
shifted_time_deltas[:,:,1] -= 1
shifted_time_deltas[:,:,1] *= -1
coord_list = (shifted_time_deltas/bin_size).numpy().astype(int)


fig, ax = plt.subplots(1, 3, figsize=(12,3))

for i in range(3):
    time_counts = np.zeros((int(2/bin_size), int(2/bin_size)))
    for i_coord in range(len(coord_list[:,i])):
        time_counts[coord_list[i_coord,i,1], coord_list[i_coord,i,0]] += 1
    time_counts /= time_counts.sum()
    time_counts /= time_counts.max()
    ax[i].imshow(time_counts, extent=[-1,1,-1,1])
plt.savefig(os.path.join(training_folder, 'results', 'dnbp_time_histograms.jpg'), dpi=fig.dpi)



bins = []
for i in range(3):
    time_counts = np.zeros((int(2/bin_size), int(2/bin_size)))
    for i_coord in range(len(coord_list[:,i])):
        time_counts[coord_list[i_coord,i,1], coord_list[i_coord,i,0]] += 1
    time_counts /= time_counts.sum()
    bins.append(time_counts)


for i in range(3):
  fig, ax = plt.subplots(figsize=(10,10))
  ax.imshow(bins[i], vmin=bins[i].min(), vmax=bins[i].max(), extent=[-1,1,-1,1])
  ax.set_xticks([-1,0,1])
  ax.set_yticks([-1,0,1])

  ax.set_yticklabels([-64,0,64])
  ax.set_xticklabels([-64,0,64])
  ax.tick_params(labelsize=20)
  if i==0:
    name='root'
  elif i==1:
    name='mid'
  else:
    name='end'
  plt.savefig(os.path.join(training_folder, 'results', 'dnbp_time_histogram_'+name+'_plot.jpg'), dpi=fig.dpi)
  v1 = np.linspace(bins[i].min(), bins[i].max(), 5, endpoint=True)
  norm = colors.Normalize(vmin=bins[i].min(), vmax=bins[i].max())
  # plt.gca().set_visible(False)
  # cax=plt.axes([0,0,0.1,2])
  
  cbar=plt.colorbar(cm.ScalarMappable(norm=norm), orientation="vertical", ax=ax, ticks=v1)
  cbar.ax.set_yticklabels(["{:4.3f}".format(i) for i in v1])
  ax.remove()
  cbar.ax.tick_params(labelsize=15)
  for t in cbar.ax.get_yticklabels():
    t.set_fontsize(30)
  plt.savefig(os.path.join(training_folder, 'results', 'dnbp_time_histogram_'+name+'_cbar.jpg'), dpi=fig.dpi, bbox_inches='tight')







# # conditioned on root, where is middle?
# p(src=1|dst=0) approx: src \sim dst - delta

# #conditioned on middle, where is end
# p(src=2|dst=1) approx: src \sim dst - delta


deltas_01 = -bpn.edge_samplers[0](num_samples).cpu().detach()
deltas_12 = -bpn.edge_samplers[1](num_samples).cpu().detach()
joint_deltas = torch.cat([deltas_01.unsqueeze(0), deltas_12.unsqueeze(0)])


fig, ax = plt.subplots(1, 2, figsize=(8,4))

for i in range(2):
    ax[i].set_xlim(-1,1)
    ax[i].set_ylim(-1,1)
    ax[i].scatter(x=joint_deltas[i,:,0], y=joint_deltas[i,:,1], s=5, c='r')
plt.savefig(os.path.join(training_folder, 'results', 'dnbp_pairwise_samples.jpg'), dpi=fig.dpi)



shifted_joint_deltas = joint_deltas.clone()
shifted_joint_deltas[:,:,0] += 1
shifted_joint_deltas[:,:,1] -= 1
shifted_joint_deltas[:,:,1] *= -1
coord_list = (shifted_joint_deltas/bin_size).numpy().astype(int)

fig, ax = plt.subplots(1, 2, figsize=(8,4))

for i in range(2):
    time_counts = np.zeros((int(2/bin_size), int(2/bin_size)))
    for i_coord in range(len(coord_list[i])):
        if coord_list[i,i_coord,1]>=time_counts.shape[0] or coord_list[i,i_coord,0]>=time_counts.shape[1]:
            continue
        time_counts[coord_list[i,i_coord,1], coord_list[i,i_coord,0]] += 1
    time_counts /= time_counts.sum()
    time_counts /= time_counts.max()
    ax[i].imshow(time_counts, extent=[-1,1,-1,1])
plt.savefig(os.path.join(training_folder, 'results', 'dnbp_pairwise_histograms.jpg'), dpi=fig.dpi)



bins=[]
for i in range(2):
    time_counts = np.zeros((int(2/bin_size), int(2/bin_size)))
    for i_coord in range(len(coord_list[i])):
        if coord_list[i,i_coord,1]>=time_counts.shape[0] or coord_list[i,i_coord,0]>=time_counts.shape[1]:
            continue
        time_counts[coord_list[i,i_coord,1], coord_list[i,i_coord,0]] += 1
    time_counts /= time_counts.sum()
    bins.append(time_counts)


for i in range(2):
  fig, ax = plt.subplots(figsize=(10,10))
  ax.imshow(bins[i], vmin=bins[i].min(), vmax=bins[i].max(), extent=[-1,1,-1,1])
  ax.set_xticks([-1,0,1])
  ax.set_yticks([-1,0,1])

  ax.set_yticklabels([-64,0,64])
  ax.set_xticklabels([-64,0,64])
  ax.tick_params(labelsize=20)
  if i==0:
    name='mid-root'
  else:
    name='end-mid'
  plt.savefig(os.path.join(training_folder, 'results', 'dnbp_pairwise_histogram_'+name+'_plot.jpg'), dpi=fig.dpi)

  v1 = np.linspace(bins[i].min(), bins[i].max(), 5, endpoint=True)
  norm = colors.Normalize(vmin=bins[i].min(), vmax=bins[i].max())
  # plt.gca().set_visible(False)
  # cax=plt.axes([0,0,0.1,2])
  cbar=plt.colorbar(cm.ScalarMappable(norm=norm), orientation="vertical", ax=ax, ticks=v1)
  cbar.ax.set_yticklabels(["{:4.3f}".format(i) for i in v1])
  ax.remove()
  cbar.ax.tick_params(labelsize=15)
  for t in cbar.ax.get_yticklabels():
    t.set_fontsize(30)
  plt.savefig(os.path.join(training_folder, 'results', 'dnbp_pairwise_histogram_'+name+'_cbar.jpg'), dpi=fig.dpi, bbox_inches='tight')








nump = 100
x = np.linspace(-1, 1, nump)
y = np.linspace(-1, 1, nump)
xv, yv = np.meshgrid(x, y)
grid = torch.from_numpy(np.concatenate([np.expand_dims(xv.flatten(), axis=1), np.expand_dims(yv.flatten(),axis=1)], axis=1))
grid = grid.view((nump*nump,1,2)).double().to(device=device)


fig, ax = plt.subplots(1, bpn.num_edges, figsize=(3*bpn.num_edges,3))
for i in range(bpn.num_edges):
    ax[i].set_ylim(-1,1)
    ax[i].set_xlim(-1,1)
    

    out = bpn.edge_densities[i](-grid.squeeze(1)).cpu()
    out = (F.interpolate(out.view(1,1,nump,nump).flip(2),(1000,1000)).squeeze()).detach().numpy()
    ax[i].imshow(out, extent=[-1,1,-1,1])
    print(out.min(), out.max())
plt.savefig(os.path.join(training_folder, 'results', 'dnbp_pairwise_densities.jpg'), dpi=fig.dpi)



for i in range(2):

  out = bpn.edge_densities[i](-grid.squeeze(1)).cpu()
  out = (F.interpolate(out.view(1,1,nump,nump).flip(2),(1000,1000)).squeeze()).detach().numpy()


  fig, ax = plt.subplots(figsize=(10,10))
  ax.imshow(out, vmin=out.min(), vmax=out.max(), extent=[-1,1,-1,1])
  ax.set_xticks([-1,0,1])
  ax.set_yticks([-1,0,1])

  ax.set_yticklabels([-64,0,64])
  ax.set_xticklabels([-64,0,64])
  ax.tick_params(labelsize=20)
  if i==0:
    name='mid-root'
  else:
    name='end-mid'
  plt.savefig(os.path.join(training_folder, 'results', 'dnbp_pairwise_density_'+name+'_plot.jpg'), dpi=fig.dpi)

  v1 = np.linspace(out.min(), out.max(), 5, endpoint=True)
  norm = colors.Normalize(vmin=out.min(), vmax=out.max())
  # plt.gca().set_visible(False)
  # cax=plt.axes([0,0,0.1,2])
  cbar=plt.colorbar(cm.ScalarMappable(norm=norm), orientation="vertical", ax=ax, ticks=v1)
  cbar.ax.set_yticklabels(["{:4.3f}".format(i) for i in v1])
  ax.remove()
  cbar.ax.tick_params(labelsize=15)
  for t in cbar.ax.get_yticklabels():
    t.set_fontsize(30)
  plt.savefig(os.path.join(training_folder, 'results', 'dnbp_pairwise_density_'+name+'_cbar.jpg'), dpi=fig.dpi, bbox_inches='tight')
