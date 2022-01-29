import os
import matplotlib.pyplot as plt


from matplotlib import colors
from matplotlib import cm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision

from diffBP.datasets import articulated_toy_dataset, articulated_transforms

out_folder = "./experiments/spider_tracking/results"

bin_size = 0.025
root_dir = './data/spider/'


train_transforms = torchvision.transforms.Compose([articulated_transforms.ToTensor(),
															articulated_transforms.Resize(size=128)])

batch_size = 6
train_dataset = articulated_toy_dataset.ArticulatedToyDataset(root_dir, mode='Train', 
															  categories=["dynamic_dynamic_noise", "dynamic_static_noise", "dynamic_dynamic_extreme_noise", "dynamic_static_extreme_noise", "dynamic_no_noise"],
															  window_size=20, data_max_length=20, 
												 transform=train_transforms)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
						shuffle=False, num_workers=0)






time_deltas_list = []
for i_batch, sample_batched in enumerate(train_dataloader):
				
	# Window: B x S x C x H x W
	# Labels: B x S x 3 x 2
	batch_images = sample_batched['window'].type(torch.double)
	batch_labels = sample_batched['labels'].type(torch.double)[:,:,:,:2]
	
	label_deltas = torch.roll(batch_labels, -1, 1) - batch_labels
	label_deltas = label_deltas[:,:-1].contiguous().view(-1,7,2)

	time_deltas_list.append(label_deltas)

# (Bx(20-1)) x 3 x 2
time_deltas = torch.cat(time_deltas_list, dim=0)


# fig, ax = plt.subplots(1, 3, figsize=(10,3))

# for i in range(3):
# 	ax[i].set_xlim(-1,1)
# 	ax[i].set_ylim(-1,1)
# 	ax[i].scatter(x=time_deltas[:,i,0], y=time_deltas[:,i,1], s=5, c='r')
# plt.savefig(os.path.join(out_path, 'training_time_samples.jpg'), dpi=fig.dpi)


# Convert samples into numpy coordinate frame
shifted_time_deltas = time_deltas.clone()
shifted_time_deltas[:,:,0] += 1
shifted_time_deltas[:,:,1] -= 1
shifted_time_deltas[:,:,1] *= -1
coord_list = (shifted_time_deltas/bin_size).numpy().astype(int)


# fig, ax = plt.subplots(1, 3, figsize=(12,3))

# for i in range(3):
# 	time_counts = np.zeros((int(2/bin_size), int(2/bin_size)))
# 	for i_coord in range(len(coord_list[:,i])):
# 		time_counts[coord_list[i_coord,i,1], coord_list[i_coord,i,0]] += 1
# 	time_counts /= time_counts.sum()
# 	time_counts /= time_counts.max()
# 	ax[i].imshow(time_counts, extent=[-1,1,-1,1])
# plt.savefig(os.path.join(out_path, 'training_time_histograms.jpg'), dpi=fig.dpi)



bins = []
for i in range(7):
	time_counts = np.zeros((int(2/bin_size), int(2/bin_size)))
	for i_coord in range(len(coord_list[:,i])):
		time_counts[coord_list[i_coord,i,1], coord_list[i_coord,i,0]] += 1
	time_counts /= time_counts.sum()
	bins.append(time_counts)


for i in range(7):
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
		name='innergreen'
	elif i==2:
		name='innerred'
	elif i==3:
		name='innerblue'
	elif i==4:
		name='outergreen'
	elif i==5:
		name='outerred'
	else:
		name='outerblue'
	plt.savefig(os.path.join(out_folder, 'training_time_histogram_'+name+'_plot.jpg'), dpi=fig.dpi)
	v1 = np.linspace(bins[i].min(), bins[i].max(), 5, endpoint=True)
	norm = colors.Normalize(vmin=bins[i].min(), vmax=bins[i].max())
	plt.gca().set_visible(False)
	# cax=plt.axes([0,0,0.1,2])
	cbar=plt.colorbar(cm.ScalarMappable(norm=norm), orientation="vertical", ax=ax, ticks=v1)
	cbar.ax.set_yticklabels(["{:4.3f}".format(i) for i in v1])
	ax.remove()
	cbar.ax.tick_params(labelsize=15)
	for t in cbar.ax.get_yticklabels():
		t.set_fontsize(30)
	plt.savefig(os.path.join(out_folder, 'training_time_histogram_'+name+'_cbar.jpg'), dpi=fig.dpi, bbox_inches='tight')










deltas_01_list = []
deltas_02_list = []
deltas_03_list = []
deltas_14_list = []
deltas_25_list = []
deltas_36_list = []
for i_batch, sample_batched in enumerate(train_dataloader):
				
	# Window: B x S x C x H x W
	# Labels: B x S x 3 x 2
	batch_images = sample_batched['window'].type(torch.double)
	batch_labels = sample_batched['labels'].type(torch.double)[:,:,:,:2]
	
	deltas_01 = (batch_labels[:,:,1] - batch_labels[:,:,0]).view(-1,2)
	deltas_02 = (batch_labels[:,:,2] - batch_labels[:,:,0]).view(-1,2)
	deltas_03 = (batch_labels[:,:,3] - batch_labels[:,:,0]).view(-1,2)
	deltas_14 = (batch_labels[:,:,4] - batch_labels[:,:,1]).view(-1,2)
	deltas_25 = (batch_labels[:,:,5] - batch_labels[:,:,2]).view(-1,2)
	deltas_36 = (batch_labels[:,:,6] - batch_labels[:,:,3]).view(-1,2)


	deltas_01_list.append(deltas_01)
	deltas_02_list.append(deltas_02)
	deltas_03_list.append(deltas_03)
	deltas_14_list.append(deltas_14)
	deltas_25_list.append(deltas_25)
	deltas_36_list.append(deltas_36)


deltas_01 = torch.cat(deltas_01_list, dim=0)
deltas_02 = torch.cat(deltas_02_list, dim=0)
deltas_03 = torch.cat(deltas_03_list, dim=0)
deltas_14 = torch.cat(deltas_14_list, dim=0)
deltas_25 = torch.cat(deltas_25_list, dim=0)
deltas_36 = torch.cat(deltas_36_list, dim=0)
joint_deltas = torch.cat([deltas_01.unsqueeze(0), deltas_02.unsqueeze(0), deltas_03.unsqueeze(0), deltas_14.unsqueeze(0), deltas_25.unsqueeze(0), deltas_36.unsqueeze(0)])


# fig, ax = plt.subplots(1, 2, figsize=(8,4))

# for i in range(6):
# 	ax[i].set_xlim(-1,1)
# 	ax[i].set_ylim(-1,1)
# 	ax[i].scatter(x=joint_deltas[i,:,0], y=joint_deltas[i,:,1], s=5, c='r')
# plt.savefig(os.path.join(out_path, 'training_pairwise_samples.jpg'), dpi=fig.dpi)



shifted_joint_deltas = joint_deltas.clone()
shifted_joint_deltas[:,:,0] += 1
shifted_joint_deltas[:,:,1] -= 1
shifted_joint_deltas[:,:,1] *= -1
coord_list = (shifted_joint_deltas/bin_size).numpy().astype(int)

# fig, ax = plt.subplots(1, 2, figsize=(8,4))

# for i in range(2):
# 	time_counts = np.zeros((int(2/bin_size), int(2/bin_size)))
# 	for i_coord in range(len(coord_list[i])):
# 		time_counts[coord_list[i,i_coord,1], coord_list[i,i_coord,0]] += 1
# 	time_counts /= time_counts.sum()
# 	time_counts /= time_counts.max()
# 	ax[i].imshow(time_counts, extent=[-1,1,-1,1])
# plt.savefig(os.path.join(out_path, 'training_pairwise_histograms.jpg'), dpi=fig.dpi)



bins=[]
for i in range(6):
	time_counts = np.zeros((int(2/bin_size), int(2/bin_size)))
	for i_coord in range(len(coord_list[i])):
		time_counts[coord_list[i,i_coord,1], coord_list[i,i_coord,0]] += 1
	time_counts /= time_counts.sum()
	bins.append(time_counts)


for i in range(6):
	fig, ax = plt.subplots(figsize=(10,10))
	ax.imshow(bins[i], vmin=bins[i].min(), vmax=bins[i].max(), extent=[-1,1,-1,1])
	ax.set_xticks([-1,0,1])
	ax.set_yticks([-1,0,1])

	ax.set_yticklabels([-64,0,64])
	ax.set_xticklabels([-64,0,64])
	ax.tick_params(labelsize=20)
	if i==0:
		name='ig-root'
	elif i==1:
		name='ir-root'
	elif i==2:
		name='ib-root'
	elif i==3:
		name='og-ig'
	elif i==4:
		name='or-ir'
	elif i==5:
		name='ob-ib'
	plt.savefig(os.path.join(out_folder, 'training_pairwise_histogram_'+name+'_plot.jpg'), dpi=fig.dpi)

	v1 = np.linspace(bins[i].min(), bins[i].max(), 5, endpoint=True)
	norm = colors.Normalize(vmin=bins[i].min(), vmax=bins[i].max())
	plt.gca().set_visible(False)
	# cax=plt.axes([0,0,0.1,2])
	cbar=plt.colorbar(cm.ScalarMappable(norm=norm), orientation="vertical", ax=ax, ticks=v1)
	cbar.ax.set_yticklabels(["{:4.3f}".format(i) for i in v1])
	ax.remove()
	cbar.ax.tick_params(labelsize=15)
	for t in cbar.ax.get_yticklabels():
		t.set_fontsize(30)
	plt.savefig(os.path.join(out_folder, 'training_pairwise_histogram_'+name+'_cbar.jpg'), dpi=fig.dpi, bbox_inches='tight')