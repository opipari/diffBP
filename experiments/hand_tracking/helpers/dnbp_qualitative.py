import os
import sys
import time
import shutil
import json
import argparse

import numpy as np
from PIL import Image

from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import figure
from skimage import io
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler

import pickle

from diffBP.networks.dnbp_hand import factors, dnbp

from diffBP.datasets import fphab_dataset




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


def visualize_joints_belief(ax, beliefs, weights, joint_idx):
	# Scatter hand joints on image
	x = beliefs[:, 0]
	y = beliefs[:, 1]

	colors = ['black', 'r', 'm', 'b', 'c', 'g']+3*['r']+3*['m']+3*['b']+3*['c']+3*['g']
	colors = ['orange', 
				'#e50000', '#ec2d31', '#f25b62', '#f98893',
				'#c20078', '#d42d84', '#e75990', '#f9869c', 
				'#0343df', '#2160d0', '#3f7dc1', '#5d9ab2',
				'#00ffff', '#3aecf4', '#74d9e8', '#aec6dd',
				'#15b01a', '#48bf4d', '#7bcf80', '#aedeb3']
	ax.scatter(x, y, 1 if weights is None else [(1/3)*2**w for w in weights], colors[joint_idx])


def _draw2djoints(ax, annots, links, alpha=1):
	"""Draw segments, one color per link"""
	colors = ['r', 'm', 'b', 'c', 'g']
	colors = ['orange', 
				'#e50000', '#ec2d31', '#f25b62', '#f98893',
				'#c20078', '#d42d84', '#e75990', '#f9869c', 
				'#0343df', '#2160d0', '#3f7dc1', '#5d9ab2',
				'#00ffff', '#3aecf4', '#74d9e8', '#aec6dd',
				'#15b01a', '#48bf4d', '#7bcf80', '#aedeb3']

	for finger_idx, finger_links in enumerate(links):
		for idx in range(len(finger_links) - 1):
			_draw2dseg(
				ax,
				annots,
				finger_links[idx],
				finger_links[idx + 1],
				c=colors[finger_links[idx + 1]],
				alpha=alpha)

def _draw2dseg(ax, annot, idx1, idx2, c='r', alpha=1):
	"""Draw segment of given color"""
	ax.plot(
		[annot[idx1, 0], annot[idx2, 0]], [annot[idx1, 1], annot[idx2, 1]],
		c=c,
		alpha=alpha, linewidth=2.5)


reorder_idx = np.array([
	0, 1, 6, 7, 8, 2, 9, 10, 11, 3, 12, 13, 14, 4, 15, 16, 17, 5, 18, 19,
	20
])


def test_plot_density_part(skeleton, belief_particles, belief_weights, std, joint_idx, img, orig_shape, save=None, path=''):
	skeleton = skeleton[reorder_idx]
	skel_proj = skeleton.copy()[:,:2]#proj_points(skeleton)
	skel_proj[:,0]=(skel_proj[:,0]*(96))+48
	skel_proj[:,1]=(skel_proj[:,1]*(96))+48
	skel_proj[:,0] = np.clip(skel_proj[:,0],0,orig_shape[1]-1)
	skel_proj[:,1] = np.clip(skel_proj[:,1],0,orig_shape[0]-1)
	#bel_proj = proj_points(belief_particles.view(-1,3).numpy()*1000)
	bel_proj = belief_particles.view(-1,3).numpy().copy()[:,:2]#proj_points(skeleton)
	bel_proj[:,0]=(bel_proj[:,0]*(96))+48
	bel_proj[:,1]=(bel_proj[:,1]*(96))+48
	bel_proj[:,0] = np.clip(bel_proj[:,0],0,orig_shape[1]-1)
	bel_proj[:,1] = np.clip(bel_proj[:,1],0,orig_shape[0]-1)

	xv, yv = np.meshgrid(np.arange(orig_shape[1]), np.arange(orig_shape[0]))
	#xv = (xv-315.944855)/475.065948
	#yv = (yv-245.287079)/475.065857
	zv = np.ones_like(xv)
	pnts_grid = np.stack((xv,yv,zv),axis=-1).astype('float32')#*skeleton[joint_idx,2]*1000

	belief_particles = belief_particles.view(1, 1, -1, 3)[:,:,:,:2]
	weights = belief_weights.view(1, 1, -1)

	pnts = (torch.tensor(pnts_grid.reshape(1,-1,1,3)[:,:,:,:2])-48)/(96)
	diffsq = (((pnts-belief_particles)/std)**2).sum(dim=-1)
	exp_val = torch.exp((-1/2) * diffsq)
	fact = 1/(np.power(std, 2)*np.power(2*np.pi, 2/2))
	fact_ = fact * exp_val
	out = (weights * fact_).sum(dim=-1).reshape((orig_shape[0], orig_shape[1]))

	mean=-8.728770159651145
	std=45.024769450434384
	img = (img*std)+mean
	img = cv2.resize(img, (orig_shape[1], orig_shape[0]))

	
	fig = figure.Figure()
	ax = fig.subplots(1,2)
	ax[1].imshow(out, vmin=0)
	ax[1].axis('off')
	visualize_joints_2d(ax[1], skel_proj, joint_idxs=False)
	visualize_joints_belief(ax[1], bel_proj, None, joint_idx)

	ax[0].imshow(img)
	ax[0].axis('off')
	visualize_joints_2d(ax[0], skel_proj, joint_idxs=False)
	visualize_joints_belief(ax[0], bel_proj, None, joint_idx)


	if save is None:
		plt.show()
	else:
		fig.savefig(os.path.join(path,'bel_'+str(save)+'.jpg'))
	plt.cla()
	plt.clf()
	plt.close()




def draw_samples(skeleton, belief_particles, belief_weights, std, rgb, dm, img, orig_shape, samples, maxes, save=None, path=''):
	skeleton = skeleton[reorder_idx]
	skel_proj = skeleton.copy()[:,:2]#proj_points(skeleton)
	skel_proj[:,0]=(skel_proj[:,0]*(96))+48
	skel_proj[:,1]=(skel_proj[:,1]*(96))+48
	skel_proj[:,0] = np.clip(skel_proj[:,0],0,orig_shape[1]-1)
	skel_proj[:,1] = np.clip(skel_proj[:,1],0,orig_shape[0]-1)


		
	mean=-8.728770159651145
	std=45.024769450434384
	img = ((img*std)+mean)*1000+dm.item()
	img = cv2.resize(img, (orig_shape[1], orig_shape[0]))
	

	px = 1/plt.rcParams['figure.dpi']
	# fig = figure.Figure(dpi=800)
	# ax = fig.subplots(1,4)
	fig,ax = plt.subplots(dpi=75)
	ax.imshow(img, cmap='bone')
	ax.axis('off')
	for joint_idx in range(len(belief_particles)):
		#bel_proj = proj_points(belief_particles[joint_idx].detach().cpu().view(-1,3).numpy()*1000)
		bel_proj = belief_particles[joint_idx].detach().cpu().view(-1,3).numpy().copy()[:,:2]
		bel_proj[:,0]=(bel_proj[:,0]*(96))+48
		bel_proj[:,1]=(bel_proj[:,1]*(96))+48
		bel_proj[:,0] = np.clip(bel_proj[:,0],0,orig_shape[1]-1)
		bel_proj[:,1] = np.clip(bel_proj[:,1],0,orig_shape[0]-1)
		visualize_joints_belief(ax, bel_proj, belief_weights[joint_idx].detach().cpu().view(-1).numpy().copy(), joint_idx)
	fig.savefig(os.path.join(path,'bel_'+str(save)+'.png'), bbox_inches='tight', pad_inches=0)
	plt.cla()
	plt.clf()
	plt.close()


	fig,ax = plt.subplots(dpi=75)
	ax.imshow(img, cmap='bone')
	ax.axis('off')
	for pi in range(samples[0].shape[0]):
		samples_j_proj = torch.cat([smp[pi].unsqueeze(0) for smp in samples]).detach().cpu().view(-1,3).numpy().copy()[:,:2]#*1000
		samples_j_proj[:,0]=(samples_j_proj[:,0]*(96))+48
		samples_j_proj[:,1]=(samples_j_proj[:,1]*(96))+48
		samples_j_proj = samples_j_proj[reorder_idx]
		#samples_j_proj = proj_points(samples_j)
		samples_j_proj[:,0] = np.clip(samples_j_proj[:,0],0,orig_shape[1]-1)
		samples_j_proj[:,1] = np.clip(samples_j_proj[:,1],0,orig_shape[0]-1)
		visualize_joints_2d(ax, samples_j_proj, joint_idxs=False)
	fig.savefig(os.path.join(path,'joint_'+str(save)+'.png'), bbox_inches='tight', pad_inches=0)
	plt.cla()
	plt.clf()
	plt.close()
		
	fig,ax = plt.subplots(dpi=75)
	ax.imshow(img, cmap='bone')
	ax.axis('off')
	samples_j_proj = torch.cat([smp for smp in maxes[0]]).detach().cpu().view(-1,3).numpy().copy()[:,:2]#*1000
	samples_j_proj[:,0]=(samples_j_proj[:,0]*(96))+48
	samples_j_proj[:,1]=(samples_j_proj[:,1]*(96))+48
	samples_j_proj = samples_j_proj[reorder_idx]
	#samples_j_proj = proj_points(samples_j)
	samples_j_proj[:,0] = np.clip(samples_j_proj[:,0],0,orig_shape[1]-1)
	samples_j_proj[:,1] = np.clip(samples_j_proj[:,1],0,orig_shape[0]-1)
	visualize_joints_2d(ax, samples_j_proj, joint_idxs=False)
	fig.savefig(os.path.join(path,'mle_'+str(save)+'.png'), bbox_inches='tight', pad_inches=0)
	plt.cla()
	plt.clf()
	plt.close()

	fig,ax = plt.subplots(dpi=75)
	ax.imshow(img, cmap='bone')
	ax.axis('off')
	visualize_joints_2d(ax, skel_proj, joint_idxs=False)
	fig.savefig(os.path.join(path,'gt_'+str(save)+'.png'), bbox_inches='tight', pad_inches=0)
	plt.cla()
	plt.clf()
	plt.close()
	

	fig,ax = plt.subplots(dpi=75)
	ax.imshow(rgb.astype(np.uint8))
	ax.axis('off')
	fig.savefig(os.path.join(path,'clr_'+str(save)+'.png'), bbox_inches='tight', pad_inches=0)
	plt.cla()
	plt.clf()
	plt.close()

	fig,ax = plt.subplots(dpi=75)
	ax.imshow(img, cmap='bone')
	ax.axis('off')
	fig.savefig(os.path.join(path,'obs_'+str(save)+'.png'), bbox_inches='tight', pad_inches=0)
	plt.cla()
	plt.clf()
	plt.close()

	# if save is None:
	# 	plt.show()
	# else:
	# 	fig.savefig(os.path.join(path,'bel_'+str(save)+'.png'))
	# plt.cla()
	# plt.clf()
	# plt.close()



def test_dnbp(config, epoch):
	torch.autograd.set_detect_anomaly(True)
	#
	# Setup train and test datasets
	#
	data_dir = config["data_dir"]
	train_batch_size = config["train_batch_size"]
	test_batch_size = config["test_batch_size"]
	device = torch.device(config["device"])


	if config["use_augmentation"]:
		train_transforms = torchvision.transforms.Compose([fphab_dataset.ToTensor(),
															fphab_dataset.Normalize()])
	else:
		train_transforms = torchvision.transforms.Compose([fphab_dataset.ToTensor(),
													fphab_dataset.Normalize()])

	test_transforms = torchvision.transforms.Compose([fphab_dataset.ToTensor(),
													fphab_dataset.Normalize()])

	print("Training Transformations:")
	print(train_transforms)
	print()
	print()


	# train_dataset = fphab_dataset.FPHABDataset(data_dir, mode='train', 
	# 													window_size=20,
	# 													transform=train_transforms)
	test_dataset = fphab_dataset.FPHABDataset(data_dir, mode='test', 
														transform=train_transforms)
	# val_dataset = articulated_toy_dataset.ArticulatedToyDataset(data_dir, mode='Test',  num_seqs=None,
	# 													categories=config['data_categories'],
	# 													window_size=20, data_max_length=20, 
	#                                                  transform=test_transforms)

	# train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size,
	# 						shuffle=True, num_workers=0, drop_last=True)
	test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size,
							shuffle=True, num_workers=0)
	# val_dataloader = DataLoader(val_dataset, batch_size=test_batch_size,
	#                         shuffle=True, num_workers=0)

	print()
	print("Length of testing data:", len(test_dataset))
	print()
	#
	# Setup model to be trained
	#
	graph = torch.tensor(config["graph"])
	edge_set = torch.tensor(config["edge_set"])
	



	if config["precision"]=="float32":
		data_type = torch.float32
	else:
		data_type = torch.double

	bpn = dnbp.DNBP(graph, edge_set, config["inc_nghbrs"], 
					  particle_count=config["train_particle_count"],
					  particle_size=config["particle_size"],
					  enc_output_feats_tot=config["enc_output_feats_tot"],
					  std=config["std"], 
					  lambd=config["lambd"],
					  device=device,
					  precision=config["precision"],
					  est_bounds=0.5,
					  est_bounds_z=0.15,
					  use_time=config["use_time"])



	# bpn.frac_resamp = config["initial_resample_rate"]
	# bpn.use_time = True



	#
	# Initialize optimizers for different factors
	
	out_folder = './experiments/hand_tracking/results/'
	epoch_folder = os.path.join(config["model_folder"], "epoch_"+str(epoch))
	bpn.node_likelihoods.load_state_dict(torch.load(os.path.join(epoch_folder,"node_liks.pt"),map_location=device), strict=False)
	bpn.likelihood_features.load_state_dict(torch.load(os.path.join(epoch_folder,"lik_feats.pt"),map_location=device), strict=False)
	bpn.edge_densities.load_state_dict(torch.load(os.path.join(epoch_folder,"edge_dense.pt"),map_location=device), strict=False)
	bpn.edge_samplers.load_state_dict(torch.load(os.path.join(epoch_folder,"edge_samps.pt"),map_location=device), strict=False)
	bpn.time_samplers.load_state_dict(torch.load(os.path.join(epoch_folder,"time_samps.pt"),map_location=device), strict=False)



	bpn = bpn.eval()
	with torch.no_grad():

		for v_batch, sample_batched_ in enumerate(test_dataloader):
			if v_batch>9:
				break
			print(v_batch,len(test_dataloader))
			sample_batched = sample_batched_
			# Window: B x S x C x H x W
			# Labels: B x S x 3 x 2
			batch_images = sample_batched['imgs']
			batch_depths = sample_batched['depth'].type(bpn.type)
			batch_labels = sample_batched['label'].type(bpn.type)
			batch_paths = sample_batched['path']
			batch_orig_shapes = sample_batched['orig_shapes']
			batch_crop_top_lefts = sample_batched['crop_top_lefts']
			batch_crop_sizes = sample_batched['crop_sizes']
			batch_depth_means = sample_batched['depth_mean']

			print(batch_depths.shape[1])
			bpn.particle_count = 200#config["test_particle_count"]
			bpn.reinit_particles(batch_depths.shape[0], batch_depths[:,0])
			os.makedirs(os.path.join(out_folder,'test',str(v_batch)), exist_ok=True)

			skeleton_path = os.path.join('./data','hand_tracking',batch_paths[0], 'pose', 'skeleton.txt')
			skeleton_vals = np.loadtxt(skeleton_path)
			skel_order = skeleton_vals[:, 0]
			skel = skeleton_vals[:, 1:].reshape(skeleton_vals.shape[0], 21, -1)
			skel = skel.astype(np.float32)

			errors = []
			entropies = []
			stats = []
			# Iterate over sequence dimension
			for i_seq in range(batch_depths.shape[1]):
				tr = batch_labels[:,i_seq]
				im = batch_images[:,i_seq]
				dp = batch_depths[:,i_seq].to(device=device)
				tl = batch_crop_top_lefts[0,i_seq]
				sz = batch_crop_sizes[0,i_seq]
				dm = batch_depth_means[0,i_seq]

				label_xyz = np.ones((21, 3), dtype = 'float32') 
				# Undo ground truth coordinate normalization
				label_xyz[:,0] = ((tr[0,:,0]*96+48)*sz[0]/96 + tl[0])
				label_xyz[:,1] = ((tr[0,:,1]*96+48)*sz[1]/96 + tl[1])
				label_xyz[:,2] = (tr[0,:,2]*1000)+dm

				# img_path = os.path.join('./data',batch_paths[0], 'color', 'color_{:04d}.jpeg'.format(i_seq))
				# img = cv2.imread(img_path)
				depth_path = os.path.join('./data','hand_tracking',batch_paths[0], 'depth', 'depth_{:04d}.png'.format(i_seq))
				depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)


				
				if i_seq==0:
					max_parts, pred_weights = bpn.max_marginals()
					max_parts = torch.cat(max_parts).detach().cpu()

					max_parts[:,0] = ((max_parts[:,0]*96+48)*sz[0]/96 + tl[0])
					max_parts[:,1] = ((max_parts[:,1]*96+48)*sz[1]/96 + tl[1])
					max_parts[:,2] = (max_parts[:,2]*1000)+dm

					max_parts[:,0] = max_parts[:,2]*(max_parts[:,0]-(315.944855))/475.065948
					max_parts[:,1] = max_parts[:,2]*(max_parts[:,1]-(245.287079))/475.065857

					euc = np.sqrt(np.sum((skel[i_seq]-max_parts.numpy())**2,axis=1))
					errors.append(euc)

					samp_particles = 5
					particle_samples = bpn.recursive_ancestral_sampling(bpn.belief_particles, bpn.belief_weights, 
																		 ith=0, parent=None, parent_idx=None, 
																		 visited=[], visited_samples=[], num_samples=samp_particles)
					max_parts, pred_weights = bpn.max_marginals()

					
					draw_samples(tr[0].cpu().numpy(), 
										bpn.belief_particles, 
										bpn.belief_weights, config["std"], 
										batch_images[0,i_seq].cpu().detach().numpy(), 
										dm,
										batch_depths[0,i_seq].cpu().detach().permute(1,2,0).numpy(),
										batch_orig_shapes[0,i_seq].cpu().numpy().astype(int), 
										[ls.view(samp_particles,3) for ls in particle_samples],
										(max_parts, pred_weights),
										save=i_seq,
										path=os.path.join(out_folder,'test',str(v_batch)))

					bin_size = 5
					entropy_bound_min = [-301.1026*1.125, -218.8729*1.125, 15.1193*1.125]
					entropy_bound_max = [338.011*1.125, 365.4826*1.125, 721.17*1.125]
					marg_entropies = []
					marg_stats = []
					for node_i in range(bpn.num_nodes):
						belief_samples = bpn.belief_particles[node_i].view(bpn.batch_size, -1, bpn.particle_size).clone().detach()

						marg_stats.append({'cov': np.cov((belief_samples[:,:,0]*96+48).cpu().numpy(), (belief_samples[:,:,1]*96+48).cpu().numpy()),
											'mean_x': np.mean((belief_samples[:,:,0]*96+48).cpu().numpy()),
											'mean_y': np.mean((belief_samples[:,:,1]*96+48).cpu().numpy())})

						belief_samples[:,:,0] = ((belief_samples[:,:,0]*96+48)*sz[0]/96 + tl[0])
						belief_samples[:,:,1] = ((belief_samples[:,:,1]*96+48)*sz[1]/96 + tl[1])
						belief_samples[:,:,2] = (belief_samples[:,:,2]*1000)+dm

						belief_samples[:,:,0] = belief_samples[:,:,2]*(belief_samples[:,:,0]-(315.944855))/475.065948
						belief_samples[:,:,1] = belief_samples[:,:,2]*(belief_samples[:,:,1]-(245.287079))/475.065857

						belief_samples = bpn.discrete_samples(belief_samples, 
																bpn.belief_weights[node_i].view(bpn.batch_size, -1), 
																5000)
						belief_samples = belief_samples.view(bpn.batch_size, -1, bpn.particle_size)
						shifted_samples = belief_samples.clone().detach()
						shifted_samples[:,:,0] -= entropy_bound_min[0]
						shifted_samples[:,:,1] -= entropy_bound_min[1]
						shifted_samples[:,:,2] -= entropy_bound_min[2]
						coord_list = (shifted_samples/bin_size).cpu().numpy().astype(int)

						x_box_size = int((entropy_bound_max[0]-entropy_bound_min[0])/bin_size)
						y_box_size = int((entropy_bound_max[1]-entropy_bound_min[1])/bin_size)
						z_box_size = int((entropy_bound_max[2]-entropy_bound_min[2])/bin_size)
						batch_ents = []

						time_counts = np.zeros((x_box_size, y_box_size, z_box_size))
						for i_coord in range(coord_list.shape[1]):
							if (0<coord_list[0,i_coord,2] and coord_list[0,i_coord,2]<z_box_size) \
								and (0<coord_list[0,i_coord,1] and coord_list[0,i_coord,1]<y_box_size) \
								and (0<coord_list[0,i_coord,0] and coord_list[0,i_coord,0]<x_box_size):
								time_counts[coord_list[0,i_coord,0], coord_list[0,i_coord,1], coord_list[0,i_coord,2]] += 1

						time_counts /= time_counts.sum()

						probs = time_counts.flatten()
						probs[probs==0]=1
						marg_entropy = -np.sum(probs*np.log(probs))
						marg_entropies.append(marg_entropy)
					entropies.append(marg_entropies)
					stats.append(marg_stats)
				
				bpn.frac_resamp = 1

				bpn.compute_feats(dp)

				for msg_it in range(2):
					for node_i in range(bpn.num_nodes):
						# Run single belief propagation message+belief update
						bpn.update(node_id=node_i)

				max_parts, pred_weights = bpn.max_marginals()
				max_parts = torch.cat(max_parts).detach().cpu()

				max_parts[:,0] = ((max_parts[:,0]*96+48)*sz[0]/96 + tl[0])
				max_parts[:,1] = ((max_parts[:,1]*96+48)*sz[1]/96 + tl[1])
				max_parts[:,2] = (max_parts[:,2]*1000)+dm

				max_parts[:,0] = max_parts[:,2]*(max_parts[:,0]-(315.944855))/475.065948
				max_parts[:,1] = max_parts[:,2]*(max_parts[:,1]-(245.287079))/475.065857

				euc = np.sqrt(np.sum((skel[i_seq]-max_parts.numpy())**2,axis=1))
				errors.append(euc)

				samp_particles = 5
				particle_samples = bpn.recursive_ancestral_sampling(bpn.belief_particles, bpn.belief_weights, 
																	 ith=0, parent=None, parent_idx=None, 
																	 visited=[], visited_samples=[], num_samples=samp_particles)
				max_parts, pred_weights = bpn.max_marginals()

				
				draw_samples(tr[0].cpu().numpy(), 
									bpn.belief_particles, 
									bpn.belief_weights, config["std"], 
									batch_images[0,i_seq].cpu().detach().numpy(), 
									dm,
									batch_depths[0,i_seq].cpu().detach().permute(1,2,0).numpy(),
									batch_orig_shapes[0,i_seq].cpu().numpy().astype(int), 
									[ls.view(samp_particles,3) for ls in particle_samples],
									(max_parts, pred_weights),
									save=i_seq+1,
									path=os.path.join(out_folder,'test',str(v_batch)))


				bin_size = 5
				entropy_bound_min = [-301.1026*1.125, -218.8729*1.125, 15.1193*1.125]
				entropy_bound_max = [338.011*1.125, 365.4826*1.125, 721.17*1.125]
				marg_entropies = []
				marg_stats = []
				for node_i in range(bpn.num_nodes):
					belief_samples = bpn.belief_particles[node_i].view(bpn.batch_size, -1, bpn.particle_size).clone().detach()

					marg_stats.append({'cov': np.cov((belief_samples[:,:,0]*96+48).cpu().numpy(), (belief_samples[:,:,1]*96+48).cpu().numpy()),
										'mean_x': np.mean((belief_samples[:,:,0]*96+48).cpu().numpy()),
										'mean_y': np.mean((belief_samples[:,:,1]*96+48).cpu().numpy())})

					belief_samples[:,:,0] = ((belief_samples[:,:,0]*96+48)*sz[0]/96 + tl[0])
					belief_samples[:,:,1] = ((belief_samples[:,:,1]*96+48)*sz[1]/96 + tl[1])
					belief_samples[:,:,2] = (belief_samples[:,:,2]*1000)+dm

					belief_samples[:,:,0] = belief_samples[:,:,2]*(belief_samples[:,:,0]-(315.944855))/475.065948
					belief_samples[:,:,1] = belief_samples[:,:,2]*(belief_samples[:,:,1]-(245.287079))/475.065857

					belief_samples = bpn.discrete_samples(belief_samples, 
															bpn.belief_weights[node_i].view(bpn.batch_size, -1), 
															5000)
					belief_samples = belief_samples.view(bpn.batch_size, -1, bpn.particle_size)
					shifted_samples = belief_samples.clone().detach()
					shifted_samples[:,:,0] -= entropy_bound_min[0]
					shifted_samples[:,:,1] -= entropy_bound_min[1]
					shifted_samples[:,:,2] -= entropy_bound_min[2]
					coord_list = (shifted_samples/bin_size).cpu().numpy().astype(int)

					x_box_size = int((entropy_bound_max[0]-entropy_bound_min[0])/bin_size)
					y_box_size = int((entropy_bound_max[1]-entropy_bound_min[1])/bin_size)
					z_box_size = int((entropy_bound_max[2]-entropy_bound_min[2])/bin_size)
					batch_ents = []

					time_counts = np.zeros((x_box_size, y_box_size, z_box_size))
					for i_coord in range(coord_list.shape[1]):
						if (0<coord_list[0,i_coord,2] and coord_list[0,i_coord,2]<z_box_size) \
							and (0<coord_list[0,i_coord,1] and coord_list[0,i_coord,1]<y_box_size) \
							and (0<coord_list[0,i_coord,0] and coord_list[0,i_coord,0]<x_box_size):
							time_counts[coord_list[0,i_coord,0], coord_list[0,i_coord,1], coord_list[0,i_coord,2]] += 1
					time_counts /= time_counts.sum()

					probs = time_counts.flatten()
					probs[probs==0]=1
					marg_entropy = -np.sum(probs*np.log(probs))
					marg_entropies.append(marg_entropy)

				entropies.append(marg_entropies)
				stats.append(marg_stats)
				bpn.update_time()

			pickle.dump({"errors":np.array(errors),"entropies":np.array(entropies),
						"stats": stats}, open(os.path.join(out_folder,'test',str(v_batch),'meta.pkl'), "wb" ) )





if __name__=='__main__':
	config_file = sys.argv[1]
	epoch = sys.argv[2]
	config = json.load(open(config_file, 'r'))
	
	print("Training DNBP Configuration")
	print("config_file:", config_file)
	for k,v in config.items():
		print(k, v)
	print()
	print()


	# Must pass config file as command argument
	test_dnbp(config=config, epoch=epoch)



