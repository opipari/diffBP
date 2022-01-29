import os
import gc
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
	ax.scatter(x, y, 1 if weights is None else [(1/3)*2**w for w in weights], colors[joint_idx])


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




def draw_samples(skeleton, belief_particles, belief_weights, std, img, orig_shape, samples, maxes, save=None, path=''):
	skeleton = skeleton[reorder_idx]
	skel_proj = skeleton.copy()[:,:2]#proj_points(skeleton)
	skel_proj[:,0]=(skel_proj[:,0]*(96))+48
	skel_proj[:,1]=(skel_proj[:,1]*(96))+48
	skel_proj[:,0] = np.clip(skel_proj[:,0],0,orig_shape[1]-1)
	skel_proj[:,1] = np.clip(skel_proj[:,1],0,orig_shape[0]-1)



        
	mean=-8.728770159651145
	std=45.024769450434384
	img = (img*std)+mean
	img = cv2.resize(img, (orig_shape[1], orig_shape[0]))
	

	px = 1/plt.rcParams['figure.dpi']
	fig = figure.Figure(dpi=800)
	ax = fig.subplots(1,4)
	ax[3].imshow(img, cmap='bone')
	ax[3].axis('off')
	for joint_idx in range(len(belief_particles)):
		#bel_proj = proj_points(belief_particles[joint_idx].detach().cpu().view(-1,3).numpy()*1000)
		bel_proj = belief_particles[joint_idx].detach().cpu().view(-1,3).numpy().copy()[:,:2]
		bel_proj[:,0]=(bel_proj[:,0]*(96))+48
		bel_proj[:,1]=(bel_proj[:,1]*(96))+48
		bel_proj[:,0] = np.clip(bel_proj[:,0],0,orig_shape[1]-1)
		bel_proj[:,1] = np.clip(bel_proj[:,1],0,orig_shape[0]-1)
		visualize_joints_belief(ax[3], bel_proj, belief_weights[joint_idx].detach().cpu().view(-1).numpy().copy(), joint_idx)
		

	ax[1].imshow(img, cmap='bone')
	ax[1].axis('off')
	for pi in range(samples[0].shape[0]):
		samples_j_proj = torch.cat([smp[pi].unsqueeze(0) for smp in samples]).detach().cpu().view(-1,3).numpy().copy()[:,:2]#*1000
		samples_j_proj[:,0]=(samples_j_proj[:,0]*(96))+48
		samples_j_proj[:,1]=(samples_j_proj[:,1]*(96))+48
		samples_j_proj = samples_j_proj[reorder_idx]
		#samples_j_proj = proj_points(samples_j)
		samples_j_proj[:,0] = np.clip(samples_j_proj[:,0],0,orig_shape[1]-1)
		samples_j_proj[:,1] = np.clip(samples_j_proj[:,1],0,orig_shape[0]-1)
		visualize_joints_2d(ax[1], samples_j_proj, joint_idxs=False)
		
		
	ax[2].imshow(img, cmap='bone')
	ax[2].axis('off')
	samples_j_proj = torch.cat([smp for smp in maxes[0]]).detach().cpu().view(-1,3).numpy().copy()[:,:2]#*1000
	samples_j_proj[:,0]=(samples_j_proj[:,0]*(96))+48
	samples_j_proj[:,1]=(samples_j_proj[:,1]*(96))+48
	samples_j_proj = samples_j_proj[reorder_idx]
	#samples_j_proj = proj_points(samples_j)
	samples_j_proj[:,0] = np.clip(samples_j_proj[:,0],0,orig_shape[1]-1)
	samples_j_proj[:,1] = np.clip(samples_j_proj[:,1],0,orig_shape[0]-1)
	visualize_joints_2d(ax[2], samples_j_proj, joint_idxs=False)

	ax[0].imshow(img, cmap='bone')
	ax[0].axis('off')
	visualize_joints_2d(ax[0], skel_proj, joint_idxs=False)


	if save is None:
		plt.show()
	else:
		fig.savefig(os.path.join(path,'bel_'+str(save)+'.png'))
	plt.cla()
	plt.clf()
	plt.close()






def run_test(config, bpn, val_dataloader, i_epoch, val_loss, val_steps, batches_seen, device):
	bpn = bpn.eval()
	with torch.no_grad():
		tracked_val_losses = [[] for _ in range(bpn.num_nodes)]
		for v_batch, sample_batched_ in enumerate(val_dataloader):
			if v_batch>0:
				break
			sample_batched = sample_batched_
			# Window: B x S x C x H x W
			# Labels: B x S x 3 x 2
			batch_depths = sample_batched['depth'].type(bpn.type)
			batch_labels = sample_batched['label'].type(bpn.type)
			batch_paths = sample_batched['path']
			batch_orig_shapes = sample_batched['orig_shapes']

			bpn.particle_count = config["test_particle_count"]
			bpn.reinit_particles(batch_depths.shape[0], batch_depths[:,0])


			# Iterate over sequence dimension
			for i_seq in range(batch_depths.shape[1]):
				if i_seq>=50:
					break
				tr = batch_labels[:,i_seq].to(device=device)
				dp = batch_depths[:,i_seq].to(device=device)
				# x = batch_images[:,i_seq].to(device=device)

				bpn.frac_resamp = 1

				bpn.compute_feats(dp)

				for node_i in range(bpn.num_nodes):
					# Run single belief propagation message+belief update
					bpn.update(node_id=node_i)

					loss = 0
					for mode in config["training_modes"]:
						# Calculate output density at ground truth
						loss -= torch.log(1e-40+bpn.density_estimation(node_i, tr[:,node_i].unsqueeze(1).unsqueeze(1), mode=mode))
					loss = loss.mean()/len(config["training_modes"])
					tracked_val_losses[node_i].append(loss.item())

				bpn.update_time()

				samp_particles = 5
				particle_samples = bpn.recursive_ancestral_sampling(bpn.belief_particles, bpn.belief_weights, 
																	 ith=0, parent=None, parent_idx=None, 
																	 visited=[], visited_samples=[], num_samples=samp_particles)
				max_parts, pred_weights = bpn.max_marginals()

				draw_samples(tr[0].cpu().numpy(), 
									bpn.belief_particles, 
									bpn.belief_weights, config["std"], 
									batch_depths[0,i_seq].cpu().detach().permute(1,2,0).numpy(), 
									batch_orig_shapes[0,i_seq].cpu().numpy().astype(int), 
									[ls.view(samp_particles,3) for ls in particle_samples],
									(max_parts, pred_weights),
									save=i_seq,
									path=i_epoch)

				test_plot_density_part(tr[0].cpu().numpy(), 
					bpn.belief_particles[0][0].cpu().detach(), bpn.belief_weights[0][0].cpu().detach(), 
					config["std"], 0, batch_depths[0,i_seq].cpu().detach().permute(1,2,0).numpy(), 
					batch_orig_shapes[0,i_seq].cpu().numpy().astype(int), 
					save=i_seq,
					path=i_epoch)
				
		val_loss.append([sum(tracked_val_losses[node_i])/len(tracked_val_losses[node_i]) for node_i in range(bpn.num_nodes)])
		val_steps.append(batches_seen)


def train_dnbp(config):
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


	train_dataset = fphab_dataset.FPHABDataset(data_dir, mode='train', 
														window_size=20,
														transform=train_transforms)
	#test_dataset = fphab_dataset.FPHABDataset(data_dir, mode='test', 
	#													transform=train_transforms)
	val_dataset = fphab_dataset.FPHABDataset(data_dir, mode='val', 
														transform=train_transforms)

	train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size,
							shuffle=True, num_workers=0, drop_last=True)
	#test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size,
	#						shuffle=True, num_workers=0)
	val_dataloader = DataLoader(val_dataset, batch_size=test_batch_size,
	                         shuffle=True, num_workers=0)

	print()
	print("Length of training data:", len(train_dataset))
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


	#bpn.reinit_particles(train_batch_size)

	# bpn.frac_resamp = config["initial_resample_rate"]
	# bpn.use_time = True



	#
	# Initialize optimizers for different factors
	
	if "start_epoch" in config.keys():
		print("Starting from "+str(config["start_epoch"]))
		start_epoch=config["start_epoch"]+1
		bpn.node_likelihoods.load_state_dict(torch.load(os.path.join(config["model_folder"], "epoch_"+str(config["start_epoch"]),"node_liks.pt")), strict=False)
		bpn.likelihood_features.load_state_dict(torch.load(os.path.join(config["model_folder"], "epoch_"+str(config["start_epoch"]),"lik_feats.pt")), strict=False)
		bpn.edge_densities.load_state_dict(torch.load(os.path.join(config["model_folder"], "epoch_"+str(config["start_epoch"]),"edge_dense.pt")), strict=False)
		bpn.edge_samplers.load_state_dict(torch.load(os.path.join(config["model_folder"], "epoch_"+str(config["start_epoch"]),"edge_samps.pt")), strict=False)
		bpn.time_samplers.load_state_dict(torch.load(os.path.join(config["model_folder"], "epoch_"+str(config["start_epoch"]),"time_samps.pt")), strict=False)
		
		optimizers = []
		schedulers = []
		for _ in range(bpn.num_nodes):
			edge_set = []
			esampparam = []
			edampparam = []
			for src_i, src_ in enumerate(bpn.inc_nghbrs[_]):
				# Determine edge index of message pass from src_->dst_
				edge_i = ((bpn.graph == torch.tensor([[min(src_,_)],
							[max(src_,_)]])).all(dim=0).nonzero().squeeze(0) 
						  == bpn.edge_set).nonzero().squeeze(0)
				edg = int(edge_i.item())
				esampparam.extend(list(bpn.edge_samplers[edg].parameters()))
				edampparam.extend(list(bpn.edge_densities[edg].parameters()))

			optimizers.append({
				"lik_optimizer": torch.optim.Adam(list(bpn.node_likelihoods[_].parameters())\
												+list(bpn.likelihood_features.parameters()), lr=config["lr"]),
				"smplr_optimizer": torch.optim.Adam(esampparam, lr=config["lr"]),
				"dens_optimizer": torch.optim.Adam(edampparam, lr=config["lr"]),
				"time_optimizer": torch.optim.Adam(bpn.time_samplers[_].parameters(), lr=config["lr"]),
			})
			schedulers.append({"lik_scheduler": lr_scheduler.ExponentialLR(optimizers[-1]["lik_optimizer"], config["gamma"]),
								"smplr_scheduler": lr_scheduler.ExponentialLR(optimizers[-1]["smplr_optimizer"], config["gamma"]),
								"dens_scheduler": lr_scheduler.ExponentialLR(optimizers[-1]["dens_optimizer"], config["gamma"]),
								"time_scheduler": lr_scheduler.ExponentialLR(optimizers[-1]["time_optimizer"], config["gamma"])})
	else:
		start_epoch = 0
		optimizers = []
		schedulers = []
		for _ in range(bpn.num_nodes):
			edge_set = []
			esampparam = []
			edampparam = []
			for src_i, src_ in enumerate(bpn.inc_nghbrs[_]):
				# Determine edge index of message pass from src_->dst_
				edge_i = ((bpn.graph == torch.tensor([[min(src_,_)],
							[max(src_,_)]])).all(dim=0).nonzero().squeeze(0) 
						  == bpn.edge_set).nonzero().squeeze(0)
				edg = int(edge_i.item())
				esampparam.extend(list(bpn.edge_samplers[edg].parameters()))
				edampparam.extend(list(bpn.edge_densities[edg].parameters()))

			optimizers.append({
				"lik_optimizer": torch.optim.Adam(list(bpn.node_likelihoods[_].parameters())\
												+list(bpn.likelihood_features.parameters()), lr=config["lr"]),
				"smplr_optimizer": torch.optim.Adam(esampparam, lr=config["lr"]),
				"dens_optimizer": torch.optim.Adam(edampparam, lr=config["lr"]),
				"time_optimizer": torch.optim.Adam(bpn.time_samplers[_].parameters(), lr=config["lr"]),
			})
			schedulers.append({"lik_scheduler": lr_scheduler.ExponentialLR(optimizers[-1]["lik_optimizer"], config["gamma"]),
								"smplr_scheduler": lr_scheduler.ExponentialLR(optimizers[-1]["smplr_optimizer"], config["gamma"]),
								"dens_scheduler": lr_scheduler.ExponentialLR(optimizers[-1]["dens_optimizer"], config["gamma"]),
								"time_scheduler": lr_scheduler.ExponentialLR(optimizers[-1]["time_optimizer"], config["gamma"])})


	

	print()  
	print('************')
	print('Start of Epoch',start_epoch)
	print('************')
	print()

	#
	# Training loop
	# 
	batches_seen = 0
	train_loss = []
	val_loss = []
	val_steps = []
	os.makedirs(os.path.join(config["model_folder"],'init'), exist_ok=True)
	run_test(config, bpn, val_dataloader, os.path.join(config["model_folder"],'init'), val_loss, val_steps, batches_seen, device)


	for i_epoch in range(start_epoch, config["num_epochs"]):
		epoch_folder = 'epoch_'+str(i_epoch)
		epoch_path = os.path.join(config["model_folder"], epoch_folder)
		os.makedirs(epoch_path, exist_ok=True)

		
		
		bpn = bpn.train()
		bpn.particle_count = config["train_particle_count"]
		for i_batch, sample_batched in tqdm(enumerate(train_dataloader)):
			# Window: B x S x C x H x W
			# Labels: B x S x 3 x 2
			batch_depths = sample_batched['depth'].type(bpn.type)
			batch_labels = sample_batched['label'].type(bpn.type)
			batch_paths = sample_batched['path']
			batch_orig_shapes = sample_batched['orig_shapes']

			bpn.reinit_particles(batch_depths.shape[0], batch_depths[:,0])


			tracked_losses = [[] for _ in range(bpn.num_nodes)]
			# Iterate over sequence dimension
			for i_seq in range(batch_depths.shape[1]):
				tr = batch_labels[:,i_seq].to(device=device)
				dp = batch_depths[:,i_seq].to(device=device)


				for node_i in range(bpn.num_nodes):
					bpn.compute_feats(dp)
					# Run single belief propagation message+belief update
					bpn.update(node_id=node_i, tru=tr)
					smplr_optimizer = optimizers[node_i]["smplr_optimizer"]
					dens_optimizer = optimizers[node_i]["dens_optimizer"]
					lik_optimizer = optimizers[node_i]["lik_optimizer"]
					time_optimizer = optimizers[node_i]["time_optimizer"]
					

					smplr_optimizer.zero_grad()
					dens_optimizer.zero_grad()
					lik_optimizer.zero_grad()
					time_optimizer.zero_grad()
					

					loss = 0
					for mode in config["training_modes"]:
						# Calculate output density at ground truth
						loss -= torch.log(bpn.density_estimation(node_i, tr[:,node_i].unsqueeze(1).unsqueeze(1), mode=mode))
					loss = loss.mean()/len(config["training_modes"])
					tracked_losses[node_i].append(loss.item())

					loss.backward()


					# Perform optimization step
					smplr_optimizer.step()
					dens_optimizer.step()					
					lik_optimizer.step()
					time_optimizer.step()

				bpn.update_time()

			train_loss.append([sum(tracked_losses[node_i])/len(tracked_losses[node_i]) for node_i in range(bpn.num_nodes)])
			batches_seen+=1



		torch.save(bpn.node_likelihoods.state_dict(), os.path.join(epoch_path, "node_liks.pt"))
		torch.save(bpn.likelihood_features.state_dict(), os.path.join(epoch_path, "lik_feats.pt"))
		torch.save(bpn.edge_densities.state_dict(), os.path.join(epoch_path, "edge_dense.pt"))
		torch.save(bpn.edge_samplers.state_dict(), os.path.join(epoch_path, "edge_samps.pt"))
		torch.save(bpn.time_samplers.state_dict(), os.path.join(epoch_path, "time_samps.pt"))


		run_test(config, bpn, val_dataloader, epoch_path, val_loss, val_steps, batches_seen, device)
					
		fig, ax = plt.subplots(2)
		ax[0].plot(np.array(train_loss).mean(1))
		ax[0].plot(np.array(val_steps), np.array(val_loss).mean(1))
		for node_i in range(bpn.num_nodes):
			ax[1].plot(np.array(train_loss)[:,node_i])
		fig.savefig(os.path.join(epoch_path, 'loss.png'))
		plt.cla()
		plt.clf()
		plt.close()


if __name__=='__main__':
	config_file = sys.argv[1]
	config = json.load(open(config_file, 'r'))
	
	print("Training DNBP Configuration")
	print("config_file:", config_file)
	for k,v in config.items():
		print(k, v)
	print()
	print()

	os.makedirs(config["model_folder"], exist_ok=True)
	shutil.copyfile(config_file, os.path.join(config["model_folder"], "config_file.json"))

	# Must pass config file as command argument
	train_dnbp(config=config)









