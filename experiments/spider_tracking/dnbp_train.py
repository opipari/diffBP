import os
import sys
import time
import shutil
import json

import numpy as np
from PIL import Image

from tqdm import tqdm

import matplotlib.pyplot as plt
from skimage import io

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler

import pickle

from diffBP.networks.dnbp_synthetic import factors, dnbp
from diffBP.datasets import articulated_toy_dataset, articulated_transforms
from diffBP.datasets import pendulum_plotting as pend_plot




def train_dnbp(config):
	torch.autograd.set_detect_anomaly(True)
	#
	# Setup train and test datasets
	#
	data_dir = config["data_dir"]
	train_batch_size = config["train_batch_size"]
	test_batch_size = config["test_batch_size"]
	train_num_seqs = config["num_seqs"]
	device = torch.device(config["device"])


	statistics = json.load(open(os.path.join(data_dir, "Train", "statistics.json"), 'r'))[str(train_num_seqs)]
	means, stds = (torch.tensor(statistics[0]), torch.tensor(statistics[1]))
	shutil.copyfile(os.path.join(data_dir, "Train", "statistics.json"), os.path.join(config["model_folder"],"statistics.json"))
	if config["use_augmentation"]:
		if data_dir.find('spider')!=-1:
			train_transforms = torchvision.transforms.Compose([articulated_transforms.RandomSequenceFlip(),
																articulated_transforms.ToTensor(),
																articulated_transforms.Resize(size=128),
																articulated_transforms.RandomGaussianNoise(device=device),
																articulated_transforms.Normalize(means, stds)])
		else:
			train_transforms = torchvision.transforms.Compose([articulated_transforms.ToTensor(),
																articulated_transforms.Resize(size=128),
																articulated_transforms.RandomGaussianNoise(device=device),
																articulated_transforms.Normalize(means, stds)
																])
	else:
		train_transforms = torchvision.transforms.Compose([articulated_transforms.ToTensor(),
															articulated_transforms.Resize(size=128),
															articulated_transforms.Normalize(means, stds)
															])

	test_transforms = torchvision.transforms.Compose([articulated_transforms.ToTensor(),
															articulated_transforms.Resize(size=128),
															articulated_transforms.Normalize(means, stds)
															])

	print("Training Transformations:")
	print(train_transforms)
	print()
	print()


	train_dataset = articulated_toy_dataset.ArticulatedToyDataset(data_dir, mode='Train', num_seqs=train_num_seqs,
														categories=config['data_categories'],
														window_size=20, data_max_length=20, 
	                                                 transform=train_transforms)
	test_datasets = [articulated_toy_dataset.ArticulatedToyDataset(data_dir, mode='Val', num_seqs=None,
														categories=[cat],
														window_size=20, data_max_length=20, 
	                                                 transform=test_transforms)
						for cat in config['data_categories']]
	val_dataset = articulated_toy_dataset.ArticulatedToyDataset(data_dir, mode='Val',  num_seqs=None,
														categories=config['data_categories'],
														window_size=20, data_max_length=20, 
	                                                 transform=test_transforms)

	train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size,
	                        shuffle=True, num_workers=0)
	test_dataloaders = [DataLoader(test_dataset, batch_size=test_batch_size,
	                        shuffle=True, num_workers=0) for test_dataset in test_datasets]
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


	bpn.reinit_particles(train_batch_size)

	# bpn.frac_resamp = config["initial_resample_rate"]
	bpn.use_time = True

	#
	# Initialize optimizers for different factors
	
	if "start_epoch" in config.keys():
		start_epoch=config["start_epoch"]+1
		bpn.node_likelihoods.load_state_dict(torch.load(os.path.join(config["model_folder"], "epoch_"+str(config["start_epoch"]),"node_liks.pt")), strict=False)
		bpn.likelihood_features.load_state_dict(torch.load(os.path.join(config["model_folder"], "epoch_"+str(config["start_epoch"]),"lik_feats.pt")), strict=False)
		bpn.edge_densities.load_state_dict(torch.load(os.path.join(config["model_folder"], "epoch_"+str(config["start_epoch"]),"edge_dense.pt")), strict=False)
		bpn.edge_samplers.load_state_dict(torch.load(os.path.join(config["model_folder"], "epoch_"+str(config["start_epoch"]),"edge_samps.pt")), strict=False)
		bpn.time_samplers.load_state_dict(torch.load(os.path.join(config["model_folder"], "epoch_"+str(config["start_epoch"]),"time_samps.pt")), strict=False)
		
		if config["train_mode"]=="joint":
			lik_optimizer = torch.load(os.path.join(config["model_folder"], "epoch_"+str(config["start_epoch"]),'lik_optim.pt'))
			smplr_optimizer = torch.load(os.path.join(config["model_folder"], "epoch_"+str(config["start_epoch"]),'smplr_optim.pt'))
			dens_optimizer = torch.load(os.path.join(config["model_folder"], "epoch_"+str(config["start_epoch"]),'dens_optim.pt'))
			time_optimizer = torch.load(os.path.join(config["model_folder"], "epoch_"+str(config["start_epoch"]),'time_optim.pt'))
		else:
			optimizers = torch.load(os.path.join(config["model_folder"], "epoch_"+str(config["start_epoch"]),'bpnet_optim.pt'))
			schedulers = []
			for opt in optimizers:
				schedulers.append({"lik_scheduler":lr_scheduler.ExponentialLR(opt["lik_optimizer"], config["gamma"]),
									"smplr_scheduler":lr_scheduler.ExponentialLR(opt["smplr_optimizer"], config["gamma"]),
									"dens_scheduler":lr_scheduler.ExponentialLR(opt["dens_optimizer"], config["gamma"]),
									"time_scheduler":lr_scheduler.ExponentialLR(opt["time_optimizer"], config["gamma"])})
	else:
		start_epoch = 0
		if config["train_mode"]=="joint":
			lik_optimizer = torch.optim.Adam(list(bpn.node_likelihoods.parameters()) \
											 + list(bpn.likelihood_features.parameters()), lr=config["lr"])
			smplr_optimizer = torch.optim.Adam(bpn.edge_samplers.parameters(), lr=config["lr"])
			dens_optimizer = torch.optim.Adam(bpn.edge_densities.parameters(), lr=config["lr"])
			time_optimizer = torch.optim.Adam(bpn.time_samplers.parameters(), lr=config["lr"])
		else:
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
					"lik_optimizer": torch.optim.Adam(list(bpn.node_likelihoods[_].parameters()) \
													 + list(bpn.likelihood_features[_].parameters()), lr=config["lr"]),
					"smplr_optimizer": torch.optim.Adam(esampparam, lr=config["lr"]),
					"dens_optimizer": torch.optim.Adam(edampparam, lr=config["lr"]),
					"time_optimizer": torch.optim.Adam(bpn.time_samplers[_].parameters(), lr=config["lr"]),
				})
				schedulers.append({"lik_scheduler": lr_scheduler.ExponentialLR(optimizers[-1]["lik_optimizer"], config["gamma"]),
									"smplr_scheduler": lr_scheduler.ExponentialLR(optimizers[-1]["smplr_optimizer"], config["gamma"]),
									"dens_scheduler": lr_scheduler.ExponentialLR(optimizers[-1]["dens_optimizer"], config["gamma"]),
									"time_scheduler": lr_scheduler.ExponentialLR(optimizers[-1]["time_optimizer"], config["gamma"])})


	nump = 100
	x = np.linspace(-1, 1, nump)
	y = np.linspace(-1, 1, nump)
	xv, yv = np.meshgrid(x, y)
	grid = torch.from_numpy(np.concatenate([np.expand_dims(xv.flatten(), axis=1), np.expand_dims(yv.flatten(),axis=1)], axis=1))
	grid = grid.view((nump*nump,1,2)).double().to(device=device)



	print()  
	print('************')
	print('Start of Epoch',start_epoch)
	# bpn.frac_resamp = 1 - ((1 - bpn.frac_resamp) * config["resample_growth_rate"])
	# print('frac:',bpn.frac_resamp)
	print('************')
	print()

	#
	# Training loop
	# 
	train_loss = []
	val_loss = []
	for i_epoch in range(start_epoch, config["num_epochs"]):
		epoch_folder = 'epoch_'+str(i_epoch)
		epoch_path = os.path.join(config["model_folder"], epoch_folder)
		os.makedirs(epoch_path, exist_ok=True)


		epoch_val_loss = []
		epoch_train_loss = []

		bpn = bpn.train()
		epoch_seqs = 0
		while epoch_seqs<1000:
			for i_batch, sample_batched in tqdm(enumerate(train_dataloader)):
				epoch_seqs += train_batch_size
				if epoch_seqs>1000:
					break
				# Window: B x S x C x H x W
				# Labels: B x S x 3 x 2
				batch_images = sample_batched['window'].type(bpn.type)
				batch_labels = sample_batched['labels'].type(bpn.type)

				# For now skip theta
				batch_labels = batch_labels[:,:,:,:2]

				bpn.reinit_particles(batch_images.shape[0])
				

				tracked_losses = []
				# Iterate over sequence dimension
				for i_seq in range(batch_images.shape[1]):
					tr = batch_labels[:,i_seq].to(device=device)
					x = batch_images[:,i_seq].to(device=device)


					if config["train_mode"]=="marginal":
						bpn.compute_feats(x)


						for i in range(tr.shape[1]):

							# Run single belief propagation message+belief update
							bpn.update(node_id=i, tru=tr)

							smplr_optimizer = optimizers[i]["smplr_optimizer"]
							dens_optimizer = optimizers[i]["dens_optimizer"]
							lik_optimizer = optimizers[i]["lik_optimizer"]
							time_optimizer = optimizers[i]["time_optimizer"]


							smplr_optimizer.zero_grad()
							dens_optimizer.zero_grad()
							lik_optimizer.zero_grad()
							time_optimizer.zero_grad()

							loss = 0
							for mode in config["training_modes"]:
								# Calculate output density at ground truth
								loss -= torch.log(1e-50+bpn.density_estimation(i, tr[:,i].unsqueeze(1).unsqueeze(1), mode=mode))
							loss = loss.mean()/len(config["training_modes"])
							tracked_losses.append(loss.item())

							loss.backward()
							

							# Perform optimization step
							if (i_epoch>4 or data_dir.find('spider')==-1) and 'w_unary' in config["training_modes"]:
								smplr_optimizer.step()
							    
							if 'w_neigh' in config["training_modes"]:
								dens_optimizer.step()

							if 'w_lik' in config["training_modes"]:
								lik_optimizer.step()

							time_optimizer.step()


					bpn.update_time()


				if i_batch%10==0:
					epoch_train_loss.append(sum(tracked_losses)/len(tracked_losses))
					# Plot output

		with torch.no_grad():
			try:
				to_show=0#np.random.randint(0,batch)
				# print('Loss:', loss)

				unnormed_im = (x*stds.view(1,3,1,1).to(device=device))+means.view(1,3,1,1).to(device=device)
				unnormed_im = torch.clamp(unnormed_im, min=0, max=1)

				# print('Unary Samples:')
				pend_plot.plot_unaries(bpn, grid, x, unnormed=unnormed_im, to_show=to_show, est_bounds=1.0, fname=os.path.join(epoch_path, 'unaries'+str(i_batch)+'.jpg'))

				# print('Belief Density:')
				pend_plot.plot_belief_particles(bpn, grid, x, unnormed=unnormed_im, to_show=to_show, est_bounds=1.0, fname=os.path.join(epoch_path, 'belief'+str(i_batch)+'.jpg'))

				# print('Pairwise Samples:')
				pend_plot.plot_pairwise_sampling(bpn, num_samples=2000, est_bounds=1.0, fname=os.path.join(epoch_path, 'pairwise_samples'+str(i_batch)+'.jpg'))

				# print('Pairwise Densities:')
				pend_plot.plot_pairwise_densities(bpn, grid, est_bounds=1.0, fname=os.path.join(epoch_path, 'pairwise_densities'+str(i_batch)+'.jpg'))

				# print('Particle Liks:')
				pend_plot.plot_msg_wgts(bpn, x, 'w_lik', unnormed=unnormed_im, bin_size=0.1, fname=os.path.join(epoch_path, 'likelihoods'+str(i_batch)+'.jpg'))

				# print('Neighbor Liks:')
				pend_plot.plot_msg_wgts(bpn, x, 'w_unary', unnormed=unnormed_im, bin_size=0.1, fname=os.path.join(epoch_path, 'neighboring_likelihoods'+str(i_batch)+'.jpg'))

				# print('Neigbor Dens:')
				pend_plot.plot_msg_wgts(bpn, x, 'w_neigh', unnormed=unnormed_im, bin_size=0.1, fname=os.path.join(epoch_path, 'neighboring_densities'+str(i_batch)+'.jpg'))


				# print('Time Delta Samples:')
				pend_plot.plot_timedelta_sampling(bpn, num_samples=2000, est_bounds=1.0, fname=os.path.join(epoch_path, 'time_samples'+str(i_batch)+'.jpg'))
			except:
				continue


			



			bpn = bpn.eval()
			tracked_val_losses = []
			for v_batch, sample_batched_ in enumerate(val_dataloader):
				if v_batch>10:
					break
				sample_batched = sample_batched_
				# Window: B x S x C x H x W
				# Labels: B x S x 3 x 2
				batch_images = sample_batched['window'].type(bpn.type)
				batch_labels = sample_batched['labels'].type(bpn.type)

				# For now skip theta
				batch_labels = batch_labels[:,:,:,:2]

				bpn.reinit_particles(batch_images.shape[0])

				# Iterate over sequence dimension
				for i_seq in range(batch_images.shape[1]):
					tr = batch_labels[:,i_seq].to(device=device)
					x = batch_images[:,i_seq].to(device=device)

					bpn.compute_feats(x)

					for i in range(tr.shape[1]):
						# Run single belief propagation message+belief update
						bpn.update(node_id=i)

						loss = 0
						for mode in config["training_modes"]:
							# Calculate output density at ground truth
							loss -= torch.log(1e-50+bpn.density_estimation(i, tr[:,i].unsqueeze(1).unsqueeze(1), mode=mode))
						loss = loss.mean()/len(config["training_modes"])
						tracked_val_losses.append(loss.item())
					bpn.update_time()

			bpn = bpn.train()
			epoch_val_loss.append(sum(tracked_val_losses)/len(tracked_val_losses))
					
				
			

					# print(bpn.time_samplers[0].layers[0].weight.abs().max())

		val_loss.append(sum(epoch_val_loss)/len(epoch_val_loss))
		train_loss.append(sum(epoch_train_loss)/len(epoch_train_loss))

		for sched in schedulers:
			if (i_epoch>4 or data_dir.find('spider')==-1):
				sched["smplr_scheduler"].step()
			sched["lik_scheduler"].step()
			sched["dens_scheduler"].step()
			sched["time_scheduler"].step()


					# if i_batch%50==0 and i_seq==10:
		# if config["train_mode"]=="joint":
		# 	train_loss.append(loss.item())
		# else:
		# 	train_loss.append(tracked_losses)

		
		# if config["train_mode"]=="joint":
		# 	print("Loss:",loss.item())
		# else:
		# 	print("Loss:", tracked_losses)
					
		
	            


		


		errors = articulated_toy_dataset.evaluate_test_bpnet(bpn, config["data_categories"], test_dataloaders, all_data=False)

		with open(os.path.join(epoch_path, 'test_errors.pkl'),'wb') as f:
			pickle.dump(errors, f)

		fig = plt.figure(figsize=(6,4))
		ax = fig.add_axes([0.1, 0.1, 0.55, 0.85])
		# ax.set_ylim(bottom=0, top=0.25)
		ax.plot(range(20), [7 for _ in range(20)], label='Arm Width')
		for cat, err in errors.items():
			ax.plot(range(20), err[0], label=str(bpn.particle_count)+' Particles\nError '+str(cat))
		ax.set_xlabel('Sequence Step')
		ax.set_ylabel('Average Euclidean Distance')
		ax.set_xticks(list(np.arange(0,25,5)))
		ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
		# plt.show()
		fig.savefig(os.path.join(epoch_path, 'evaluation.jpg'), dpi=fig.dpi)


		# if config["train_mode"]=="joint":
		# 	fig = plt.figure()
		# 	plt.plot(range(len(train_loss)), train_loss)
		# 	plt.xlabel('Training Step')
		# 	plt.ylabel('Training Loss')
		# 	# plt.show()
		# 	fig.savefig(os.path.join(epoch_path, 'train_loss.jpg'), dpi=fig.dpi)
		# else:
			
		# 	for joint_i in range(bpn.num_nodes):
		# 		fig = plt.figure()
		# 		plt.plot(range(len(train_loss)), [tl[joint_i] for tl in train_loss])
		# 		plt.xlabel('Training Step')
		# 		plt.ylabel('Training Loss')
		# 		# plt.show()
		# 		fig.savefig(os.path.join(epoch_path, 'train_loss_'+str(joint_i)+'.jpg'), dpi=fig.dpi)


		fig = plt.figure()
		plt.plot(range(len(train_loss)), train_loss, label="Training Loss")
		plt.plot(range(len(val_loss)), val_loss, label="Validation Loss")
		plt.xlabel('Epoch')
		plt.ylabel('Density Loss')
		plt.legend()
		# plt.show()
		fig.savefig(os.path.join(epoch_path, 'train_loss.jpg'), dpi=fig.dpi)
		plt.close('all')


		torch.save(bpn.node_likelihoods.state_dict(), os.path.join(epoch_path, "node_liks.pt"))
		torch.save(bpn.likelihood_features.state_dict(), os.path.join(epoch_path, "lik_feats.pt"))
		torch.save(bpn.edge_densities.state_dict(), os.path.join(epoch_path, "edge_dense.pt"))
		torch.save(bpn.edge_samplers.state_dict(), os.path.join(epoch_path, "edge_samps.pt"))
		torch.save(bpn.time_samplers.state_dict(), os.path.join(epoch_path, "time_samps.pt"))

		if config["train_mode"]=="joint":
			torch.save(lik_optimizer, os.path.join(epoch_path,'lik_optim.pt'))
			torch.save(smplr_optimizer, os.path.join(epoch_path,'smplr_optim.pt'))
			torch.save(dens_optimizer, os.path.join(epoch_path,'dens_optim.pt'))
			torch.save(time_optimizer, os.path.join(epoch_path,'time_optim.pt'))
		else:
			torch.save(optimizers, os.path.join(epoch_path,'bpnet_optim.pt'))
		print()  
		print('************')
		print('End of Epoch',i_epoch)
		# bpn.frac_resamp = 1 - ((1 - bpn.frac_resamp) * config["resample_growth_rate"])
		# print('frac:',bpn.frac_resamp)
		print('************')
		print()


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


