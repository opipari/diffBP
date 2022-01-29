import os
import sys
import time
import shutil
import json

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from skimage import io

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torch.utils.data import Dataset, DataLoader


import pickle

from diffBP.networks.dnbp_synthetic import factors, dnbp
from diffBP.datasets import articulated_toy_dataset, articulated_transforms
from diffBP.datasets import pendulum_plotting as pend_plot




def evaluate_test_dnbp(config, epoch, data_dir, num_test_particles=200, num_test_messages=2):
	"""Evaluation function which runs through test set to compute error."""
	# Error is broken down by w/wo occlusion and over the sequence length as mean euclidean distance
	graph = torch.tensor(config["graph"])
	edge_set = torch.tensor(config["edge_set"])
	device = torch.device(config["device"])
	# test_categories = config['data_categories']


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
						density_std=config["std"], 
						lambd=config["lambd"],
						device=device,
						precision=config["precision"])
	
	epoch_folder = os.path.join(config["model_folder"], "epoch_"+str(epoch))
	bpn.node_likelihoods.load_state_dict(torch.load(os.path.join(epoch_folder,"node_liks.pt"),map_location=device), strict=False)
	bpn.likelihood_features.load_state_dict(torch.load(os.path.join(epoch_folder,"lik_feats.pt"),map_location=device), strict=False)
	bpn.edge_densities.load_state_dict(torch.load(os.path.join(epoch_folder,"edge_dense.pt"),map_location=device), strict=False)
	bpn.edge_samplers.load_state_dict(torch.load(os.path.join(epoch_folder,"edge_samps.pt"),map_location=device), strict=False)
	bpn.time_samplers.load_state_dict(torch.load(os.path.join(epoch_folder,"time_samps.pt"),map_location=device), strict=False)

	bpn.particle_count = 200
	output_path = os.path.join(epoch_folder,str(bpn.particle_count)+"_particles")
	os.makedirs(output_path, exist_ok=True)
	bpn.reinit_particles(config["test_batch_size"])

	bpn.use_time = True



	statistics = json.load(open(os.path.join(config["model_folder"], "statistics.json"), 'r'))[str(config["num_seqs"])]
	means, stds = (torch.tensor(statistics[0]), torch.tensor(statistics[1]))
	test_transforms = torchvision.transforms.Compose([articulated_transforms.ToTensor(),
															articulated_transforms.Resize(size=128),
															articulated_transforms.Normalize(means, stds)
															])

	test_categories = config['data_categories']
	if data_dir.find('pendulum')!=-1:
		test_categories = ["dynamic", "noise"]
	elif data_dir.find('spider')!=-1:
		test_categories = ["dynamic_dynamic_noise", "dynamic_static_noise"]
	

	if data_dir.find('spider')!=-1:
		seq_len = 20
	else:
		seq_len = 100
	test_datasets = [articulated_toy_dataset.ArticulatedToyDataset(data_dir, mode='', num_seqs=None,
																	categories=[cat],
																	window_size=seq_len, data_max_length=seq_len, 
																	transform=test_transforms)
						for cat in test_categories]

	test_loaders = [DataLoader(test_dataset, batch_size=config["test_batch_size"], shuffle=False, num_workers=0) for test_dataset in test_datasets]



	bpn = bpn.eval()
	all_errors = {}
	with torch.no_grad():

		for i_cat in range(len(test_categories)):

			cat_errors = [[] for _ in range(bpn.num_nodes)]
			for i_batch, sample_batched in tqdm(enumerate(test_loaders[i_cat]), total=len(test_loaders[i_cat])):
				# Window: B x S x C x H x W
				# Labels: B x S x 3 x 2
				batch_images = sample_batched['window'].type(bpn.type)
				batch_labels = sample_batched['labels'].type(bpn.type)

				# For now skip theta
				batch_labels = batch_labels[:,:,:,:2]

				bpn.reinit_particles(batch_images.shape[0])
				bpn.frac_resamp = 1.0

				# Iterate over sequence dimension
				for i_seq in range(batch_images.shape[1]):
					tr = batch_labels[:,i_seq].to(device=bpn.device)
					x = batch_images[:,i_seq].to(device=bpn.device)

					bpn.compute_feats(x)
					for msg_pass in range(num_test_messages):
						bpn.update()

					max_parts, pred_weights = bpn.max_marginals()

					for i_joint in range(bpn.num_nodes):
						euc = torch.sqrt(((64*max_parts[i_joint]-64*tr[:,i_joint])**2).sum(dim=-1)).cpu().squeeze()
						cat_errors[i_joint].extend([e.item() for e in euc])
				
			all_errors[test_categories[i_cat]] = cat_errors

	return all_errors


if __name__=='__main__':
	config_file = sys.argv[1]
	epoch = sys.argv[2]
	data_dir = sys.argv[3]
	config = json.load(open(config_file, 'r'))
	
	print("DNBP Quantitative Evaluation")
	print("config_file:", config_file)
	for k,v in config.items():
		print(k, v)
	print()
	print()

	errors = evaluate_test_dnbp(config, epoch, data_dir, num_test_particles=200, num_test_messages=2)
	
	epoch_folder = os.path.join(config["model_folder"], "epoch_"+str(epoch))
	output_path = os.path.join(epoch_folder,"200_particles")
	if len(sys.argv)>4:
		num_dist = sys.argv[4]
		with open(os.path.join(output_path, 'evaluation_errors_'+str(num_dist)+'.pkl'),'wb') as f:
			pickle.dump(errors, f)
	else:
		with open(os.path.join(output_path, 'evaluation_errors.pkl'),'wb') as f:
			pickle.dump(errors, f)

