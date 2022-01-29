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

from diffBP.networks import lstm
from diffBP.datasets import articulated_toy_dataset, articulated_transforms
from diffBP.datasets import pendulum_plotting as pend_plot


#, lstmnet, test_categories, test_loaders, test_batch_size, window_size, device=torch.device('cpu'), data_len=20
def evaluate_test_lstm(config, epoch, data_dir):
	"""Evaluation function which runs through test set to compute error."""
	# Error is broken down by w/wo occlusion and over the sequence length as mean euclidean distance

	device = torch.device(config["device"])
	model_folder = config["model_folder"]
	epoch_folder = os.path.join(model_folder, "epoch_"+str(epoch))

	lstm_model = lstm.LSTM(config["input_dim"], config["enc_hidden_feats_tot"], config["hidden_dim"], config["layer_dim"], 
							config["num_joints"], config["output_dim"]).type(torch.float32).to(device=device)
	lstm_model.load_state_dict(torch.load(os.path.join(epoch_folder, "lstm_model.pt")), strict=False)

	test_categories = config['data_categories']
	if data_dir.find('pendulum')!=-1:
		test_categories = ["dynamic", "noise"]
	elif data_dir.find('spider')!=-1:
		test_categories = ["dynamic_dynamic_noise", "dynamic_static_noise"]

	statistics = json.load(open(os.path.join(config["model_folder"], "statistics.json"), 'r'))[str(config["num_seqs"])]
	means, stds = (torch.tensor(statistics[0]), torch.tensor(statistics[1]))
	test_transforms = torchvision.transforms.Compose([articulated_transforms.ToTensor(),
						articulated_transforms.Resize(size=128),
						articulated_transforms.Normalize(means, stds)
						])

	if data_dir.find('spider')!=-1:
		seq_len = 20
	else:
		seq_len = 100
	test_datasets = [articulated_toy_dataset.ArticulatedToyDataset(data_dir, mode='', num_seqs=None,
																		categories=[cat],
																		window_size=seq_len, data_max_length=seq_len, 
																		transform=test_transforms)
																		for cat in test_categories]
	test_dataloaders = [DataLoader(test_dataset, batch_size=config["test_batch_size"],
	shuffle=False, num_workers=0) for test_dataset in test_datasets]



	lstm_model = lstm_model.eval()
	all_errors = {}
	with torch.no_grad():

		for i_cat in range(len(test_categories)):

			cat_errors = [[] for _ in range(lstm_model.num_joints)]
			for i_batch, sample_batched in tqdm(enumerate(test_dataloaders[i_cat]), total=len(test_dataloaders[i_cat])):
				# Window: B x S x C x H x W
				# Labels: B x S x 3 x 2
				batch_images = sample_batched['window'].type(torch.float32)
				batch_labels = sample_batched['labels'].type(torch.float32)
				# For now skip theta
				batch_labels = batch_labels[:,:,:,:2]

				hidden = [torch.zeros(config["layer_dim"],batch_images.shape[0],config["hidden_dim"]).type(torch.float32).to(device=device), 
				torch.zeros(config["layer_dim"],batch_images.shape[0],config["hidden_dim"]).type(torch.float32).to(device=device)]


				input = batch_images.to(device=device)
				target = batch_labels.to(device=device)

				pred, hidden = lstm_model(input, hidden)

				for i_seq in range(seq_len):
					euc = torch.sqrt(((64*pred[:,i_seq] - 64*target[:,i_seq])**2).sum(dim=-1)).cpu()
					for i_joint in range(euc.shape[1]):
						cat_errors[i_joint].extend([e.item() for e in euc[:,i_joint]])


			all_errors[test_categories[i_cat]] = cat_errors

	return all_errors





if __name__=='__main__':
	config_file = sys.argv[1]
	epoch = sys.argv[2]
	data_dir = sys.argv[3]
	print(data_dir)
	config = json.load(open(config_file, 'r'))

	print("LSTM Quantitative Evaluation")
	print("config_file:", config_file)
	for k,v in config.items():
		print(k, v)
	print()
	print()

	errors = evaluate_test_lstm(config, epoch, data_dir)


	if len(sys.argv)>4:
		num_dist = sys.argv[4]
		epoch_folder = os.path.join(config["model_folder"], "epoch_"+str(epoch))
		with open(os.path.join(epoch_folder, 'evaluation_errors_'+str(num_dist)+'.pkl'),'wb') as f:
			pickle.dump(errors, f)
	else:
		epoch_folder = os.path.join(config["model_folder"], "epoch_"+str(epoch))
		with open(os.path.join(epoch_folder, 'evaluation_errors.pkl'),'wb') as f:
			pickle.dump(errors, f)