import os
import sys
import time
import shutil
import json

import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision

import pickle
import torch.nn.functional as F

from diffBP.networks import lstm
from diffBP.datasets import articulated_toy_dataset, articulated_transforms
from diffBP.datasets import pendulum_plotting as pend_plot




def train_lstm(config):


	device = torch.device(config["device"])

	#
	# Setup train and test datasets
	#
	data_dir = config["data_dir"]
	train_batch_size = config["train_batch_size"]
	test_batch_size = config["test_batch_size"]
	train_num_seqs = config["num_seqs"]


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
																articulated_transforms.Normalize(means, stds)])
	else:
		train_transforms = torchvision.transforms.Compose([articulated_transforms.ToTensor(),
															articulated_transforms.Resize(size=128),
															articulated_transforms.Normalize(means, stds)])

	print("Training Transformations:")
	print(train_transforms)
	print()
	print()


	test_transforms = torchvision.transforms.Compose([articulated_transforms.ToTensor(),
															articulated_transforms.Resize(size=128),
															articulated_transforms.Normalize(means, stds)])


	train_dataset = articulated_toy_dataset.ArticulatedToyDataset(data_dir, mode='Train', num_seqs=train_num_seqs, 
														categories=config['data_categories'],
														window_size=config["window_size"], data_max_length=20, 
	                                                 transform=train_transforms)
	val_dataset = articulated_toy_dataset.ArticulatedToyDataset(data_dir, mode='Val',  num_seqs=None,
														categories=config['data_categories'],
														window_size=20, data_max_length=20, 
	                                                 transform=test_transforms)

	test_datasets = [articulated_toy_dataset.ArticulatedToyDataset(data_dir, mode='Val',  num_seqs=None,
														categories=[cat],
														window_size=20, data_max_length=20, 
	                                                 transform=test_transforms)
						for cat in config['data_categories']]

	train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size,
	                        shuffle=True, num_workers=0)
	val_dataloader = DataLoader(val_dataset, batch_size=test_batch_size,
	                        shuffle=True, num_workers=0)
	test_dataloaders = [DataLoader(test_dataset, batch_size=test_batch_size,
	                        shuffle=False, num_workers=0) for test_dataset in test_datasets]



	#
	# Setup model to be trained
	#
	
	model = lstm.LSTM(config["input_dim"], config["enc_hidden_feats_tot"], config["hidden_dim"], config["layer_dim"], 
							config["num_joints"], config["output_dim"]).type(torch.float32).to(device=device)

	if "start_epoch" in config.keys():
		start_epoch=config["start_epoch"]+1
		model_path = os.path.join(config["model_folder"], "epoch_"+str(config["start_epoch"]), 'lstm_model.pt')
		optimizer_path = os.path.join(config["model_folder"], "epoch_"+str(config["start_epoch"]), 'lstm_optim.pt')
		model.load_state_dict(torch.load(model_path), strict=False)
		optimizer = torch.load(optimizer_path)
	else:
		start_epoch = 0
		optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])



	scheduler = lr_scheduler.ExponentialLR(optimizer, config["gamma"])


	#
	# Initialize optimizers for different factors
	#
		
	# optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])
	# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.7)

	train_criterion = nn.MSELoss().to(device=device)


	def get_lr(optimizer):
		for param_group in optimizer.param_groups:
			return param_group['lr'] 
	if start_epoch!=0:
		print('Starting LSTM Retraining')
	else:
		print('Starting LSTM Training')

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

		model = model.train()
		epoch_seqs = 0
		while epoch_seqs<1000:
			for i_batch, sample_batched_ in tqdm(enumerate(train_dataloader)):
				epoch_seqs += train_batch_size
				if epoch_seqs>1000:
					break
				sample_batched = sample_batched_
				# Window: B x S x C x H x W
				# Labels: B x S x 3 x 2

				batch_images = sample_batched['window'].type(torch.float32).to(device=device)
				batch_labels = sample_batched['labels'].type(torch.float32).to(device=device)


				# For now skip theta
				batch_labels = batch_labels[:,:,:,:2]

				batch_size = batch_images.shape[0]

				optimizer.zero_grad()

				hidden = (torch.zeros(config["layer_dim"], batch_size, config["hidden_dim"]).type(torch.float32).to(device=device), 
			    			torch.zeros(config["layer_dim"], batch_size, config["hidden_dim"]).type(torch.float32).to(device=device))
				pred, hidden = model(batch_images, hidden)


				# print(pred[0,0,:,0], batch_labels[0,0,:,0])
				# print(train_criterion(pred[0,0,:,0], batch_labels[0,0,:,0]))

				#loss = train_criterion(pred, batch_labels)
				loss = torch.mean(torch.sqrt(torch.sum((pred-batch_labels)**2, 3)))
				loss.backward()

				

				optimizer.step()

				if i_batch%25==0:
					epoch_train_loss.append(loss.item())


					val_l=0
					with torch.no_grad():
						for v_batch, sample_batched_ in enumerate(val_dataloader):
							if v_batch>10:
								break
							sample_batched = sample_batched_
							# Window: B x S x C x H x W
							# Labels: B x S x 3 x 2
							batch_images = sample_batched['window'].type(torch.float32).to(device=device)
							batch_labels = sample_batched['labels'].type(torch.float32).to(device=device)
							# For now skip theta
							batch_labels = batch_labels[:,:,:,:2]

							batch_size = batch_images.shape[0]


							hidden = (torch.zeros(config["layer_dim"], batch_size, config["hidden_dim"]).type(torch.float32).to(device=device), 
						    			torch.zeros(config["layer_dim"], batch_size, config["hidden_dim"]).type(torch.float32).to(device=device))
							pred, hidden = model(batch_images, hidden)

							val_l += torch.mean(torch.sqrt(torch.sum((pred-batch_labels)**2, 3)))


					epoch_val_loss.append(val_l.item()/10)

		val_loss.append(sum(epoch_val_loss)/len(epoch_val_loss))
		train_loss.append(sum(epoch_train_loss)/len(epoch_train_loss))
		scheduler.step()


		
		
		print('loss',loss.item())

		disp_b = 0
		for im in range(config["window_size"]):
			fig, ax = plt.subplots()
			unnormed_im = (sample_batched['window'][disp_b][im].permute((1,2,0))*stds.view(1,1,3))+means.view(1,1,3)
			ax.imshow(torch.clamp(unnormed_im, min=0, max=1), extent=[-1, 1, -1, 1])
			if config["num_joints"]==3:
				clrs=['r','g','b']
				clrs_tru=['pink', 'yellow', 'black']
			else:
				clrs=['y', "#24e056", "#f50505", "#0800eb", "#1b5e2d", "#d17171", "#64b1e8"]
				clrs_tru=7*['r']
			for i_joint in range(config["num_joints"]):
				# ax.scatter(x=batch_labels[disp_b][im][i_joint][0].cpu().detach(), y=batch_labels[disp_b][im][i_joint][1].cpu().detach(), s=60, c=clrs_tru[i_joint])
				ax.scatter(x=pred[disp_b][im][i_joint][0].cpu().detach(), y=pred[disp_b][im][i_joint][1].cpu().detach(), s=40, c=clrs[i_joint],linewidths=1,edgecolors='black')
			fig.savefig(os.path.join(epoch_path, 'predictions_'+str(im)+'.jpg'), dpi=fig.dpi)
			plt.close('all')
			# print()
			# print()

		# scheduler.step()
		# print(get_lr(optimizer))

		errors = articulated_toy_dataset.evaluate_test_lstm(model, config["data_categories"], test_dataloaders, 
															config["test_batch_size"], config["window_size"], config["layer_dim"], hidden_dim=config["hidden_dim"], device=device)

		with open(os.path.join(epoch_path, 'test_errors.pkl'),'wb') as f:
			pickle.dump(errors, f)

		fig = plt.figure(figsize=(6,4))
		ax = fig.add_axes([0.1, 0.1, 0.55, 0.85])
		# ax.set_ylim(bottom=0, top=0.25)
		ax.plot(range(20), [7 for _ in range(20)], label='Arm Width')
		for cat, err in errors.items():
			ax.plot(range(20), err[0], label='LSTM\nError '+str(cat))
		ax.set_xlabel('Sequence Step')
		ax.set_ylabel('Average Euclidean Distance')
		ax.set_xticks(list(np.arange(0,25,5)))
		ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
		# plt.show()
		fig.savefig(os.path.join(epoch_path, 'evaluation.jpg'), dpi=fig.dpi)

		fig = plt.figure()
		plt.plot(range(len(train_loss)), train_loss, label="Training Loss")
		plt.plot(range(len(val_loss)), val_loss, label="Validation Loss")
		plt.xlabel('Epoch')
		plt.ylabel('Euclidean Loss')
		plt.legend()
		# plt.show()
		fig.savefig(os.path.join(epoch_path, 'train_loss.jpg'), dpi=fig.dpi)
		plt.close('all')

		torch.save(model.state_dict(), os.path.join(epoch_path,'lstm_model.pt'))
		torch.save(optimizer, os.path.join(epoch_path,'lstm_optim.pt'))

		print()  
		print('************')
		print('End of Epoch '+str(i_epoch))
		print('************')
		print()


if __name__=='__main__':
	config_file = sys.argv[1]
	config = json.load(open(config_file, 'r'))
	
	print("Training LSTM Configuration")
	print("config_file:", config_file)
	for k,v in config.items():
		print(k, v)
	print()
	print()

	os.makedirs(config["model_folder"], exist_ok=True)
	shutil.copyfile(config_file, os.path.join(config["model_folder"], "config_file.json"))

	# Must pass config file as command argument
	train_lstm(config)
