import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffBP.networks.dnbp_synthetic import factors


class Decoder(nn.Module):
	def __init__(self, output_dim, input_dim=64):
		super(Decoder, self).__init__()
		self.decoder = nn.Sequential(
			nn.Linear(input_dim, 64),
			nn.ReLU(),
			nn.Linear(64, 32),
			nn.ReLU(),
			nn.Linear(32, output_dim))
	def forward(self,x):
		return self.decoder(x)



class LSTM(nn.Module):
	def __init__(self, input_dim, enc_hidden_feats_tot, hidden_dim, num_layer, num_joints, output_dim):
		super(LSTM, self).__init__()
		self.output_dim = output_dim
		self.input_size = 128
		self.num_joints = num_joints
		# Modify network to be vanilla LSTM
		self.encoder = factors.LikelihoodFeatures(hidden_features=enc_hidden_feats_tot, output_feats=enc_hidden_feats_tot)
		self.fc = nn.Sequential(
			nn.Linear(enc_hidden_feats_tot,hidden_dim),
			nn.ReLU())

		self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layer, batch_first=True)
		self.decoders = nn.ModuleList([Decoder(output_dim, input_dim=hidden_dim) for _ in range(num_joints)])


	def forward(self, x, hidden):
		batch_size = x.shape[0]
		window_size = x.shape[1]
		x = F.interpolate(x.view(x.shape[0]*x.shape[1], x.shape[2], x.shape[3], x.shape[4]), (self.input_size, self.input_size))

		features =  self.encoder(x)
		
		lstm_input = self.fc(features).view(batch_size, window_size, -1)
		# print(lstm_input.shape, [hid.shape for hid in hidden])
		lstm_output, hidden = self.lstm(lstm_input, hidden)
		# print(lstm_output.shape, [hid.shape for hid in hidden])
		# print((lstm_output[0,-1]==hidden[0][0,0]).all(), (lstm_output[0,-1]==hidden[0][1,0]).all())
		# print()
		# print()
		lstm_output = lstm_output.reshape(window_size*batch_size,-1)

		out = self.decoders[0](lstm_output)
		out = out.reshape(batch_size, window_size, -1).unsqueeze(2)
		for i in range(self.num_joints-1):
			output = self.decoders[i+1](lstm_output)
			output = output.reshape(batch_size, window_size, -1).unsqueeze(2)
			out = torch.cat((out, output), 2)

		return out, hidden