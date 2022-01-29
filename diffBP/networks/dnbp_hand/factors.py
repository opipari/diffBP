import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class PairwiseDensity(nn.Module):
    def __init__(self, particle_size=2, n_hidden=3, n_features=32):
        super(PairwiseDensity, self).__init__()
        
        self.layers = nn.Sequential(*nn.ModuleList([nn.Linear(particle_size, n_features), nn.ReLU()]
                                                    + [l for l in n_hidden*[nn.Linear(n_features, n_features), nn.ReLU()]]
                                                    + [nn.Linear(n_features, 1), nn.Sigmoid()]))

    # Input is num_pairs x (2*particle_size)
    def forward(self, x, minn=0.005):
        out = self.layers(x*20)
        out = (out * (1-minn)) + minn
        return out
    
    
    
    
    
class PairwiseSampler(nn.Module):
    def __init__(self, particle_size=2, n_hidden=1, n_features=64, rand_features=64, est_bounds=1, device=torch.device('cpu'), precision=torch.float32):
        super(PairwiseSampler, self).__init__()
        
        self.est_bounds = est_bounds
        self.rand_features = rand_features
        self.device = device
        self.type = precision
        self.layers = nn.Sequential(*nn.ModuleList([nn.Linear(rand_features, n_features), nn.ReLU()]
                                                    + [l for l in n_hidden*[nn.Linear(n_features, n_features), nn.ReLU()]] 
                                                    + [nn.Linear(n_features, particle_size)]))#, nn.Tanh()
    
    
    def forward(self, num_samples):
        with torch.cuda.device(self.device):
            rand = torch.cuda.FloatTensor(num_samples, self.rand_features).normal_().type(self.type) #torch.randn((num_samples, self.rand_features)).type(self.type).to(device=self.device)
        return self.layers(rand)*self.est_bounds/2
    
    
    
class LikelihoodFeatures(nn.Module):
    def __init__(self, input_dim=1, hidden_features=128, output_feats=64):
        super(LikelihoodFeatures, self).__init__()
       
        self.input_size_full = 96
        self.input_size_half = 48
        self.input_size_qurt = 24

        self.full_skip1 = nn.Sequential(
            nn.Conv2d(input_dim, hidden_features, 3),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.full_skip2 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 3),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_features, hidden_features, 3),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU()
        )

        self.half_skip = nn.Sequential(
            nn.Conv2d(input_dim, hidden_features, 3),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_features, hidden_features, 3),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_features, hidden_features, 3),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU()
        )

        self.qurt_skip = nn.Sequential(
            nn.Conv2d(input_dim, hidden_features, 3),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_features, hidden_features, 3),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_features, hidden_features, 3),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU()
        )

        self.combine = nn.Conv2d(4*hidden_features, hidden_features//2, 1)
        self.norm = nn.BatchNorm2d(hidden_features//2)
   
    def forward(self, x):
        x_full = F.interpolate(x, (self.input_size_full,self.input_size_full))
        x_half = F.interpolate(x, (self.input_size_half,self.input_size_half))
        x_qurt = F.interpolate(x, (self.input_size_qurt,self.input_size_qurt))

        full_feats_1 = F.interpolate(self.full_skip1(x_full), (self.input_size_half,self.input_size_half))
        full_feats_2 = F.interpolate(self.full_skip2(full_feats_1), (self.input_size_half,self.input_size_half))
        half_feats = F.interpolate(self.half_skip(x_half), (self.input_size_half,self.input_size_half))
        qurt_feats = F.interpolate(self.qurt_skip(x_qurt), (self.input_size_half,self.input_size_half))

        out = F.relu(self.norm(self.combine(torch.cat([full_feats_1,full_feats_2,half_feats,qurt_feats],dim=1))))
        out = torch.flatten(out,start_dim=1)

        return out
    
class UnaryDensity(nn.Module):
    def __init__(self, particle_size=2, gross_features=153600, in_features=64, n_hidden=1, n_features=64):
        super(UnaryDensity, self).__init__()
        self.feat_reducer = nn.Sequential(nn.Linear(gross_features,in_features), nn.BatchNorm1d(in_features), nn.ReLU())

        self.layers = nn.Sequential(*nn.ModuleList([nn.Linear(3+in_features, n_features), nn.ReLU()]
                                                    + [l for l in n_hidden*[nn.Linear(n_features, n_features), nn.ReLU()]]
                                                    + [nn.Linear(n_features, 1), nn.Sigmoid()]))

    # Input is num_pairs x (2*particle_size)
    def forward(self, msg_particles, feats, minn=0.0005):
        out = self.feat_reducer(feats)
        out = torch.cat((msg_particles,
                        out.unsqueeze(1).repeat((1,msg_particles.shape[0]//feats.shape[0],1)).view(msg_particles.shape[0],-1)),
                        dim=1)
        out = self.layers(out)
        #out = (out * (1-minn)) + minn
        out = out + minn
        return out
    