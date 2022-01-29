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
    def __init__(self, input_dim=3, hidden_features=32, output_feats=64):
        super(LikelihoodFeatures, self).__init__()
        
        self.input_size = 128
        #print(int(np.floor(np.log2(self.input_size)))-1)

        self.layers = nn.Sequential(
            nn.Conv2d(input_dim, hidden_features, 3),
             nn.ReLU(),
             nn.MaxPool2d(2),
             nn.Conv2d(hidden_features, hidden_features, 3),
             nn.ReLU(),
             nn.MaxPool2d(2),
             nn.Conv2d(hidden_features, hidden_features, 3),
             nn.ReLU(),
             nn.MaxPool2d(2),
             nn.Conv2d(hidden_features, hidden_features, 3),
             nn.ReLU(),
             nn.MaxPool2d(2),
             nn.Conv2d(hidden_features, output_feats, 3),
             nn.ReLU(),
             nn.MaxPool2d(4))
    
    
    def forward(self, x):
        x = F.interpolate(x, (self.input_size,self.input_size))
        return self.layers(x).squeeze(2).squeeze(2)
    
    
    
    
class UnaryDensity(nn.Module):
    def __init__(self, particle_size=2, in_features=64, n_hidden=1, n_features=64):
        super(UnaryDensity, self).__init__()
        
        self.layers = nn.Sequential(*nn.ModuleList([nn.Linear(particle_size+in_features, n_features), nn.ReLU()]
                                                    + [l for l in n_hidden*[nn.Linear(n_features, n_features), nn.ReLU()]]
                                                    + [nn.Linear(n_features, 1), nn.Sigmoid()]))

    # Input is num_pairs x (2*particle_size)
    def forward(self, x, minn=0.005):
        out = self.layers(x)
        out = (out * (1-minn)) + minn
        return out
    