import numpy as np

import torch
import torch.nn.functional as F

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        images, labels = sample['window'], sample['labels']

        # swap color axis because
        # numpy image: S x H x W x C
        # torch image: S x C X H X W
        images = images.transpose((0, 3, 1, 2))
        return {'window': torch.from_numpy(images),
                'labels': torch.from_numpy(labels)}


class RandomSequenceFlip(object):
    """Randomly flip image sequence and labels."""
    
    def __call__(self, sample):
        images, labels = sample['window'], sample['labels']
        # images: S x H x W x C
        # labels: S x #jnts x #pos
        # pos is (horizontal, vertical)
        # center is (0,0), upper left is (-1, 1), lower right is (1, -1)
        
        
        # 50-50 vertical flip
        if np.random.random()>0.5:
            images = np.flip(images,axis=1).copy()
            labels[:,:,1] = -labels[:,:,1]
        
        # 50-50 horizontal flip
        if np.random.random()>0.5:
            images = np.flip(images,axis=2).copy()
            labels[:,:,0] = -labels[:,:,0]
        
        return {'window': images,
                'labels': labels}


class RandomGaussianNoise(object):
    """Randomly flip image sequence and labels."""

    def __init__(self, device=torch.device("cuda:0")):
        self.device=device
    
    def __call__(self, sample):
        images, labels = sample['window'], sample['labels']
        # images: S x H x W x C
        # labels: S x #jnts x #pos
        

        with torch.cuda.device(self.device):
            noise_t = torch.cuda.FloatTensor(images.shape).normal_().cpu().type(torch.float64)*0.0784
        images = torch.clamp(images+noise_t, min=0, max=1)

        return {'window': images,
                'labels': labels}

def random_pattern(shape=(500,500,3)):
    pattern = np.zeros(shape)

    xv, yv = np.meshgrid(np.arange(0,shape[0]), np.arange(0,shape[1]), sparse=False, indexing='ij')
    xv += np.random.randint(0,shape[0])
    yv += np.random.randint(0,shape[1])
    for i in range(3):
        random_color = np.random.rand(2)

        a = random_color[0]
        b = random_color[1]
        scvert = np.random.uniform(low=0.001, high=0.1)
        schorz = np.random.uniform(low=0.001, high=0.1)

        pattern[:,:,i] = ((((np.sin(yv*scvert)+1)/2)*((np.sin(xv*schorz)+1)/2)))*(b-a)+a
        
            
    return pattern

class RandomBackground(object):
    def __init__(self, frac_random=0.25):
        self.frac_random = frac_random

    def __call__(self, sample):
        images, labels = sample['window'], sample['labels']

        if np.random.rand()<self.frac_random:
            pattern = torch.tensor(random_pattern()).permute(2,0,1).unsqueeze(0)
            pattern = F.interpolate(pattern, (images.shape[2], images.shape[3])).squeeze(0)

            for i in range(images.shape[0]):
                img_ar = images[i]

                background_mask = (img_ar[3,:,:]==0).unsqueeze(0).repeat(3,1,1)
                img_ar[:3,:,:][background_mask] = pattern[background_mask]

        return {'window': images,
                'labels': labels}

class Normalize(object):
    """Randomly flip image sequence and labels."""

    def __init__(self, mean, std):
        self.mean=mean.view(1,3,1,1)
        self.std=std.view(1,3,1,1)
    
    def __call__(self, sample):
        images, labels = sample['window'], sample['labels']
        # images: S x H x W x C
        # labels: S x #jnts x #pos

        images = (images-self.mean) / self.std

        return {'window': images,
                'labels': labels}
    
    
class Resize(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        images, labels = sample['window'], sample['labels']
        if images.shape[1]==4:
            images = images[:,:3,:,:]
        return {'window': F.interpolate(images, (self.size, self.size)),
                'labels': labels}
