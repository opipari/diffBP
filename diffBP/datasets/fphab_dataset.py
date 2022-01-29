import os
import pickle

import numpy as np
import cv2
from skimage import io
import random

import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms


class FPHABDataset(Dataset):
    
    def __init__(self, root_dir, mode='train', window_size=15, data_max_length=20, transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.window_size = window_size
        self.data_max_length = data_max_length
        self.transform = transform
        self.h = 96
        self.w = 96
        
        self.valid_windows_per_seq = self.data_max_length - self.window_size + 1

        
        self.data_path = self.root_dir

        with open(os.path.join(self.data_path, self.mode+'.txt'),'r') as f:
          self.sequence_paths = f.read().splitlines()



    def __len__(self):
        return len(self.sequence_paths)

    def __getitem__(self, idx):
        seq_idx = idx
        window_start_idx = 0

        #chosen_path = 'F-PHAB/Video_files/Subject_4/close_milk/1/' # 
        chosen_path = self.sequence_paths[seq_idx]
        seq_len = len(os.listdir(os.path.join(self.data_path, chosen_path, 'color')))
        while seq_len<self.window_size+5:
            chosen_path = self.sequence_paths[random.randint(0,len(self.sequence_paths)-1)]
            seq_len = len(os.listdir(os.path.join(self.data_path, chosen_path, 'color')))

        if self.mode=='train':
            imgs = np.zeros((self.window_size, self.h, self.w, 3))
            depths = np.zeros((self.window_size, self.h, self.w, 1))
            shapes = np.zeros((self.window_size, 2))
            window = self.window_size
            offset = random.randint(0,seq_len-window-1)
        else:
            imgs = np.zeros((seq_len, self.h, self.w, 3))
            depths = np.zeros((seq_len, self.h, self.w, 1))
            shapes = np.zeros((seq_len, 2))
            window = seq_len
            offset = 0



        skeleton_path = os.path.join(self.data_path, chosen_path,
                                  'pose_cropped', 'skeleton_cropped.txt')
        skeleton_vals = np.loadtxt(skeleton_path)
        skel_order = skeleton_vals[:, 0]

        tls = skeleton_vals[offset:offset+window, 1:3]
        szs = skeleton_vals[offset:offset+window, 3:5]
        depth_mean = skeleton_vals[offset:offset+window, 5]

        skel = skeleton_vals[offset:offset+window, 6:].reshape(window, 21, -1)
        labels = skel
        labels[:,:,2] = (labels[:,:,2]-depth_mean.reshape(-1,1))/1000
        labels[:,:,0] = (labels[:,:,0]-48)/(96)
        labels[:,:,1] = (labels[:,:,1]-48)/(96)

        for img in range(window):
            idx = img+offset

            color_file = os.path.join(self.data_path, chosen_path,
                                  'color_cropped', 'color_'+f'{idx:04}'+'.jpeg')
            if os.path.exists(color_file):
                color = cv2.cvtColor(cv2.imread(color_file), cv2.COLOR_BGR2RGB)
                imgs[img] = color

            depth_file = os.path.join(self.data_path, chosen_path,
                                  'depth_cropped', 'depth_'+f'{idx:04}'+'.png')
            depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)-depth_mean[img]
            shapes[img] = depth.shape

            depths[img] = depth.reshape(self.h, self.w, 1)


        sample = {'label': labels, 'depth': depths, 'imgs':imgs, 'path':chosen_path, 'orig_shapes': shapes,
                'crop_top_lefts': tls, 'crop_sizes': szs, 'depth_mean': depth_mean}
    
        if self.transform:
            sample = self.transform(sample)

        return sample

    
class Normalize:
    def __init__(self):
        self._depth_mean = [-8.728770159651145]
        self._depth_std = [45.024769450434384]

        self._hand_mean = [[-212.13689469, -159.91543285,  422.42955196]]
        self._hand_std = [[35.41995314, 33.71831965, 32.08108402]]

        #self._depth_mean = [19025.14930492213]
        #self._depth_std = [9880.916071806689]

    def __call__(self, sample):
        #image, depth = sample['image'], sample['depth']
        depth = sample['depth']
        #image = image / 255

        #for i in range(image.shape[0]):
        #    image[i] = torchvision.transforms.Normalize(
        #        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image[i])

        for i in range(depth.shape[0]):
            #depth_0 = depth[i] == 0
            depth_ = torchvision.transforms.Normalize(
                mean=self._depth_mean, std=self._depth_std)(depth[i])
            # set invalid values back to zero again
            #depth_[depth_0] = 0
            depth[i] = depth_

        #sample['image'] = image
        sample['depth'] = depth

        return sample


class ToTensor:
    def __call__(self, sample):
        #image, depth = sample['image'], sample['depth']
        #image = image.transpose((0, 3, 1, 2))
        depth = sample['depth']
        depth = depth.transpose((0, 3, 1, 2)).astype('float32')

        #sample['image'] = torch.from_numpy(image).float()
        sample['depth'] = torch.from_numpy(depth).float()

        if 'label' in sample:
            label = sample['label']
            sample['label'] = torch.from_numpy(label).float()

        return sample
