import os
import pickle

import numpy as np
from skimage import io

from tqdm import tqdm

import torch
from torch.utils.data import Dataset


class ArticulatedToyDataset(Dataset):
    """Dataset class to read in pendulum images and labels."""
    
    # dataset must be organized in single folder with separate subdir for each sequence
    def __init__(self, root_dir, mode='Train', categories=[], num_seqs=None, window_size=5, data_max_length=20, transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.categories = categories
        self.window_size = window_size
        self.data_max_length = data_max_length
        self.transform = transform
        self.num_seqs = num_seqs
        # assert(self.data_fraction>0.0 and self.data_fraction<=1.0)

        self.category_sizes = []
        print(self.categories)
        for cat in self.categories:
            # int(self.data_fraction * len([fld for fld in os.listdir(os.path.join(root_dir, mode, cat)) if fld.startswith("seq")]))
            if self.num_seqs:
                num_seq_to_use = self.num_seqs//len(self.categories)
                print(num_seq_to_use)
                if cat==self.categories[0] and num_seq_to_use==0:
                    num_seq_to_use = 1
                assert num_seq_to_use < len([fld for fld in os.listdir(os.path.join(root_dir, mode, cat)) if fld.startswith("seq")])
                self.category_sizes.append(num_seq_to_use)
            else:
                self.category_sizes.append(len([fld for fld in os.listdir(os.path.join(root_dir, mode, cat)) if fld.startswith("seq")]))
        self.category_sizes = np.array(self.category_sizes).cumsum()



        self.valid_windows_per_seq = self.data_max_length - self.window_size + 1

        
        self.data_path = os.path.join(self.root_dir, self.mode)

        self.file_type = '.jpg'# if root_dir.find('transparent')==-1 else '.png'


        example_image_shape = io.imread(os.path.join(self.data_path, self.categories[0], 'seq_0', 'im_0'+self.file_type)).shape
        assert example_image_shape[0]==example_image_shape[1]
        self.hw = example_image_shape[0]
        self.data_shape = (self.window_size, self.hw, self.hw, (3 if self.file_type=='.jpg' else 4))
        

    def __len__(self):
        return self.valid_windows_per_seq * self.category_sizes[-1]

    def __getitem__(self, idx):
        seq_idx = idx // self.valid_windows_per_seq
        cat_idx = (seq_idx<self.category_sizes).nonzero()[0].min()
        if cat_idx>0:
            seq_idx -= self.category_sizes[cat_idx-1]
        window_start_idx = idx % self.valid_windows_per_seq


        seq_folder = os.path.join(self.data_path, self.categories[cat_idx], 'seq_'+str(seq_idx))

        imgs = np.zeros(self.data_shape)
        for img in range(self.window_size):
            img_file = os.path.join(seq_folder, 'im_'+str(window_start_idx+img)+self.file_type)
            imgs[img] = io.imread(img_file)
            
        # Ensure images are normalized to range [-1, +1]
        imgs = imgs / 255.0

        with open(os.path.join(seq_folder, 'truth.pkl'),'rb') as f:
            labels = pickle.load(f)['link_groundtruth'][window_start_idx:(window_start_idx+self.window_size)]
            # shape should be (20, N, 3)
            # (sequences, sequence length, N joints, x y theta coord)
        
        # Ensure labels are normalized to [-1, +1]
        labels = labels / (self.hw/2)
        
        sample = {'window': imgs, 'labels': labels}
    
        if self.transform:
            sample = self.transform(sample)

        return sample



    
def evaluate_test_bpnet(bpnet, test_categories, test_loaders, data_len=20, all_data=True):
    """Evaluation function which runs through test set to compute error."""
    # Error is broken down by w/wo occlusion and over the sequence length as mean euclidean distance
    
    bpnet = bpnet.eval()
    all_errors = {}
    with torch.no_grad():

        for i_cat in range(len(test_categories)):
            max_error_tot = [0 for _ in range(data_len)]
            max_count_tot = [0 for _ in range(data_len)]

            exp_error_tot = [0 for _ in range(data_len)]
            exp_count_tot = [0 for _ in range(data_len)]

            mle_error_tot = [0 for _ in range(data_len)]
            mle_count_tot = [0 for _ in range(data_len)]
            
            
            max_beliefs = [0 for _ in range(data_len)]
            max_beliefs_tot = [0 for _ in range(data_len)]

            if all_data:
                total=len(test_loaders[i_cat])
            else:
                total=2
            for i_batch, sample_batched in enumerate(test_loaders[i_cat]):
                if not all_data and (i_batch>total):
                    break
                # Window: B x S x C x H x W
                # Labels: B x S x 3 x 2
                batch_images = sample_batched['window'].type(bpnet.type)
                batch_labels = sample_batched['labels'].type(bpnet.type)

                # For now skip theta
                batch_labels = batch_labels[:,:,:,:2]

                bpnet.reinit_particles(batch_images.shape[0])

                bpnet.frac_resamp = 1.0

                # Iterate over sequence dimension
                for i_seq in range(batch_images.shape[1]):
                    tr = batch_labels[:,i_seq].to(device=bpnet.device)
                    x = batch_images[:,i_seq].to(device=bpnet.device)

                    # Run single belief propagation message+belief update
                    bpnet.compute_feats(x)
                    bpnet.update()
                    bpnet.update()
                    bpnet.update()

                    # pred_parts, pred_weights = bpnet.max_marginals()
                    max_parts, pred_weights = bpnet.max_marginals()
                    exp_parts = bpnet.exp_marginals()
                    mle_parts = bpnet.mle_marginals(x)
                    max_weights = bpnet.max_belief_weights()
                
                    # bpnet.update_time()

                    max_error = 0
                    for i_joint in range(bpnet.num_nodes):
                        max_error += torch.mean(torch.sqrt(((64*max_parts[i_joint]-64*tr[:,i_joint])**2).sum(dim=-1))).cpu()
                    max_error = max_error / bpnet.num_nodes
                    
                    max_error_tot[i_seq] += max_error
                    max_count_tot[i_seq] += 1

                    exp_error = 0
                    for i_joint in range(bpnet.num_nodes):
                        exp_error += torch.mean(torch.sqrt(((64*exp_parts[i_joint]-64*tr[:,i_joint])**2).sum(dim=-1))).cpu()
                    exp_error = exp_error / bpnet.num_nodes
                    
                    exp_error_tot[i_seq] += exp_error
                    exp_count_tot[i_seq] += 1

                    mle_error = 0
                    for i_joint in range(bpnet.num_nodes):
                        mle_error += torch.mean(torch.sqrt(((64*mle_parts[i_joint]-64*tr[:,i_joint])**2).sum(dim=-1))).cpu()
                    mle_error = mle_error / bpnet.num_nodes
                    
                    mle_error_tot[i_seq] += mle_error
                    mle_count_tot[i_seq] += 1
                    max_beliefs[i_seq] += sum(max_weights)/len(max_weights)
                    max_beliefs_tot[i_seq] += 1

            all_errors[test_categories[i_cat]] = (([(mle_error_tot[i]/mle_count_tot[i]).item() for i in range(len(mle_count_tot))],
                                                [(max_error_tot[i]/max_count_tot[i]).item() for i in range(len(max_count_tot))],
                                                    [(exp_error_tot[i]/exp_count_tot[i]).item() for i in range(len(exp_count_tot))],
                                [(max_beliefs[i]/max_beliefs_tot[i]).item() for i in range(len(max_beliefs_tot))]))

    return all_errors
    
def evaluate_test_lstm(lstmnet, test_categories, test_loaders, test_batch_size, window_size, num_lstm_layers, all_data=True, hidden_dim=64, device=torch.device('cpu'), data_len=20):
    """Evaluation function which runs through test set to compute error."""
    # Error is broken down by w/wo occlusion and over the sequence length as mean euclidean distance

    model = lstmnet.eval()
    all_errors = {}
    with torch.no_grad():

        for i_cat in range(len(test_categories)):
            
            error_tot = [0 for _ in range(data_len)]
            count_tot = [0 for _ in range(data_len)]
    

            
            if all_data:
                total=len(test_loaders[i_cat])
            else:
                total=(10//test_batch_size)
            for i_batch, sample_batched in tqdm(enumerate(test_loaders[i_cat]), total=total):
                if not all_data and (i_batch>total):
                    break
                # Window: B x S x C x H x W
                # Labels: B x S x 3 x 2
                batch_images = sample_batched['window'].type(torch.float32)
                batch_labels = sample_batched['labels'].type(torch.float32)
                # For now skip theta
                batch_labels = batch_labels[:,:,:,:2]

                hidden = [torch.zeros(num_lstm_layers,batch_images.shape[0],hidden_dim).type(torch.float32).to(device=device), 
                            torch.zeros(num_lstm_layers,batch_images.shape[0],hidden_dim).type(torch.float32).to(device=device)]

                for i_window in range(batch_images.shape[1]//window_size):
                    input = batch_images[:,i_window*window_size:(i_window+1)*window_size].to(device=device)
                    target = batch_labels[:,i_window*window_size:(i_window+1)*window_size].to(device=device)
                    
                    pred, hidden = model(input, hidden)

                    for i_seq in range(i_window*window_size, (i_window+1)*window_size):
                        error = torch.mean(torch.sqrt(((64*pred[:,i_seq-i_window*window_size] 
                                                            - 64*target[:,i_seq-i_window*window_size])**2).sum(dim=-1))).cpu()
                        
                        error_tot[i_seq] += error
                        count_tot[i_seq] += 1
                    
            all_errors[test_categories[i_cat]] = ([(error_tot[i]/count_tot[i]).item() for i in range(len(error_tot))], ) 

    return all_errors

def evaluate_test_hand(bpnet, test_categories, test_loaders, std, data_len=20):
    """Evaluation function which runs through test set to compute error."""
    # Error is broken down by w/wo occlusion and over the sequence length as mean euclidean distance
    
    bpnet = bpnet.eval()

    all_errors = {}
    with torch.no_grad():

        for i_cat in range(len(test_categories)):
            max_error_tot = [0 for _ in range(data_len)]
            max_count_tot = [0 for _ in range(data_len)]

            exp_error_tot = [0 for _ in range(data_len)]
            exp_count_tot = [0 for _ in range(data_len)]

            mle_error_tot = [0 for _ in range(data_len)]
            mle_count_tot = [0 for _ in range(data_len)]
            
            
            max_beliefs = [0 for _ in range(data_len)]
            max_beliefs_tot = [0 for _ in range(data_len)]

            for i_batch, sample_batched in enumerate(test_loaders[i_cat]):
                # Window: B x S x C x H x W
                # Labels: B x S x 3 x 2
                batch_images = sample_batched['window'].type(torch.double)
                batch_labels = sample_batched['labels'].type(torch.double)

                # For now skip theta
                batch_labels = batch_labels[:,:,:,:2]

                bpnet.reinit_particles(batch_images.shape[0])

                # Iterate over sequence dimension
                for i_seq in range(batch_images.shape[1]):
                    tr = batch_labels[:,i_seq].to(device=bpnet.device)
                    x = batch_images[:,i_seq].to(device=bpnet.device)

                    # Run single belief propagation message+belief update
                    bpnet.update(x, std, tr)
                    bpnet.update_time()

                    # pred_parts, pred_weights = bpnet.max_marginals()
                    max_parts, pred_weights = bpnet.max_marginals()
                    exp_parts = bpnet.exp_marginals()
                    mle_parts = bpnet.mle_marginals(x, std, tr)
                    max_weights = bpnet.max_belief_weights()
                
                    bpnet.update_time()

                    max_error = 0
                    for i_joint in range(bpnet.num_nodes):
                        max_error += torch.mean(torch.sqrt(((64*max_parts[i_joint]-64*tr[:,i_joint])**2).sum(dim=-1))).cpu()
                    max_error = max_error / bpnet.num_nodes
                    
                    max_error_tot[i_seq] += max_error
                    max_count_tot[i_seq] += 1

                    exp_error = 0
                    for i_joint in range(bpnet.num_nodes):
                        exp_error += torch.mean(torch.sqrt(((64*exp_parts[i_joint]-64*tr[:,i_joint])**2).sum(dim=-1))).cpu()
                    exp_error = exp_error / bpnet.num_nodes
                    
                    exp_error_tot[i_seq] += exp_error
                    exp_count_tot[i_seq] += 1

                    mle_error = 0
                    for i_joint in range(bpnet.num_nodes):
                        mle_error += torch.mean(torch.sqrt(((64*mle_parts[i_joint]-64*tr[:,i_joint])**2).sum(dim=-1))).cpu()
                    mle_error = mle_error / bpnet.num_nodes
                    
                    mle_error_tot[i_seq] += mle_error
                    mle_count_tot[i_seq] += 1
                    max_beliefs[i_seq] += sum(max_weights)/len(max_weights)
                    max_beliefs_tot[i_seq] += 1

            all_errors[test_categories[i_cat]] = (([(mle_error_tot[i]/mle_count_tot[i]).item() for i in range(len(mle_count_tot))],
                                                [(max_error_tot[i]/max_count_tot[i]).item() for i in range(len(max_count_tot))],
                                                    [(exp_error_tot[i]/exp_count_tot[i]).item() for i in range(len(exp_count_tot))],
                                [(max_beliefs[i]/max_beliefs_tot[i]).item() for i in range(len(max_beliefs_tot))]))

    return all_errors