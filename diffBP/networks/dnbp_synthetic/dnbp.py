import numpy as np
import torch
import torch.nn as nn

from diffBP.networks.dnbp_synthetic import factors


class DNBP(nn.Module):
    def __init__(self, graph, edge_set, inc_nghbrs, shared_feats=True, enc_hidden_feats_tot=32, enc_output_feats_tot=64, multi_edge_samplers=True, mode='low_var', batch_size=3, particle_count=100, 
                 particle_size=2, est_bounds=1, std=0.1, density_std=0.01, lambd=0.8, device=torch.device('cpu'), precision='float32'):
        super(DNBP, self).__init__()
        
        # graph should be formatted in sorted COO format
        self.graph = graph
        # edge_set should index each edge single time on first occurence in graph
        self.edge_set = edge_set
        self.inc_nghbrs = inc_nghbrs
        self.mode = mode
        
        self.num_edges = self.graph.shape[1]//2
        self.num_nodes = int(self.graph.max())+1
        
        self.batch_size = batch_size
        self.particle_count = particle_count
        self.particle_size = particle_size
        
        self.est_bounds = est_bounds
        
        self.std = std
        self.density_std = density_std
        self.device = device
        
        
        # Particle Filter Resampling [0,1]
        self.lambd = lambd
        self.time_step = 0
        self.frac_resamp = 0.
        
        # Use learned time sampling factor (True) or Gaussian diffusion (False)
        self.use_time = True
        
        self.shared_feats = shared_feats
        self.multi_edge_samplers = multi_edge_samplers

        if precision=="float32":
          self.type = torch.float32
        else:
          self.type = torch.double
        #
        # FACTOR DEFININTIONS
        #
        # feature extractor network shared across nodes
        if self.shared_feats:
          self.likelihood_features = factors.LikelihoodFeatures(hidden_features=enc_hidden_feats_tot, output_feats=enc_output_feats_tot).type(self.type).to(device=self.device)
          # likelihood factors to measure unary potential for each node
          self.node_likelihoods = nn.ModuleList([factors.UnaryDensity(in_features=enc_output_feats_tot, particle_size=particle_size).to(device=self.device) 
                                                 for _ in range(self.num_nodes)]).type(self.type)
        else:
          self.likelihood_features = nn.ModuleList([factors.LikelihoodFeatures(hidden_features=(enc_hidden_feats_tot//self.num_nodes), output_feats=enc_output_feats_tot//self.num_nodes).to(device=self.device)
                                                 for _ in range(self.num_nodes)]).type(self.type)
        
          # likelihood factors to measure unary potential for each node
          self.node_likelihoods = nn.ModuleList([factors.UnaryDensity(in_features=enc_output_feats_tot//self.num_nodes, particle_size=particle_size).to(device=self.device) 
                                                 for _ in range(self.num_nodes)]).type(self.type)
        
        if self.multi_edge_samplers:
          # edge sampling factors to sample from each conditional neighbor relationship
          self.edge_samplers = nn.ModuleList([nn.ModuleList([factors.PairwiseSampler(particle_size=particle_size, 
                                                                                     est_bounds=est_bounds,
                                                                                    device=self.device,
                                                                                    precision=self.type).to(device=self.device) 
                                                             for i in range(2)]) for _ in range(self.num_edges)]).type(self.type)
        else:
          self.edge_samplers = nn.ModuleList([factors.PairwiseSampler(particle_size=particle_size, 
                                                                                     est_bounds=est_bounds,
                                                                                    device=self.device,
                                                                                    precision=self.type).to(device=self.device) 
                                                             for _ in range(self.num_edges)]).type(self.type)

        # edge density factors to compute compatibility of neighboring particles
        self.edge_densities = nn.ModuleList([factors.PairwiseDensity(particle_size=particle_size).to(device=self.device) 
                                             for _ in range(self.num_edges)]).type(self.type)
        
        # time factors to propagate particles through time
        self.time_samplers = nn.ModuleList([factors.PairwiseSampler(particle_size=particle_size, 
                                                                    est_bounds=est_bounds,
                                                                   device=self.device,
                                                                   precision=self.type).to(device=self.device) 
                                            for _ in range(self.num_nodes)]).type(self.type)
        #
        # END OF FACTOR DEFINITIONS
        #
        
        
        
        
        
        #
        # PARTICLE DEFINITIONS
        #
        # initialize belief and message particles
        # use a list for each because the graph may lead to variable sized neighbor dimensions
        self.reinit_particles(self.batch_size)
        #
        # END OF PARTICLE DEFINITIONS
        #
        
        

        
        
        
    # Helper function used in forward pass to avoid affecting likelihood gradients
    # Used only where training the pairwise sampling networks to ensure no 'interference' during training
    # likelihood factors' training should depend only on the corresponding node, not its neighbors 
    def turn_off_lik_grads(self):
        for p in self.node_likelihoods.parameters():
            p.requires_grad = False
    
    # Helper function used in forward pass to undo the effect of turn_off_lik_grads
    # Ensures likelihood factors have gradients updated in backward pass
    def turn_on_lik_grads(self):
        for p in self.node_likelihoods.parameters():
            p.requires_grad = True
            
            

            
            
    # Reset particles to uniform
    # Particle positions sampled from uniform [-est_bounds, est_bounds]
    # Particle weights set to uniform 1/num_particles
    def reinit_particles(self, size):
        self.batch_size = size
        self.time_step = 0
        self.frac_resamp = 0.0

        self.belief_particles = [torch.empty(self.batch_size, len(self.inc_nghbrs[_]), self.particle_count, 
                                             self.particle_size).to(device=self.device).uniform_(-self.est_bounds, 
                                                                                                 self.est_bounds).type(self.type)
                                     for _ in range(self.num_nodes)]
        self.belief_weights = [torch.ones(self.batch_size, len(self.inc_nghbrs[_]), 
                                          self.particle_count).to(device=self.device).type(self.type)
                               / (len(self.inc_nghbrs[_]) * self.particle_count) 
                                   for _ in range(self.num_nodes)]
        
        self.message_particles = [torch.empty(self.batch_size, len(self.inc_nghbrs[_]), self.particle_count, 
                                              self.particle_size).to(device=self.device).uniform_(-self.est_bounds, 
                                                                                                 self.est_bounds).type(self.type)
                                      for _ in range(self.num_nodes)]
        self.message_weights = [torch.ones(self.batch_size, len(self.inc_nghbrs[_]), 
                                           self.particle_count).to(device=self.device).type(self.type)
                                / (len(self.inc_nghbrs[_]) * self.particle_count)
                                    for _ in range(self.num_nodes)]
        
        self.message_weights_unary = [torch.ones(self.batch_size, len(self.inc_nghbrs[i]), 
                                                 self.particle_count).to(device=self.device).type(self.type)
                                      / (len(self.inc_nghbrs[i]) * self.particle_count)
                                      for i in range(self.num_nodes)]
        self.message_weights_neigh = [torch.ones(self.batch_size, len(self.inc_nghbrs[i]), 
                                                 self.particle_count).to(device=self.device).type(self.type)
                                      / (len(self.inc_nghbrs[i]) * self.particle_count)
                                      for i in range(self.num_nodes)]
        self.belief_weights_lik = [torch.ones(self.batch_size, len(self.inc_nghbrs[i]), 
                                                 self.particle_count).to(device=self.device).type(self.type)
                                      / (len(self.inc_nghbrs[i]) * self.particle_count)
                                      for i in range(self.num_nodes)]

    def update_time(self):
        self.time_step += 1
        self.frac_resamp = 1-(self.lambd ** self.time_step)
#         # reset belief_particle locations to uniform [-est_bounds, est_bounds]
#         self.belief_particles = [bel_p.uniform_(-self.est_bounds, self.est_bounds) for bel_p in self.belief_particles]
#         # reset belief_weights to uniform (normalized)
#         for bel_w in self.belief_weights:
#             bel_w[:,:,:] = 1.0 / (bel_w.shape[1] * bel_w.shape[2])
        

#         self.message_particles = [msg_p.uniform_(-self.est_bounds, self.est_bounds) for msg_p in self.message_particles]
#         for msg_w in self.message_weights:
#             msg_w[:,:,:] = 1.0 / (msg_w.shape[1] * msg_w.shape[2])
        
#         for wgt_u in self.message_weights_unary:
#             wgt_u[:,:,:] = 1.0 / (wgt_u.shape[1] * wgt_u.shape[2])
        
#         for wgt_n in self.message_weights_neigh:
#             wgt_n[:,:,:] = 1.0 / (wgt_n.shape[1] * wgt_n.shape[2])
            
#         for wgt_l in self.belief_weights_lik:
#             wgt_l[:,:,:] = 1.0 / (wgt_l.shape[1] * wgt_l.shape[2])
            
    
    
    
    # Perform message update step (resample, then calculate w_unary and w_neigh for all particles)
    def message_update(self, glbl_feats, node_id=None, tru=None):
        # Be sure that tru data is only past during training to avoid invalid results
        assert((self.training and (tru is not None)) or (not self.training and (tru is None)))
        
        # Global features needed for likelihood computation
        # detach from computation graph to ensure no gradients go into feature extractor
        # reasoning is we want to avoid letting the sampler interfere w/ likelihood training
        if self.shared_feats:
          glbl_feats = glbl_feats.detach()
        else:
          glbl_feats = [feats.detach() for feats in glbl_feats]#.detach()
        # turn off likelihood gradients to ensure the node_likelihoods don't update gradients during message update
        # be sure to turn back on the gradients for likelihood training
        # this should work by ensuring no forward pass registers for likelihood networks, but 
        # backward can still propagate past likelihood networks and into samplers
        self.turn_off_lik_grads()
        
        
        # Store detached copy of initial messages for w_neigh calculation
        old_messages = [old.clone() for old in self.message_particles]
        for old in old_messages:
            old.requires_grad = False
        old_message_weights = [old.clone() for old in self.message_weights]
        for old in old_message_weights:
            old.requires_grad = False


        if node_id is None:
          iters = range(self.num_nodes)
        else:
          iters = [node_id]
            
        # Iterate all destination nodes for message pass
        for dst_ in iters:
            # Start the estimation process with non-differentiable resampling
            # All resampling is done batch-wise
            # This is accomplished by using additional memory inplace of additional computation
            # For example, we duplicate the cumulative sum of random chance for every particle that shares the sum
            # This could certainly be made more efficient (e.g. custom resample kernel to share memory on hardware)
            resamp_particles = int(self.frac_resamp * self.particle_count)
            if resamp_particles>0:
                # Explicitly detach particles that are being sampled from
                # batch x pseudo_bel x particle_size
                dst_belief_particles = self.belief_particles[dst_].view(self.batch_size, -1, self.particle_size).detach()
                dst_belief_weights = self.belief_weights[dst_].view(self.batch_size, -1).detach()


                if self.mode=='low_var':
                    rand_chance = ((torch.arange(resamp_particles) / float(resamp_particles)).repeat(self.batch_size, 
                                                                                              len(self.inc_nghbrs[dst_]), 1) 
                                   + (torch.rand(self.batch_size, len(self.inc_nghbrs[dst_]), 1) / float(resamp_particles)))
                else:
                    rand_chance = torch.rand(self.batch_size, len(self.inc_nghbrs[dst_]), resamp_particles)
                # batch x incoming x 1 x resamp_particles
                rand_chance = rand_chance.unsqueeze(2).type(self.type).to(device=self.device)

                # batch x incoming x pseudo_bel x resamp_particles
                cum_sum = (dst_belief_weights).cumsum(1).unsqueeze(1).unsqueeze(3).repeat((1, len(self.inc_nghbrs[dst_]), 
                                                                                           1, resamp_particles))

                # Due to the fact that pytorch argmin/argmax does not guarantee a tiebreak, we currently use
                # this inverse-argmax hack which ensures the difference closest to zero(positive side) is selected
                # ISSUE: Theres a small chance that the denominator is zero and results in NaN. Hasn't been observed thus far.
                # batch x incoming x pseudo_bel x resamp_particles
                rand_ind = torch.argmax(1 / (cum_sum - rand_chance), dim=2)

                # batch x incoming x resamp_particles
                # Duplicate random indices for each of the particle_size dimensions in order to use torch.gather
                rand_ind = rand_ind.unsqueeze(-1).repeat(1, 1, 1, self.particle_size)
                # batch x incoming x resamp_particles x particle_size

                # batch x incoming x pseudo_bel x particle_size
                dst_belief_particles = dst_belief_particles.unsqueeze(1).repeat(1, len(self.inc_nghbrs[dst_]), 1, 1)

                # batch x incoming x particles x particle_size
                resampled_particles = torch.gather(dst_belief_particles, 2, rand_ind)

                if self.use_time:
                    time_delta = self.time_samplers[dst_](self.batch_size * len(self.inc_nghbrs[dst_]) 
                                                          * resamp_particles).view(self.batch_size, len(self.inc_nghbrs[dst_]), 
                                                                                      resamp_particles, -1)
                else:
                    time_delta = torch.randn((self.batch_size, len(self.inc_nghbrs[dst_]), resamp_particles, 
                                              self.particle_size)).to(device=self.device) * self.std
                self.message_particles[dst_][:,:,:resamp_particles] = torch.clamp(resampled_particles + time_delta, 
                                                           min=-self.est_bounds, max=self.est_bounds)
                
                # For remaining particles sample from uniform [-est_bounds, est_bounds]
                self.message_particles[dst_][:,:,resamp_particles:] = torch.empty(self.batch_size, len(self.inc_nghbrs[dst_]), 
                                                                                  self.particle_count-resamp_particles, 
                                                                                  self.particle_size).uniform_(-self.est_bounds, 
                                                                                         self.est_bounds).type(self.type).to(device=self.device)
            else:
                self.message_particles[dst_] = torch.empty(self.batch_size, len(self.inc_nghbrs[dst_]), self.particle_count, 
                                                           self.particle_size).uniform_(-self.est_bounds, 
                                                                                        self.est_bounds).type(self.type).to(device=self.device)
                
            # After resampling, particle weights must be set to uniform
            # This step breaks differentiability to past weights
            self.message_weights[dst_] = (torch.ones(self.batch_size, len(self.inc_nghbrs[dst_]), 
                                                    self.particle_count).type(self.type).to(device=self.device) 
                                          / (len(self.inc_nghbrs[dst_]) * self.particle_count))
            
            
            
            
            # To calculate, modify standard PMPNBP by using multiple samples to smooth performance
            # Multiple samples decreases weight deviation for particles (good particles more consistently have large weight)
            num_int = 10
            
            for src_i, src_ in enumerate(self.inc_nghbrs[dst_]):
                # Determine edge index of message pass from src_->dst_
                edge_i = ((self.graph == torch.tensor([[min(src_,dst_)],
                                                       [max(src_,dst_)]])).all(dim=0).nonzero().squeeze(0) 
                          == self.edge_set).nonzero().squeeze(0)
                
                # Isolate outgoing particles from src_->dst_ for w_unary calculation
                # These outgoing particles are in the frame of dst_
                msgs = self.message_particles[dst_][:,src_i].contiguous().view(-1,1,self.particle_size)
                
                # Generate delta samples to translate particles from dst_ frame to src_ frame
                if self.multi_edge_samplers:
                  if src_ < dst_:
                      s = self.edge_samplers[edge_i][0](msgs.shape[0]*num_int)
                  else:
                      s = self.edge_samplers[edge_i][1](msgs.shape[0]*num_int)
                else:
                  if src_ < dst_:
                      s = self.edge_samplers[edge_i](msgs.shape[0]*num_int)
                  else:
                      s = -self.edge_samplers[edge_i](msgs.shape[0]*num_int)

                # Translate particles into src_ frame using sampled deltas
                samples = msgs + s.view(msgs.shape[0],num_int,-1)
                # Change view for feeding into network
                samples = samples.view(msgs.shape[0]*num_int,-1)

                # Concatenate features with particles and feed into likelihood network of src_
                # Remember, samples are now in src_'s frame
                # This is the key step that causes us to turn off likelihood gradients; we don't want the src_ likelihood
                # factor learning to weight particles based on feedback from dst_ ground truth
                if self.shared_feats:
                  wgts = self.node_likelihoods[src_](torch.cat((samples, 
                                                                glbl_feats.unsqueeze(1).repeat((1, self.particle_count * num_int, 
                                                                                                1)).view(self.batch_size 
                                                                                                         * self.particle_count 
                                                                                                         * num_int, -1)),
                                                               dim=1)).view(self.batch_size, self.particle_count, num_int)
                else:
                  wgts = self.node_likelihoods[src_](torch.cat((samples, 
                                                                glbl_feats[src_].unsqueeze(1).repeat((1, self.particle_count * num_int, 
                                                                                                1)).view(self.batch_size 
                                                                                                         * self.particle_count 
                                                                                                         * num_int, -1)),
                                                               dim=1)).view(self.batch_size, self.particle_count, num_int)
                # Average over the num_int samples
                wgts = wgts.mean(dim=-1)
                # Normalize scores of outgoing particles, these are the w_unary scores
                wgts = wgts / wgts.sum(dim=1,keepdim=True)
                # Store unary scores for use later
                self.message_weights_unary[dst_][:,src_i] = wgts
                
                
                 
               
                
                
                ign_indx = (torch.tensor(self.inc_nghbrs[src_])==dst_).nonzero().squeeze()
                
                # relevant neighbor message shape: batch x Num_Neighbors x particle_count x particle_size
                relv = torch.cat([old_messages[src_][:,:ign_indx], 
                           old_messages[src_][:,(ign_indx+1):]], dim=1)
                # print(src_,'->',dst_,relv.shape)
                if relv.shape[1]>0:
                    if self.training:
                        relv_weights = ((1/(self.density_std*np.sqrt(2*np.pi))) 
                                        * torch.exp((-1/2) * (((relv-tru[:,src_,:].unsqueeze(1).unsqueeze(1)) 
                                                               / self.density_std)**2).sum(dim=-1)))
                    else:
                        relv_weights = torch.cat([old_message_weights[src_][:,:ign_indx], 
                                   old_message_weights[src_][:,(ign_indx+1):]], dim=1)

                    # if src_>0 and src_<4:
                    #   print(src_, tru[0,src_,:])

                    # Perform weighting in delta space (independent of postion)
                    # Ensure delta direction is same as w_unary to make plotting meaningful
                    # src_ frame (relv) = dst_ frame (msg_p) + delta (diff)
                    # delta (diff) = src_ frame (relv) - dst_ frame (msg_p)


                    # Graph:
                    # 1->2->3


                    # Where we are in the loop: Calculate the weights of the msgs from 2 to 3
                    # outgoing messages: msgs(2->3) i:M
                    # w_neigh(i) = \prod neighbors [(pairwise(msgs(1->2) - msgs(2->3){i})


                    # Where we are in the loop: Calculate the weights of the msgs from 1 to 2
                    # outgoing messages: msgs(1->2) i:M
                    # w_neigh(i) = sum_{neighbors of node 1 excluding node 2(destinate)} 

                    if self.training:
                      if src_<dst_:
                        diff = tru[:,src_,:].unsqueeze(1).unsqueeze(1).unsqueeze(1) - self.message_particles[dst_][:,src_i].unsqueeze(2).unsqueeze(2)
                      else:
                        diff = self.message_particles[dst_][:,src_i].unsqueeze(2).unsqueeze(2) - tru[:,src_,:].unsqueeze(1).unsqueeze(1).unsqueeze(1)
                      relv_weights = 1
                    else:
                      if src_<dst_:
                        diff = relv.unsqueeze(1) - self.message_particles[dst_][:,src_i].unsqueeze(2).unsqueeze(2)
                      else:
                        diff = self.message_particles[dst_][:,src_i].unsqueeze(2).unsqueeze(2) - relv.unsqueeze(1)
                      relv_weights = relv_weights.unsqueeze(1)

                    # print(relv.unsqueeze(1).shape, self.message_particles[dst_][:,src_i].unsqueeze(2).unsqueeze(2).shape)
                    # print(diff.shape)
                    num_p_dst = diff.shape[1]
                    num_n = diff.shape[2]
                    num_p_src = diff.shape[3]
                    # reshape delta for feeding into network
                    diff = diff.view(self.batch_size*num_p_dst*num_n*num_p_src,self.particle_size)
                    dens = self.edge_densities[edge_i](diff)
                    dens = dens.view(self.batch_size, num_p_dst, num_n, num_p_src)
                    # print(dens.shape,relv_weights.unsqueeze(1).shape)
                    # Scale weights by neighbor scores
                    dens = dens * relv_weights

                    # In future, replace squeeze with product operation. Needed for graphs with more neighbors
                    dens = dens.sum(dim=-1)
                    # print(dens.shape)
                    dens = torch.prod(dens, dim=2)
                    dens = dens / dens.sum(dim=1, keepdim=True)
                    # print(dens.shape)
                    # dens = relv_weights = ((1/(self.density_std*np.sqrt(2*np.pi))) 
                    #                     * torch.exp((-1/2) * (((self.message_particles[dst_][:,src_i].unsqueeze(1)-tru[:,src_,:].unsqueeze(1).unsqueeze(1)) 
                    #                                            / self.density_std)**2).sum(dim=-1))).squeeze()
                else:
                    dens = torch.ones_like(self.message_weights_neigh[dst_][:,src_i]) / self.particle_count
                # print(dens.shape, relv.shape, self.message_particles[dst_][:,src_i].unsqueeze(2).unsqueeze(2).shape, tru[:,src_,:].unsqueeze(1).unsqueeze(1).shape)
                
                self.message_weights_neigh[dst_][:,src_i] = dens
            
        return
        
        
    # Belief update step, combine all incoming messages to form belief
    def belief_update(self, glbl_feats, node_id=None):
        
        # Turn gradients back on for training the unary potentials
        self.turn_on_lik_grads()
        
        if node_id is None:
          iters = range(self.num_nodes)
        else:
          iters = [node_id]
        # Iterate over each node in graph, update message weights to form belief set
        for _ in iters:

            # Calculate the destination node unary scores

            if self.shared_feats:
              message_liks = self.node_likelihoods[_](torch.cat((self.message_particles[_].view(-1, self.particle_size), 
                                                               glbl_feats.unsqueeze(1).repeat((1, len(self.inc_nghbrs[_]) 
                                                                                               * self.particle_count, 
                                                                                               1)).view(self.batch_size
                                                                                                        *len(self.inc_nghbrs[_])
                                                                                                        *self.particle_count,
                                                                                                        -1)), 
                                                              dim=1))
            else:
              message_liks = self.node_likelihoods[_](torch.cat((self.message_particles[_].view(-1, self.particle_size), 
                                                                 glbl_feats[_].unsqueeze(1).repeat((1, len(self.inc_nghbrs[_]) 
                                                                                                 * self.particle_count, 
                                                                                                 1)).view(self.batch_size
                                                                                                          *len(self.inc_nghbrs[_])
                                                                                                          *self.particle_count,
                                                                                                          -1)), 
                                                                dim=1))
            message_liks = message_liks.squeeze(1).view(self.batch_size, len(self.inc_nghbrs[_]), self.particle_count)
            message_liks = message_liks / message_liks.sum(dim=2,keepdim=True)
            self.belief_weights_lik[_][:,:,:] = message_liks
            
            


            incoming_reweights = self.belief_weights_lik[_] * self.message_weights_unary[_] * self.message_weights_neigh[_]
            incoming_reweights = incoming_reweights / incoming_reweights.sum(2, keepdim=True)
            incoming_reweights = incoming_reweights / len(self.inc_nghbrs[_])
            


            self.belief_weights[_] = incoming_reweights
            self.belief_particles[_] = self.message_particles[_]
    
        return


    def compute_feats(self, x):
      if self.shared_feats:
          self.global_features = self.likelihood_features(x)
      else:
          self.global_features = [extractor(x) for extractor in self.likelihood_features]   
           
        
    
    # Evaluated the likelihood factors for given data
    def likelihood(self, node_id, x, grid):
        if self.shared_feats:
          global_features = self.likelihood_features(x)
        else:
          global_features = self.likelihood_features[node_id](x)
        num_grid = grid.view(-1, self.particle_size).shape[0]
    
        gd = (grid.view(-1, self.particle_size))
        gf = global_features.repeat((num_grid,1))
        g = torch.cat((gd, gf), dim=1)

        message_liks = self.node_likelihoods[node_id](g)
        return message_liks
    
    
    # Perform the full forward pass
    def update(self, node_id=None, tru=None):
        for dst_ in range(self.num_nodes):
            self.belief_particles[dst_] = self.belief_particles[dst_].detach()
            self.belief_weights[dst_] = self.belief_weights[dst_].detach()
            self.message_particles[dst_] = self.message_particles[dst_].detach()
            self.message_weights[dst_] = self.message_weights[dst_].detach()
            self.belief_weights_lik[dst_] = self.belief_weights_lik[dst_].detach()
            self.message_weights_unary[dst_] = self.message_weights_unary[dst_].detach()
            self.message_weights_neigh[dst_] = self.message_weights_neigh[dst_].detach()
        
        # if self.shared_feats:
        #     global_features = self.likelihood_features(x)
        # else:
        #     if node_id is None:
        #       global_features = [extractor(x) for extractor in self.likelihood_features]   
        #     else:
        #       global_features = self.likelihood_features[node_id](x)
          
        self.message_update(self.global_features, node_id=node_id, tru=tru)
        self.belief_update(self.global_features, node_id=node_id)
        

        return
    
    def max_marginals(self):
        pred_particles = []
        pred_weights = []
        for node_id in range(self.num_nodes):
            ind = torch.max(self.belief_weights[node_id].view(self.batch_size, -1), dim=1)
            preds = torch.gather(self.belief_particles[node_id].view(self.batch_size, -1, 2), 1, ind[1].long().unsqueeze(1).unsqueeze(1).repeat(1,1,2)).squeeze(1)
            pred_particles.append(preds)
            pred_weights.append(ind[0])
        return pred_particles, pred_weights


    def exp_marginals(self):
        pred_particles = []
        for node_id in range(self.num_nodes):
            preds = torch.sum(self.belief_weights[node_id].view(self.batch_size, -1, 1) 
              * self.belief_particles[node_id].view(self.batch_size, -1, 2), dim=1)
            pred_particles.append(preds)

        return pred_particles

    def mle_marginals(self, x):

        if self.shared_feats:
          global_features = self.likelihood_features(x)
        else:
          global_features = [extractor(x) for extractor in self.likelihood_features]#self.likelihood_features(x)

        pred_particles = []
        for node_id in range(self.num_nodes):
          parts = self.belief_particles[node_id].view(self.batch_size, -1, self.particle_size)
          num_parts = parts.shape[1]
          if self.shared_feats:
            gf = global_features.unsqueeze(1).repeat((1,num_parts,1))
          else:
            gf = global_features[node_id].unsqueeze(1).repeat((1,num_parts,1))
          g = torch.cat((parts, gf), dim=2).view(self.batch_size*num_parts, -1)
          parts_liks = self.node_likelihoods[node_id](g).view(self.batch_size, num_parts)
          ind = torch.max(parts_liks, dim=1)
          preds = torch.gather(parts, 1, ind[1].long().unsqueeze(1).unsqueeze(1).repeat(1,1,2)).squeeze(1)
          pred_particles.append(preds)
        return pred_particles
    
    
    def max_belief_weights(self):
        weight_max = []
        for node_id in [2]:
            maxx = torch.mean(torch.max(self.belief_weights[node_id].view(self.batch_size, -1), dim=1)[0])
            weight_max.append(maxx)
        return weight_max


    # def ancestral_sample_joint(self):
    #   samp_particles = 1000


    #   i_joint = 0
    #   dst_belief_particles = bpn.belief_particles[i_joint].view(bpn.batch_size, -1, bpn.particle_size).detach()
    #   dst_belief_weights = bpn.belief_weights[i_joint].view(bpn.batch_size, -1).detach()
    #   sampled_particles = discrete_samples(dst_belief_particles, dst_belief_weights, samp_particles, i_joint, bpn.batch_size, len(bpn.inc_nghbrs[i_joint])).view(bpn.batch_size, -1, bpn.particle_size)
    #   base_particles = sampled_particles.unsqueeze(2)

    #   i_joint = 1
    #   dst_belief_particles = bpn.belief_particles[i_joint].view(bpn.batch_size, -1, bpn.particle_size).detach()
    #   dst_belief_weights = bpn.belief_weights[i_joint].view(bpn.batch_size, -1).detach()
    #   nghbr_mid_particles = discrete_samples(dst_belief_particles, dst_belief_weights, int(bpn.particle_count/len(bpn.inc_nghbrs[i_joint])), i_joint, bpn.batch_size, len(bpn.inc_nghbrs[i_joint])).view(bpn.batch_size, -1, bpn.particle_size)
    #   nghbr_mid_particles = nghbr_mid_particles.unsqueeze(1)

    #   base_mid_delta = nghbr_mid_particles - base_particles


    #   print(base_particles.shape, nghbr_mid_particles.shape, base_mid_delta.shape)
    #   mid_cond_wgts = bpn.edge_densities[0](base_mid_delta.view(-1, bpn.particle_size)).view(bpn.batch_size, samp_particles, base_mid_delta.shape[2])
    #   mid_cond_wgts = mid_cond_wgts.view(-1, mid_cond_wgts.shape[2])
    #   mid_cond_wgts = mid_cond_wgts / mid_cond_wgts.sum(dim=1, keepdim=True)
    #   nghbr_mid_particles = nghbr_mid_particles.view(-1, nghbr_mid_particles.shape[2], bpn.particle_size).repeat(1,samp_particles,1,1).view(-1, nghbr_mid_particles.shape[2], bpn.particle_size)
    #   mid_particles = discrete_samples(nghbr_mid_particles, mid_cond_wgts, 1, i_joint, nghbr_mid_particles.shape[0], 1).view(bpn.batch_size, samp_particles, bpn.particle_size)
    #   mid_particles = mid_particles.unsqueeze(2)




    #   i_joint = 2
    #   dst_belief_particles = bpn.belief_particles[i_joint].view(bpn.batch_size, -1, bpn.particle_size).detach()
    #   dst_belief_weights = bpn.belief_weights[i_joint].view(bpn.batch_size, -1).detach()
    #   nghbr_end_particles = discrete_samples(dst_belief_particles, dst_belief_weights, int(bpn.particle_count/len(bpn.inc_nghbrs[i_joint])), i_joint, bpn.batch_size, len(bpn.inc_nghbrs[i_joint])).view(bpn.batch_size, -1, bpn.particle_size)
    #   nghbr_end_particles = nghbr_end_particles.unsqueeze(1)

    #   end_mid_delta =  mid_particles - nghbr_end_particles

    #   print(end_mid_delta.shape, mid_particles.shape, nghbr_end_particles.shape)

    #   end_cond_wgts = bpn.edge_densities[1](end_mid_delta.view(-1, bpn.particle_size)).view(bpn.batch_size, samp_particles, end_mid_delta.shape[2])
    #   end_cond_wgts = end_cond_wgts.view(-1, end_cond_wgts.shape[2])
    #   end_cond_wgts = end_cond_wgts / end_cond_wgts.sum(dim=1, keepdim=True)
    #   nghbr_end_particles = nghbr_end_particles.view(-1, nghbr_end_particles.shape[2], bpn.particle_size).repeat(1,samp_particles,1,1).view(-1, nghbr_end_particles.shape[2], bpn.particle_size)
    #   end_particles = discrete_samples(nghbr_end_particles, end_cond_wgts, 1, i_joint, nghbr_end_particles.shape[0], 1).view(bpn.batch_size, samp_particles, bpn.particle_size)
    #   end_particles = end_particles.unsqueeze(2)

    
    # Generate density estimate with weighted gaussian kernels
    def density_estimation(self, node_id, x, mode='belief'):
        belief_particles = self.belief_particles[node_id].view(self.batch_size, 1, -1, self.particle_size).double()
        
        if mode=='belief':
            weights = self.belief_weights[node_id].view(self.batch_size, 1, -1).double()
        elif mode=='w_lik':
            weights = self.belief_weights_lik[node_id].view(self.batch_size, 1, -1).double()
        elif mode=='w_unary':
            weights = self.message_weights_unary[node_id].view(self.batch_size, 1, -1).double()
        elif mode=='w_neigh':
            weights = self.message_weights_neigh[node_id].view(self.batch_size, 1, -1).double()
        else:
            raise 
           

        x = x.double()
        diffsq = (((x-belief_particles)/self.std)**2).sum(dim=-1)
        exp_val = torch.exp((-1/2) * diffsq)
        fact = 1/(self.std*np.sqrt(2*np.pi))
        fact_ = fact * exp_val
        out = (weights * fact_).sum(dim=-1)

        return out

    # particles: Batch x NumParticles x ParticleSize
    # weights: Batch x NumParticles
    def discrete_samples(self, particles, weights, num_samples):
        batch_size = particles.shape[0]
        particle_size = particles.shape[2]
        
        rand_chance = ((torch.arange(num_samples) / float(num_samples)).repeat(batch_size, 1) 
                      + (torch.rand(batch_size, 1) / float(num_samples)))

        # batch x 1 x resamp_particles
        rand_chance = rand_chance.unsqueeze(1).type(self.type).to(device=self.device)

        # batch x (incoming x pseudo_bel) x resamp_particles
        cum_sum = (weights).cumsum(1).unsqueeze(2).repeat((1, 1, num_samples))

        # Due to the fact that pytorch argmin/argmax does not guarantee a tiebreak, we currently use
        # this inverse-argmax hack which ensures the difference closest to zero(positive side) is selected
        # ISSUE: Theres a small chance that the denominator is zero and results in NaN. Hasn't been observed thus far.
        # batch x (incoming x pseudo_bel) x resamp_particles
        rand_ind = torch.argmax(1 / (cum_sum - rand_chance), dim=1)

        # batch x resamp_particles
        # Duplicate random indices for each of the particle_size dimensions in order to use torch.gather
        rand_ind = rand_ind.unsqueeze(-1).repeat(1, 1, particle_size)
        # batch x resamp_particles x particle_size

        # batch x (incoming x pseudo_bel) x particle_size
        #dst_belief_particles = dst_belief_particles

        # batch x (incoming x particles) x particle_size
        sampled_particles = torch.gather(particles, 1, rand_ind)
        return sampled_particles


    # belief_particles: Batch x IncNghbrs x NumParticles x ParticleSize
    # belief_weights: Batch x IncNghbrs x NumParticles
    def recursive_ancestral_sampling(self, belief_particles, belief_weights, ith, parent, parent_idx, visited, visited_samples, num_samples=15):

        to_sample_belief_particles = belief_particles[ith].view(belief_particles[ith].shape[0], -1, belief_particles[ith].shape[3])
        batch_size, particle_count, particle_size = to_sample_belief_particles.shape
        
        to_sample_belief_weights = belief_weights[ith].view(batch_size, -1)

        ith_idx = len(visited)
        visited.append(ith)
        
        if ith==0:
            sampled_particles = self.discrete_samples(to_sample_belief_particles, to_sample_belief_weights, num_samples)
            sampled_particles = sampled_particles.unsqueeze(2)
            visited_samples.append(sampled_particles)
            
        else:
            edge_i = ((self.graph == torch.tensor([[min(ith,parent)],
                                                           [max(ith,parent)]])).all(dim=0).nonzero().squeeze(0) 
                              == self.edge_set).nonzero().squeeze(0)
            
            sampled_particles = self.discrete_samples(to_sample_belief_particles, to_sample_belief_weights, particle_count)
            sampled_particles = sampled_particles.unsqueeze(1)
    #         print(parent,'->',ith, edge_i)
            if parent<ith:
                diff = visited_samples[parent_idx] - sampled_particles
            else:
                diff = sampled_particles - visited_samples[parent_idx]


            cond_wgts = self.edge_densities[edge_i](diff.view(-1, particle_size)).view(batch_size, num_samples, diff.shape[2])
            cond_wgts = cond_wgts.view(-1, cond_wgts.shape[2])
            cond_wgts = cond_wgts / cond_wgts.sum(dim=1, keepdim=True)
    #         print(cond_wgts.shape)
            
            sampled_particles = sampled_particles.view(-1, 1, sampled_particles.shape[2], particle_size).repeat(1,num_samples,1,1).view(-1, sampled_particles.shape[2], particle_size)
            conditioned_particles = self.discrete_samples(sampled_particles, cond_wgts, 1).view(batch_size, num_samples, particle_size)
            conditioned_particles = conditioned_particles.unsqueeze(2)
            visited_samples.append(conditioned_particles)
        
        for nghbr in self.inc_nghbrs[ith]:
            if nghbr not in visited:
                self.recursive_ancestral_sampling(belief_particles, belief_weights, nghbr, ith, ith_idx, visited, visited_samples, num_samples)
            
        
        if ith==0:
            return [x for _,x in sorted(zip(visited,visited_samples))]