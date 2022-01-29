import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def plot_unaries(bpnet, grid, x, unnormed=None, std=None, tru=None, to_show=0, est_bounds=1.0, nump=100, fname=None):
    fig, ax = plt.subplots(1, bpnet.num_nodes, figsize=(3*bpnet.num_nodes,3))
    
    for i in range(bpnet.num_nodes):
        ax[i].set_xlim(-est_bounds,est_bounds)
        ax[i].set_ylim(-est_bounds,est_bounds)


        if unnormed is None:
            unnormed = x
        
        ax[i].imshow(unnormed[to_show].cpu().permute(1,2,0), extent=[-est_bounds, est_bounds, -est_bounds, est_bounds])
        
        if std:
            out = bpnet.likelihood(i, x[to_show].unsqueeze(0), std, tru[to_show,i], grid.unsqueeze(0)).cpu()
        else:
            out = bpnet.likelihood(i, x[to_show].unsqueeze(0), grid.unsqueeze(0)).cpu()
        out = (F.interpolate(out.view(1,1,nump,nump).flip(2),(1000,1000)).squeeze()/out.max()).detach().numpy()
        
        if bpnet.num_nodes==3:
            if i==0:
                out_c = plt.cm.Reds(out)
            elif i==1:
                out_c = plt.cm.Greens(out)
            elif i==2:
                out_c = plt.cm.Blues(out)
        else:
            out_c = plt.cm.Reds(out)

        out_c[:,:,3] = out
        ax[i].imshow(out_c, extent=[-est_bounds,est_bounds,-est_bounds,est_bounds])
    
    if fname:
        fig.savefig(fname, dpi=fig.dpi)
        plt.close('all')
    else:
        plt.show()

    
def plot_belief_particles(bpnet, grid, x, unnormed=None, with_particles=True, std=None, to_show=0, est_bounds=1.0, nump=100, s=2, fname=None):
    fig, ax = plt.subplots(1, bpnet.num_nodes, figsize=(3*bpnet.num_nodes,3))
    
    for i in range(bpnet.num_nodes):
        if bpnet.num_nodes==3:
            if i==0:
                c = 'r'
            elif i==1:
                c = 'g'
            elif i==2:
                c = 'b'
        else:
            c = 'r'

        if unnormed is None:
            unnormed = x

        ax[i].imshow(unnormed[to_show].cpu().permute(1,2,0), extent=[-est_bounds, est_bounds, -est_bounds, est_bounds])
        
        if with_particles:
            ax[i].scatter(x=bpnet.belief_particles[i].view(bpnet.batch_size, -1, bpnet.particle_size)[to_show,:,0].cpu().detach().numpy(), 
                            y=bpnet.belief_particles[i].view(bpnet.batch_size, -1, bpnet.particle_size)[to_show,:,1].detach().cpu().numpy(), c=c, s=s)
        
        if std:
            out = bpnet.density_estimation(i, grid, std=std).cpu()[to_show]
        else:
            out = bpnet.density_estimation(i, grid).cpu()[to_show]
        out = (F.interpolate(out.view(1,1,nump,nump).flip(2),(1000,1000)).squeeze()/out.max()).detach().numpy()
        if bpnet.num_nodes==3:
            if i==0:
                out_c = plt.cm.Reds(out)
            elif i==1:
                out_c = plt.cm.Greens(out)
            elif i==2:
                out_c = plt.cm.Blues(out)
        else:
            out_c = plt.cm.Reds(out)
        out_c[:,:,3] = out
        ax[i].imshow(out_c, extent=[-est_bounds,est_bounds,-est_bounds,est_bounds])
        
    if fname:
        fig.savefig(fname, dpi=fig.dpi)
        plt.close('all')
    else:
        plt.show()
    
#     for i in range(3):
#         figg, axx = plt.subplots()
#         axx.hist(bpnet.belief_weights[i].view(bpnet.batch_size, -1)[to_show].cpu().detach(), bins=200)
#         plt.show()
    
# def plot_intermediate_particles():

    
    
def plot_pairwise_sampling(bpnet, num_samples=10000, est_bounds=1.0, s=5, fname=None):
    if bpnet.multi_edge_samplers:
        to_plot = 2
    else:
        to_plot = 1
    fig, ax = plt.subplots(to_plot, bpnet.num_edges, figsize=(3*bpnet.num_edges, 3*to_plot))
    for i in range(bpnet.num_edges):
        for j in range(to_plot):
            if bpnet.multi_edge_samplers:
                samp = bpnet.edge_samplers[i][j](num_samples).cpu().detach()
            else:
                samp = bpnet.edge_samplers[i](num_samples).cpu().detach()
            if to_plot>1:    
                ax[j,i].set_ylim(-est_bounds,est_bounds)
                ax[j,i].set_xlim(-est_bounds,est_bounds)
                ax[j,i].scatter(x=samp[:,0], y=samp[:,1], c='r', s=s)
            else:
                ax[i].set_ylim(-est_bounds,est_bounds)
                ax[i].set_xlim(-est_bounds,est_bounds)
                ax[i].scatter(x=samp[:,0], y=samp[:,1], c='r', s=s)
    if fname:
        fig.savefig(fname, dpi=fig.dpi)
        plt.close('all')
    else:
        plt.show()
    
    
def plot_pairwise_densities(bpnet, grid, std=None, est_bounds=1.0, nump=100, fname=None):
    fig, ax = plt.subplots(1, bpnet.num_edges, figsize=(3*bpnet.num_edges,3))
    for i in range(bpnet.num_edges):
        ax[i].set_ylim(-est_bounds,est_bounds)
        ax[i].set_xlim(-est_bounds,est_bounds)
        
        if std:
            out = bpnet.edge_densities[i](grid.squeeze(1), std).cpu()
        else:
            out = bpnet.edge_densities[i](grid.squeeze(1)).cpu()
        out = (F.interpolate(out.view(1,1,nump,nump).flip(2),(1000,1000)).squeeze()).detach().numpy()
        ax[i].imshow(out, extent=[-est_bounds,est_bounds,-est_bounds,est_bounds])

    if fname:
        fig.savefig(fname, dpi=fig.dpi)
        plt.close('all')
    else:
        plt.show()
    
    
def plot_timedelta_sampling(bpnet, num_samples=10000, est_bounds=1.0, s=5, ground_truth=False, fname=None):
    fig, ax = plt.subplots(1, bpnet.num_nodes, figsize=(3*bpnet.num_nodes,3))
    for i in range(bpnet.num_nodes):
        ax[i].set_ylim(-est_bounds,est_bounds)
        ax[i].set_xlim(-est_bounds,est_bounds)
   
        if ground_truth:
            samp = bpnet.time_samplers[i](i, num_samples).cpu().detach()
        else:
            samp = bpnet.time_samplers[i](num_samples).cpu().detach()
        ax[i].scatter(x=samp[:,0], y=samp[:,1], c='r', s=s)
    
    if fname:
        fig.savefig(fname, dpi=fig.dpi)
        plt.close('all')
    else:
        plt.show()
    

def plot_msg_wgts(bpnet, x, mode, unnormed=None, to_show=0, bin_size=0.2, fname=None):
    fig, ax = plt.subplots(1, bpnet.num_nodes, figsize=(3*bpnet.num_nodes,3))
    
    if mode=='w_lik':
        wgt_to_use = [wgt[to_show].cpu() for wgt in bpnet.belief_weights_lik]
    elif mode=='w_unary':
        wgt_to_use = [wgt[to_show].cpu() for wgt in bpnet.message_weights_unary]
    elif mode=='w_neigh':
        wgt_to_use = [wgt[to_show].cpu() for wgt in bpnet.message_weights_neigh]
    else:
        raise
    msgs = [m[to_show].cpu() for m in bpnet.message_particles]
    
    
    for i in range(bpnet.num_nodes):
        scores = np.zeros((int(np.rint(2/bin_size)), int(np.rint(2/bin_size))))
        counts = np.zeros((int(np.rint(2/bin_size)), int(np.rint(2/bin_size))))
        wgt_to_use[i] = wgt_to_use[i].view(-1)
        wgt_to_use[i] /= wgt_to_use[i].sum()
        
        
        
        msgs[i][:,:,0] += 1
        msgs[i][:,:,1] -= 1
        msgs[i][:,:,1] *= -1
        msgs[i]-=0.0001
        msgs[i] /= bin_size
        msgs[i] = msgs[i].type(torch.int)
        msgs[i] = msgs[i].view(-1, 2)

        for i_part, coord in enumerate(msgs[i]):
            scores[coord[1], coord[0]] += wgt_to_use[i][i_part]
            counts[coord[1], coord[0]] += 1
        counts[counts==0]=1
        out = np.divide(scores, counts)
        out = out/out.max()
        if bpnet.num_nodes==3:
            if i==0:
                out_c = plt.cm.Reds(out)
            elif i==1:
                out_c = plt.cm.Greens(out)
            elif i==2:
                out_c = plt.cm.Blues(out)
        else:
            out_c = plt.cm.Reds(out)
        out_c[:,:,3] = out
        
        if unnormed is None:
            unnormed = x

        ax[i].imshow(unnormed[to_show].permute(1,2,0).cpu(), extent=[-1,1,-1,1])
        ax[i].imshow(out_c, extent=[-1,1,-1,1])
    

    if fname:
        fig.savefig(fname, dpi=fig.dpi)
        plt.close('all')
    else:
        plt.show()






def density_estimation(x, msgs, wgts):
    belief_particles = msgs.view(1, 1, -1, 2).double()
    

    weights = wgts.view(1, 1, -1).double()
    
       

    x = x.double()
    diffsq = (((x-belief_particles)/0.05)**2).sum(dim=-1)
    exp_val = torch.exp((-1/2) * diffsq)
    fact = 1/(0.05*np.sqrt(2*np.pi))
    fact_ = fact * exp_val
    out = (weights * fact_).sum(dim=-1)

    return out


def plot_msg(bpnet, x, unnormed=None, to_show=0, bin_size=0.2, est_bounds=1.0, fname=None):
    
    
    
    msgs = [m[to_show] for m in bpnet.message_particles][0]
    
    
    # for i in range(bpnet.num_nodes):



    wgt_lik = [wgt[to_show] for wgt in bpnet.belief_weights_lik][0]

    wgt_unary = [wgt[to_show] for wgt in bpnet.message_weights_unary][0]

    wgt_neigh = [wgt[to_show] for wgt in bpnet.message_weights_neigh][0]


    msg_wgts = wgt_unary*wgt_neigh
    msg_wgts = msg_wgts / msg_wgts.sum(1,keepdim=True)

    tmp_wgt = wgt_lik * wgt_unary * wgt_neigh
    tmp_wgt = tmp_wgt / tmp_wgt.sum()

    # tmp_wgt_1 = wgt_lik * wgt_unary * wgt_neigh
    # tmp_wgt_1 = tmp_wgt_1 / tmp_wgt_1.sum()
    # print(msg_wgts.sum(dim=1, keepdim=True))
    # print(wgt_unary.sum(dim=1, keepdim=True))
    # print(wgt_neigh.sum(dim=1, keepdim=True))
    # print(wgt_lik.sum(dim=1, keepdim=True))
    # print(tmp_wgt-tmp_wgt_1)
    # print(wgt_lik.shape, msgs.shape)
    # print(wgt_unary.shape, msgs.shape)
    # print(wgt_neigh.shape, msgs.shape)


    outname01 = fname+'0->1.jpg'
    outname21 = fname+'2->1.jpg'
        

    # ax.scatter(x=msgs[0,:,0].cpu().detach().numpy(), 
    #                 y=msgs[0,:,1].detach().cpu().numpy(), c='g', s=2)
        
    nump=100


    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')

    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_xticks([-1,-0.5,0,0.5,1])
    ax.set_yticks([-1,-0.5,0,0.5,1])

    out = density_estimation(x, msgs[0], msg_wgts[0]).cpu()[to_show]
    out = (F.interpolate(out.view(1,1,nump,nump).flip(2),(1000,1000)).squeeze()/out.max()).detach().numpy()

    out_c = plt.cm.Greens(out)
    out_c[:,:,3] = out
    ax.imshow(unnormed[to_show].permute(1,2,0).cpu(), extent=[-est_bounds,est_bounds,-est_bounds,est_bounds], alpha=0.5)
    ax.imshow(out_c, extent=[-est_bounds,est_bounds,-est_bounds,est_bounds])
    

    fig.savefig(outname01, dpi=300)
    plt.close(fig)



    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')

    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_xticks([-1,-0.5,0,0.5,1])
    ax.set_yticks([-1,-0.5,0,0.5,1])
    
    out = density_estimation(x, msgs[1], msg_wgts[1]).cpu()[to_show]
    out = (F.interpolate(out.view(1,1,nump,nump).flip(2),(1000,1000)).squeeze()/out.max()).detach().numpy()

    out_c = plt.cm.Greens(out)
    out_c[:,:,3] = out
    ax.imshow(unnormed[to_show].permute(1,2,0).cpu(), extent=[-est_bounds,est_bounds,-est_bounds,est_bounds], alpha=0.5)
    ax.imshow(out_c, extent=[-est_bounds,est_bounds,-est_bounds,est_bounds])
    

    fig.savefig(outname21, dpi=300)
    plt.close(fig)




    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')

    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_xticks([-1,-0.5,0,0.5,1])
    ax.set_yticks([-1,-0.5,0,0.5,1])
    
    out = density_estimation(x, msgs, tmp_wgt/tmp_wgt.sum()).cpu()[to_show]
    out = (F.interpolate(out.view(1,1,nump,nump).flip(2),(1000,1000)).squeeze()/out.max()).detach().numpy()

    out_c = plt.cm.Greens(out)
    out_c[:,:,3] = out
    ax.imshow(unnormed[to_show].permute(1,2,0).cpu(), extent=[-est_bounds,est_bounds,-est_bounds,est_bounds], alpha=0.5)
    ax.imshow(out_c, extent=[-est_bounds,est_bounds,-est_bounds,est_bounds])
    

    fig.savefig(fname+'unary.jpg', dpi=300)
    plt.close(fig)


    
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')

    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_xticks([-1,-0.5,0,0.5,1])
    ax.set_yticks([-1,-0.5,0,0.5,1])
    
    out = density_estimation(x, msgs, msg_wgts/msg_wgts.sum()).cpu()[to_show]
    out = (F.interpolate(out.view(1,1,nump,nump).flip(2),(1000,1000)).squeeze()/out.max()).detach().numpy()

    out_c = plt.cm.Greens(out)
    out_c[:,:,3] = out
    ax.imshow(unnormed[to_show].permute(1,2,0).cpu(), extent=[-est_bounds,est_bounds,-est_bounds,est_bounds], alpha=0.5)
    ax.imshow(out_c, extent=[-est_bounds,est_bounds,-est_bounds,est_bounds])
    

    fig.savefig(fname+'union.jpg', dpi=300)
    plt.close(fig)



    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')

    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_xticks([-1,-0.5,0,0.5,1])
    ax.set_yticks([-1,-0.5,0,0.5,1])
    
    out = density_estimation(x, msgs, tmp_wgt).cpu()[to_show]
    out = (F.interpolate(out.view(1,1,nump,nump).flip(2),(1000,1000)).squeeze()/out.max()).detach().numpy()

    out_c = plt.cm.Greens(out)
    out_c[:,:,3] = out
    ax.imshow(unnormed[to_show].permute(1,2,0).cpu(), extent=[-est_bounds,est_bounds,-est_bounds,est_bounds], alpha=0.5)
    ax.imshow(out_c, extent=[-est_bounds,est_bounds,-est_bounds,est_bounds])
    

    fig.savefig(fname+'bel.jpg', dpi=300)
    plt.close(fig)