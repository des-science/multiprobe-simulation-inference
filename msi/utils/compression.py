# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created March 2023
Author: Arne Thomsen

Compress summary vectors to a lower dimensional space. 

Meant for the GPU nodes of the Perlmutter cluster at NERSC.
MOPED taken from https://cosmo-gitlab.phys.ethz.ch/jafluri/arne_handover/-/blob/77c173b425e341d1d6600c0eb5196dc0d7276be3/networks/GP_emulator_network.ipynb
By Janis Fluri
"""

import numpy as np

def PCA():
    pass

def moped_reduce(all_fidu, all_vali, grid_label, all_grid, take=None, grid_cov=False, with_bary=False, splits=[],
                take_mean=False, do_plots=True):
    """
    Performs a MOPED compression on the data using the fiducial predictions to estimate the derivtives
    :param all_fidu: Array of the fiducial predictions
    :param all_vali: Array of the validation predictions
    :param grid_label: Label for the grid points
    :param all_grid: Array of the grid predictions
    :param take: List of parameter indices to take for the compression, can be a list of lists with lengths of splits argument
    :param grid_cov: Use grid and fiducial predictions for the compression covariance matrix
    :param with_bary: Wheter or not the predictions include the baryonification parameters
    :param splits: Perform multiple independent MOPED compressions on the data, this argument with be forwarded to 
                   np.split(..., axis=-1) to split the data
    :param take_mean: Take the mean of the independent MOPED compressions, different length compressions will be padded
    :param do_plots: Create some plots to check
    """
    
    # print some stuff
    print(f"All preds have shape: {all_fidu.shape}")
    print(f"All preds have shape: {all_grid.shape}")
    
    # perform the splitting if necessary
    if isinstance(splits, list) and len(splits) > 0 or isinstance(splits, int) and splits > 0:
        print("Splitting arrays according to: ", splits)
        
    fidu_splits = np.split(all_fidu, indices_or_sections=splits, axis=-1)
    vali_splits = np.split(all_vali, indices_or_sections=splits, axis=-1)
    grid_splits = np.split(all_grid, indices_or_sections=splits, axis=-1)
    
    fidu_collect = []
    vali_collect = []
    grid_collect = []
    
    # handle the take arg
    if take is None:
        all_take = [None]*len(fidu_splits)
    elif isinstance(take[0], list):
        all_take = take
    else:
        all_take = [take]*len(fidu_splits)
    max_take = np.max([len(t) for t in all_take])
    
    # perform all the MOPED compressions
    for all_fidu, all_vali, all_grid, take in zip(fidu_splits, vali_splits, grid_splits, all_take):

        # Choose the cov
        if grid_cov:
            inv_cov = np.linalg.pinv(np.cov(np.concatenate([all_fidu[0], all_grid]), rowvar=False))
        else:
            fidu_cov = np.cov(all_fidu[0], rowvar=False)
            inv_cov = np.linalg.pinv(fidu_cov)

        # get the derivatives
        derivatives = []
        for i in range((all_fidu.shape[0] - 1)//2):
            m1 = np.mean(all_fidu[2*i+1], axis=0)
            m2 = np.mean(all_fidu[2*i+2], axis=0)
            derivatives.append((m2 - m1)/(2*constants.fiducial_deltas[i]))

        if take is None:
            if with_bary:
                take = np.arange(9)
                max_take = 9
            else:
                take = np.arange(7)
                max_take = 7

        zero_ind = take[0]
        # get the b mat for the compression
        bs = []
        b1 = inv_cov.dot(derivatives[zero_ind])/np.sqrt(derivatives[zero_ind].dot(inv_cov.dot(derivatives[zero_ind])))
        bs = [b1]
        
        # only some b vecs are relevant
        for counter, i in enumerate(take[1:]):
            bi = inv_cov.dot(derivatives[i])
            for j in range(counter+1):
                bi -= derivatives[i].dot(bs[j])*bs[j]
            norm = derivatives[i].dot(inv_cov.dot(derivatives[i]))
            for j in range(counter+1):
                norm -= derivatives[i].dot(bs[j])**2
            bs.append(bi/np.sqrt(norm))

        # now we compress
        b_mat = np.stack(bs, axis=0)
        print("Genereated compression matrix with shape: ", b_mat.shape)
        if do_plots:
            plt.figure(figsize=(12,8))
            plt.imshow(b_mat)
            plt.title("b-mat")
            plt.colorbar()

        # compress the fidu preds
        all_fidu = np.einsum("ij,abj->abi", b_mat, all_fidu)
        print("Compressed fidu: ", all_fidu.shape)
        
        # plot to check
        if do_plots:
            plt.figure(figsize=(8,8))
            plt.imshow(np.cov(np.stack(all_fidu[0], axis=0), rowvar=False))
            plt.title("Compressed cov")
            plt.colorbar()

        # load the vali preds
        print("Loading vali preds")
        print(f"All preds have shape: {all_vali.shape}")

        all_vali = np.einsum("ij,aj->ai", b_mat, all_vali)
        print("Compressed vali: ", all_vali.shape)

        print("Compressing the grid")    
        all_grid = np.einsum("ij,aj->ai", b_mat, all_grid)
        print("Compressed grid: ", all_grid.shape)
        
        # collect and pad to max take
        if take_mean:
            fidu_collect.append(np.pad(all_fidu, [[0,0], [0,0], [0,max_take-all_fidu.shape[-1]]]))
            vali_collect.append(np.pad(all_vali, [[0,0], [0,max_take-all_vali.shape[-1]]]))
            grid_collect.append(np.pad(all_grid, [[0,0], [0,max_take-all_grid.shape[-1]]]))
        else:
            fidu_collect.append(all_fidu)
            vali_collect.append(all_vali)
            grid_collect.append(all_grid)
        
    # concat or mean
    if take_mean:
        all_fidu = np.mean(fidu_collect, axis=0)
        all_vali = np.mean(vali_collect, axis=0)
        all_grid = np.mean(grid_collect, axis=0)

    else:
        all_fidu = np.concatenate(fidu_collect, axis=-1)
        all_vali = np.concatenate(vali_collect, axis=-1)
        all_grid = np.concatenate(grid_collect, axis=-1)
    
    if do_plots:
        plt.figure(figsize=(8,8))
        plt.imshow(np.corrcoef(all_fidu[0], rowvar=False))
        plt.title("Compressed Fidu Corr")
        plt.colorbar()

        plt.figure(figsize=(8,8))
        plt.imshow(np.cov(all_fidu[0], rowvar=False))
        plt.title("Compressed Fidu Cov")
        plt.colorbar()

        plt.figure(figsize=(8,8))
        plt.imshow(np.corrcoef(all_grid[-280:], rowvar=False))
        plt.title("Compressed Gird Corr (single cosmo)")
        plt.colorbar()

        plt.figure(figsize=(8,8))
        plt.imshow(np.corrcoef(all_grid, rowvar=False))
        plt.title("Compressed Gird Corr (all cosmo)")
        plt.colorbar()
    
    return all_fidu, all_vali, grid_label, all_grid
    
     