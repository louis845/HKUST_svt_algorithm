import os

os.environ["SCIPY_USE_PROPACK"] = "1"

import utils
import scipy
import scipy.sparse
import scipy.sparse.linalg
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import math


# returns the no. of useful singular values (non-zero after shrinkage) and representation by SVD
# pack_used is used for debug
def shrinkage_operator(M, tau, suggested_rank, increment):
    # find first few singular values and vectors
    suggested_rank = max(min(suggested_rank, min(M.shape)), 1)
    
    u,s,v = None,None,None
    pack_used = 'none'
    
    try:
        u,s,v = scipy.sparse.linalg.svds(M, k = suggested_rank, which = 'LM', solver = 'propack')
        pack_used = 'propack'
    except np.linalg.LinAlgError as e:
        suggested_rank = min(suggested_rank, min(M.shape)-1)
        u,s,v = scipy.sparse.linalg.svds(M, k = suggested_rank, which = 'LM', solver = 'arpack')
        pack_used = 'arpack'
    useful_singular_values = (s > tau).sum()
    while useful_singular_values == suggested_rank and suggested_rank < min(M.shape):
        suggested_rank = max(min(suggested_rank + increment, min(M.shape)), 1)
        
        try:
            u,s,v = scipy.sparse.linalg.svds(M, k = suggested_rank, which = 'LM', solver = 'propack')
            pack_used = 'propack'
        except np.linalg.LinAlgError as e:
            suggested_rank = min(suggested_rank, min(M.shape)-1)
            u,s,v = scipy.sparse.linalg.svds(M, k = suggested_rank, which = 'LM', solver = 'arpack')
            pack_used = 'arpack'
        useful_singular_values = (s > tau).sum()
    
    
    # decrease by tau
    s = np.maximum(s - tau, 0)
    return useful_singular_values,u,s,v,pack_used

def construct_matrix_from_svd(u,s,v,locations):
    u = cp.array(u)
    s = cp.array(s)
    v = cp.array(v)
    result = cp.multiply(u,s) @ v

    return utils.filter_locations(result.get(), locations)

def multiply_matrix_gpu(u,s,v):
    u = cp.array(u)
    s = cp.array(s)
    v = cp.array(v)
    result = cp.multiply(u,s) @ v
    
    return cp.asnumpy(result)

def svt_algorithm_plot_info(fro, log10fro, nuc, rank):
    fig, ax = plt.subplots()

    sequence1 = fro
    sequence2 = log10fro
    sequence3 = nuc
    sequence4 = rank

    p1, = ax.plot(sequence1, color="red")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("fro")
    ax2 = ax.twinx()
    ax2.set_ylabel("log10 fro")
    ax3 = ax.twinx()
    ax3.set_ylabel("nuclear")
    ax4 = ax.twinx()
    ax4.set_ylabel("rank")
    ax3.spines["right"].set_position(("axes", 1.3))
    ax4.spines["right"].set_position(("axes", 1.5))

    p2, = ax2.plot(sequence2, color="blue")
    p3, = ax3.plot(sequence3, color="green")
    p4, = ax4.plot(sequence4, color="orange")

    ax.yaxis.label.set_color(p1.get_color())
    ax2.yaxis.label.set_color(p2.get_color())
    ax3.yaxis.label.set_color(p3.get_color())
    ax4.yaxis.label.set_color(p4.get_color())

    plt.show()

# M is a sparse matrix, locations are an array of points (i,j)
# tolerance_absolute means that the stopping of the SVT algorithm depends on relative or absolute cutoff
def svt_algorithm(M, locations, step_size, tolerance, tau, increment, log, tolerance_absolute = False):
    # frobenius norm of M, used for stopping criterion
    norm_of_m = scipy.sparse.linalg.norm(M, ord='fro')
    
    Y = utils.create_empty_sparse_matrix(M.shape)
    
    # the guess of the rank of the matrix
    suggested_rank = 1
    
    # previous frobenius norm
    previous_fro = -1
    
    # previous rank
    previous_rank = -1
    
    # how many times the l2 fails to decrease
    fixed_count = 0
    
    u,s,v = None,None,None
    
    fro_norm_history = []
    log_fro_norm_history = []
    nuc_norm_history = []
    rank_history = []
    
    while True:
        notsuccess = True
        count = 0
        while notsuccess:
            try:
                suggested_rank,u,s,v,pack_used = shrinkage_operator(Y, tau, suggested_rank, increment)
                notsuccess = False
            except Exception as e:
                print(e)
                count = count + 1
                if count == 5:
                    print("Error in SVT algorithm, SVD doesn't work after 5 trials.")
                    return None
        
        projection = M - construct_matrix_from_svd(u, s, v, locations)
        this_fro = scipy.sparse.linalg.norm(projection, ord='fro')
        
        if tolerance_absolute:
            if this_fro <= tolerance:
                break
        else:
            # this is just stopping criterion in paper
            if this_fro/norm_of_m <= tolerance:
                break
        
        # check if frobenius norm has increased by a margin
        if (this_fro-previous_fro)/previous_fro > 0.05:
            if log:
                svt_algorithm_plot_info(fro_norm_history, log_fro_norm_history, nuc_norm_history, rank_history)
            return None
        
        # check if rank has decreased
        if suggested_rank - previous_rank < 0:
            if log:
                svt_algorithm_plot_info(fro_norm_history, log_fro_norm_history, nuc_norm_history, rank_history)
            return None
            
        Y = Y + step_size * projection
        previous_fro = this_fro
        previous_rank = suggested_rank
        
        if log:
            print("fro: ",this_fro,"    nuc:",max(s),"     last rank:",suggested_rank)
            fro_norm_history.append(this_fro)
            log_fro_norm_history.append(math.log10(this_fro))
            nuc_norm_history.append(max(s))
            rank_history.append(suggested_rank)
    
    if log:
        svt_algorithm_plot_info(fro_norm_history, log_fro_norm_history, nuc_norm_history, rank_history)
        
    X = multiply_matrix_gpu(u,s,v)
    return X

"""
Uses the parameters if the rank is known
"""
def svt_algorithm_auto_params_known_rank(M, locations, rank, log = False, tolerance = 0.001):
    step_size = 1.2 * M.shape[0]*M.shape[1] / len(locations[0])
    
    max_size = int(max(M.shape[0],M.shape[1]))
    increment = max(4, int(max(M.shape[0],M.shape[1])/50))
    K = scipy.sparse.linalg.norm(M, ord='fro')
    K = (K*K) / (rank*len(locations[0]))
    
    tau = 5 * max_size
    otau = tau
    
    mresult = None
    while mresult is None:
        print("Step size: ",step_size,"    tau: ",tau)
        mresult = svt_algorithm(M/K, locations, step_size, tolerance, tau, increment, log)
        # increase thresholding for next step
        tau += otau
    
    return mresult * K

"""
Small step size version for stability
"""
def svt_algorithm_small_step_size(M, locations, log = False, suggested_tau = None, step_size = 1.99, increment = 50, tolerance = 0.001, tolerance_absolute = False):
    print("Step size: ",step_size)
    
    max_size = int(max(M.shape[0],M.shape[1]))
    
    tau = 5 * max_size
    if suggested_tau is not None:
        tau = suggested_tau
    print("Tau:",tau)
    # assume the multiplier is at most 20
    result = svt_algorithm(M, locations, step_size, tolerance, tau, increment, log, tolerance_absolute)
    if not (result is None):
        return result
    print("Error: svt unknown rank unable to complete matrix--------------------------")
    return None

# use parameters recommended by the paper
def svt_algorithm_standard_parameters(M, locations, tau, increment, log = False):
    # m/n1n2
    step_size = 1.2 * M.shape[0]*M.shape[1] / len(locations[0])
    print("Step size: ",step_size)
    tolerance = 0.001
    return svt_algorithm(M, locations, step_size, tolerance, tau, increment, log)