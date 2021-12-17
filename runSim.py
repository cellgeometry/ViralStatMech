import glob
import h5py as h5
import numpy as np
from matplotlib import pyplot as plt
import sys
import pickle as pkl
import time
from scipy import linalg as LA
from scipy import optimize as op
from functools import partial
from numba import jit
import multiprocessing as mp
from itertools import product
from ray.util.multiprocessing import Pool as rpool
import ray
import sys

'''
Helper function to compute exact infection probabilities
 for integer viral load
We use numba to accelerate these calculations

Variables are defined as follows:
	num_vir: non-negative int, number of viruses in the environment
			 attempting to infect cells
	num_cells: non-negative int, number of cells in pool
	e: 1-D numpy array of length M+1 with real-valued components in [0,1]
		the replicative fitnesses of viruses in the current simulation
	prob_vec: 1-D numpy array of length M+1 with real-valued 
				components in [0,1] whose (L-1) sum is 1
			  the match-number distribution of viruses in the environment
	psi_r: 1-D numpy array of length M+1 with real-valued 
				components in [0,1] whose (L-1) sum is <= 1
			  the match-number distribution of viruses in cells
'''
@jit("float64[:](int64, int64, float64[:], float64[:], float64[:])", nopython=True, cache=True)
def lambda_int(num_vir,num_cells, e, prob_vec, psi_r):
    psi_lambda = psi_r
    if num_vir != 0:
        for i in range(num_vir):
            psi_lambda = psi_lambda + (1.0 / num_cells) * prob_vec * (
                1 - np.power(1 - e * (1 - np.sum(psi_lambda)), num_cells))
    return psi_lambda


'''
Function to compute interpolated infection probabilities
We use numba to accelerate these calculations

Variables are defined as follows:
	num_vir: non-negative real, number of viruses in the environment
			 attempting to infect cells
	num_cells: non-negative int, number of cells in pool
	e: 1-D numpy array of length M+1 with real-valued components in [0,1]
		the replicative fitnesses of viruses in the current simulation
	prob_vec: 1-D numpy array of length M+1 with real-valued 
				components in [0,1] whose (L-1) sum is 1
			  the match-number distribution of viruses in the environment
	psi_r: 1-D numpy array of length M+1 with real-valued 
				components in [0,1] whose (L-1) sum is <= 1
			  the match-number distribution of viruses in cells
'''
@jit("UniTuple(float64[:], 2)(int64, int64, float64[:], float64[:], float64[:])", nopython=True, cache=True)
def calc_lambda(num_vir, num_cells, e, prob_vec, psi_r):
    inv = int(num_vir)
    return (num_vir-inv)*lambda_int(1+inv,num_cells, e, prob_vec, psi_r), (
        1-num_vir+inv)*lambda_int(inv,num_cells, e, prob_vec, psi_r)


'''
This is the main code for running simulations

Variables are defined as follows:
	iterations: positive int, number of cycles to run the simulation
	N: non-negative real, initial number of viruses in the environment
	c: positive int, number of cells in pool
	T: positive real, permissivity of cells
	phi: positive real, single-virus fecundity (# of offspring reproduced)
	p: real in [0, 1], probability of post-immune viruses staying in cells
	M: positive integer, maximum match number (must be set to 50)
	A: real in [0, 1], maximum immune clearance intensity
	v: int in [0, 50], half-max point of immune clearance function
	w: real in [0, 1], sequence degeneracy in mutation (must be 0.7867)
	initial_dist: 1-D positive real numpy array of length M+1
				  a probability vector whose components sum (L1) to 1
				  the initial match-number distribution of viruses
				  in the environment

In our simulations, we always set the following:
	c=1000, phi = 20, p=1.0, M=50, v=6, w=0.7867
'''
def run_iter(iterations, N, T, A, initial_dist, c=1000, phi=20, p=1.0, M=50, v=6, w=0.7867):
    t_start = time.time()
    
    # result storage instantiation
    N_list = np.zeros(iterations+1)
    lambda_list = np.zeros((iterations+1, M+1))
    psi_i = np.zeros(M+1)
    psi_i_list = np.zeros((iterations+1, M+1))
    psi_xi = np.zeros(M+1)
    psi_xi_list = np.zeros((iterations+1, M+1))
    psi_r = np.zeros(M+1)
    psi_r_list = np.zeros((iterations+1, M+1))
    prob_dist_list = np.zeros((iterations+1,M+1))

    # set and store initial values
    N_list[0] = N
    prob_dist = initial_dist   
    prob_dist_list[0] = prob_dist
    
    # arrhenius temperature (permissivity) fitness function
    e = np.exp(-(M-np.arange(M+1))/T)
    # immunity definition
    xi = A/(1.0 + np.exp(-(np.arange(M+1)-v)/2))

    # construct mutation matrix for example in manuscript
    mut_mat = np.zeros((M+1,M+1))
    for i in range(0,M):
        # one fewer match
        mut_mat[i][i+1] = (w/100.0)*((i+1)*1.0/(1.0 + np.exp(-(1.0*(i+1)-10)/2)))
        # one more match
        mut_mat[i+1][i] = (w/235.45)*(np.exp(4.709*(1.0-1.0*i/50.0))-1)
        # same number of matches

    # edge cases for mutation matrix
    for i in range(1,M):
        mut_mat[i][i] = 1.0 - mut_mat[i+1][i] - mut_mat[i-1][i]
    mut_mat[0][0] = 1.0 - mut_mat[1][0]
    mut_mat[M][M] = 1.0 - mut_mat[M-1][M]

    # iterative simulation code
    for i in range(iterations):

    	# calculate and store the distribution after infection
        psi_n = calc_lambda(N, c, e, prob_dist, psi_r)
        psi_i = psi_n[0] + psi_n[1]
        psi_i_list[i + 1] = psi_i

        # relative probability of viruses escaping immune response
        psi_xi = psi_i*(1.0 - xi)
        psi_xi_list[i + 1] = psi_xi

        # relative probability of viruses staying in the cell after reproduction
        psi_r = psi_xi*(1.0 - e)*p
        psi_r_list[i + 1] = psi_r
              
        # expected value of new viruses in environment
        N = c*phi*np.dot(e,psi_xi)
        N = np.round(N, 12)

        # probability of escaping cells after reproduction
        prob_dist = ((1-p)*(1-e)+e)*psi_xi/np.dot(((1-p)*(1-e)+e),psi_xi)

        # probabilities of matches post mutation
        prob_dist = np.dot(mut_mat,prob_dist)
        # make sure this is a probability vector and sums to 1
        prob_dist = prob_dist / np.sum(prob_dist)

        # store results
        N_list[i + 1] = N
        prob_dist_list[i+1] = prob_dist

        # set zero condition and halt, if reached
        if N == 0.0:
            prob_dist_list[i+2:] = prob_dist_list[i+2:] - 1
            psi_i_list[i+2:] = psi_i_list[i+2:] - 1
            psi_xi_list[i+2:] = psi_xi_list[i+2:] - 1
            psi_r_list[i+2:] = psi_r_list[i+2:] - 1
            break

    print("Total time taken is {}".format(time.time() - t_start))
    return prob_dist_list, N_list, psi_i_list, psi_xi_list, psi_r_list

'''
Helper function for running multiple simulations in parallel
Variables are defined as follows:
	i: length-3 list whose components are
		num of iterations, initial num of viruses, and initial dist flag
	j: length-2 list whose components are
		permissivity value, immunity value
	outpath: string, base filepath for results. Must end in '/' for linux
'''
def multiproc_run(i, j, outpath):
    import numpy as np
    natural_initial = np.array([3.13953260e-21, 1.33428859e-17, 1.57488776e-14, 6.91381689e-12,
       1.27910726e-09, 1.07591451e-07, 4.36224080e-06, 9.00254968e-05,
       1.00199084e-03, 6.41564309e-03, 2.53578797e-02, 6.64702686e-02,
       1.23479311e-01, 1.71871012e-01, 1.87129781e-01, 1.64435823e-01,
       1.19195867e-01, 7.23539813e-02, 3.71601819e-02, 1.62635234e-02,
       6.09636624e-03, 1.96443667e-03, 5.45622527e-04, 1.30893469e-04,
       2.71633142e-05, 4.88185519e-06, 7.60469466e-07, 1.02732171e-07,
       1.20386028e-08, 1.22375530e-09, 1.07882508e-10, 8.24354760e-12,
       5.45535764e-13, 3.12302321e-14, 1.54421361e-15, 6.58225325e-17,
       2.41279120e-18, 7.58289692e-20, 2.03569061e-21, 4.64698672e-23,
       8.96963640e-25, 1.45376396e-26, 1.96123439e-28, 2.17792376e-30,
       1.96218985e-32, 1.40672185e-34, 7.81160534e-37, 3.22946953e-39,
       9.32969778e-42, 1.67612024e-44, 1.40606883e-47], dtype=np.float64)
    uniform_initial = np.ones(51) / 51
    # define your initial distribution here
    # it must be a 1-D numpy matrix with entries in [0, 1]
    # and its components must sum (L-1) to 1.
    your_dist = np.ones(51) / 51

    if i[2] == 0:
        with open(outpath + 'natural/01kcells_{:05d}v/{:07.02f}_{:.2f}.p'.format(i[1],j[0],j[1]), "wb") as f:
            pkl.dump(run_iter(iterations=i[0], N=i[1], T=j[0], A=j[1], initial_dist=natural_initial, c=1000, 
            	phi=20, p=1.0, M=50, v=6, w=0.7867), 
            f, protocol=4)
    elif i[2] == 1:
        with open(outpath + 'uniform/01kcells_{:05d}v/{:07.02f}_{:.2f}.p'.format(i[1],j[0],j[1]), "wb") as f:
            pkl.dump(run_iter(iterations=i[0], N=i[1], T=j[0], A=j[1], initial_dist=uniform_initial, c=1000, 
            	phi=20, p=1.0, M=50, v=6, w=0.7867), 
            f, protocol=4)
    elif i[2] == 2:
    	with open(outpath + 'mydist/01kcells_{:05d}v/{:07.02f}_{:.2f}.p'.format(i[1],j[0],j[1]), "wb") as f:
            pkl.dump(run_iter(iterations=i[0], N=i[1], T=j[0], A=j[1], initial_dist=your_dist, c=1000, 
            	phi=20, p=1.0, M=50, v=6, w=0.7867), 
            f, protocol=4)
    else:
    	print('Invalid initial distribution')   
    return



'''
Code for running simulations

We include three examples:
1) running and saving a single simulation
2) running multiple simulations in parallel, using python multiprocessing
3) running multiple simulations in parallel, using ray multiprocessing 
	(better for clusters but also works locally)
'''

# run simulation(s)
if __name__ == 'main':

	# set only one of the following to True
	# run a single simulation
	run_single = True
	# run many simulations in parallel, using python multiprocessing
	run_multi = False
	# run many simulations in parallel, using Ray
	run_multi_ray = False

	if run_single:
		# set number of iteratinos
		iters = 100000
		# set number of viruses initially in environment
		init_vir = 10000
		# set permittivity value
		perm = 100
		# set immunity max value
		imm = 0.5
		# set initial distribution
		init_dist = np.ones(51) / 51

		# set where you want to save the results
		output_fn = 'YOUR_OUTPUT_FILE'

		# run and save
		with open(output_fn, 'wb') as f:
			pkl.dump(run_iter(iterations=iters, N=init_vir, T=perm, A=imm, initial_dist=init_dist))

	else:
		# permittivity valuse from paper
		# log-distributed from 0.1 to 300 (inclusive) with 50 points,
		# then extended nine additional values
		perms = np.logspace(np.log(0.1), np.log(300), num=50, base=np.e)
		perms = np.array(list(perms) + [perms[-1] * np.power(perms[-1] / perms[-2], _) for _ in range(1,10)])
		# immunity values from paper
		# linearly distribution from 0 to 1 (inclusive) with 50 points
		imms = np.linspace(0,1,50)
		# 100K iterations
		iters = 100000
		# 10K viruses initially in environment
		init_vir = 10000
		# set flag for natural (0), uniform (1), or your (2) initial distribution
		init_dist flag = 0
		# set output path for completed simulations
		output_path = 'YOUR_OUTPUT_PATH'

		# generate permissivity-immunity pairs for parallel simulations
		params = list(product(perms, imms))

		# fill out inputs for multiproc_run
		params = list(zip([iters, init_vir, init_dist_flag] * len(params), params, [output_path] * len(params)))

		# number of threads you will use for simulations
		# recommended: max number of threads - 1
		num_procs = mp.cpu_count() - 1

		# only run a single type of multiprocessing
		if (run_multi ^ run_multi_ray):
			if run_multi:
				pool = mp.Pool(processes=num_procs)
			else:
				pool = rpool(processes=num_procs)

			out = pool.starmap(multiproc_run, params)
			pool.close()
			pool.join()

		else:
			print('Multiprocessing simulation flag error.')

