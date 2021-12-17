import glob
import h5py as h5
import numpy as np
from matplotlib import pyplot as plt
import sys
import pickle as pkl

'''
This is helper code to generate HDF5 databases
from pickled simulation files.
It may be a slow process, as the data is compressed as it is stored.
'''

# path to the folder in which your pkl files are stored
pkl_path = 'YOUR_FOLDER_PATH/'
# full filename and path for the newly created hdf5 file
hdf5_out_fn = 'YOUR_HDF5_FILENAME_PATH.hdf5'

# set the number of sampled permissivity and immunity values, iterations
# and max match number
# this should be set from your simulation conditions
# current values are from the publication
perm_steps = 59
imm_steps = 50
iterations = 100000
M = 50

# obtain all simulations files and sort them alpha-numerically
filenames = list(sorted(glob.glob(pkl_path + '*.p')))

# generate the permissivity values from the simulations
perm_vals = np.logspace(np.log(0.1), np.log(300), num=50, base=np.e)
perm_vals = np.array(list(perm_vals) + [
    perm_vals[-1] * np.power(perm_vals[-1] / perm_vals[-2], _) for _ in range(1,10)])

# build a dictionary linking permissivity values in filenames
# to full float values
perms = {'{:07.02f}'.format(_[0]): _[1] for _ in zip(perm_vals, range(perm_steps))}

# build a dictionary linking permissivity values in filenames
# to full float values
imms = {'{:.2f}'.format(_[0]): _[1] for _ in zip(np.linspace(0,1,50), range(imm_steps))}

# generate hdf5 file for output
# NOTE: this will fail if the file already exists
f = h5.File(hdf5_out_fn, 'w', libver='latest')

# generate dataset storage space in the hdf5 database
# probability distributions
prob = f.create_dataset("prob_dists", (perm_steps, imm_steps, iterations+1, M+1), compression="lzf", dtype=np.float64, shuffle=True)
# order parameter as calculated from probability distributions
ordparam = f.create_dataset("ord_param", (perm_steps, imm_steps, iterations+1), compression="lzf", dtype=np.float64, shuffle=True)
# environmental viral loads
pop = f.create_dataset("pop_size", (perm_steps, imm_steps, iterations+1), compression="lzf", dtype=np.float64, shuffle=True)
# viral load inside cells just after infection
psii = f.create_dataset("psi_i", (perm_steps, imm_steps, iterations+1, M+1), compression="lzf", dtype=np.float64, shuffle=True)
# viral load inside cells just after immune clearance
psixi = f.create_dataset("psi_xi", (perm_steps, imm_steps, iterations+1, M+1), compression="lzf", dtype=np.float64, shuffle=True)
# viral load inside cells just after replication
psir = f.create_dataset("psi_r", (perm_steps, imm_steps, iterations+1, M+1), compression="lzf", dtype=np.float64, shuffle=True)

# go through all simulations and load their data into the hdf5 database
for fn in filenames:
    with open(fn, 'rb') as g:
        data = pkl.load(g)
        # the lookup data for permissivity and immunity assumes the output
        # filename structure as written in multiproc_run() in runSim.py
        # if you change it there, change it here accordingly
        inp = fn.split('/')[-1].split('_')
        i = perms[inp[0]]
        j = imms[inp[1][:-2]]
        print(i,j)
        f['prob_dists'][i, j] = data[0]
        # calculate the order parameter (sample mean) of the prob dist
        f['ord_param'][i, j] = (np.sum(np.arange(51))/50.) * np.average(f['prob_dists'][i, j], weights = np.arange(51), axis=1)
        f['pop_size'][i, j] = data[1]
        f['psi_i'][i, j] = data[2]
        f['psi_xi'][i, j] = data[3]
        f['psi_r'][i, j] = data[4]

f.close()
