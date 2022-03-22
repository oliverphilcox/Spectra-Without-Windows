# compute_bk_fisher.py (Oliver Philcox, 2021)
### Compute the combined fisher matrices for the binned bispectrum of survey data with FKP or ML weightings.
### This can then be used to construct the full bispectrum estimates.
### Note compute_bk_randoms.py must be run on N_mc sims before this script to compute Fisher matrix contributions

# Import modules
from nbodykit.lab import *
import sys, os, time, configparser
import numpy as np

# Read command line arguments
if len(sys.argv)!=2:
    raise Exception("Need to specify parameter file!")
else:
    paramfile = str(sys.argv[1]) # parameter file

############################### INPUT PARAMETERS ###############################

# Load input file, and read in a few crucial parameters
config = configparser.ConfigParser(interpolation=None,converters={'list': lambda x: [i.strip() for i in x.split(',')]})
if not os.path.exists(paramfile):
    raise Exception("Parameter file does not exist!")
config.read(paramfile)

string = config['sample']['type']

# Define weight type and sample name
weight_type = str(config['settings']['weights'])
if weight_type=='FKP':
    wtype = 0
elif weight_type=='ML':
    wtype = 1
else:
    raise Exception("weights must be 'FKP' or 'ML'")

# Binning parameters
k_min, k_max, dk = float(config['bk-binning']['k_min']),float(config['bk-binning']['k_max']),float(config['bk-binning']['dk'])
assert k_max>k_min
assert dk>0
assert k_min>=0
lmax = int(config['bk-binning']['lmax'])
assert (lmax//2)*2==lmax, "l-max must be even!"

## Directories
mcdir = str(config['directories']['monte_carlo'])
outdir =  str(config['directories']['output'])

# Number of MC simulations
N_mc = int(config['settings']['N_mc'])

# Create directories
if not os.path.exists(outdir): os.makedirs(outdir)

# Summarize parameters
print("\n###################### PARAMETERS ######################\n")
print("Data-type: %s"%string)
print("Weight-Type: %s"%weight_type)
print("\nk-min: %.3f"%k_min)
print("k-max: %.3f"%k_max)
print("dk: %.3f"%dk)
print("l-max: %d"%lmax)
print("\nMonte Carlo Directory: %s"%mcdir)
print("Output Directory: %s"%outdir)
print("\n########################################################")

init = time.time()


############################### DEFINE BINNING ##############################

n_l = lmax//2+1
n_k = int((k_max-k_min)/dk)

def test_bin(a,b,c,tol=1e-6):
    """Test bin to see if it satisfies triangle conditions, being careful of numerical overlaps."""
    k_lo = np.arange(k_min,k_max,dk)
    k_hi = k_lo+dk
    # Maximum k3 possible
    k_up = k_hi[a]+k_hi[b]
    if a>b: k_do = k_lo[a]-k_hi[b]
    elif a==b:
        k_do = 0.
    else:
        k_do = k_lo[b]-k_hi[a]
    if k_lo[c]+tol>=k_up or k_hi[c]-tol<=k_do:
        return 0
    else:
        return 1

# First check which elements must be computed
bins_index = []
n_bins = 0
for a in range(n_k):
    for b in range(a,n_k):
        for c in range(b,n_k):
            if not test_bin(a,b,c): continue
            bins_index.append([a,b,c])
            n_bins += (lmax//2+1)

# Compute triangle factor
Delta_abc = np.zeros(n_bins//n_l)
i = 0
for a in range(n_k):
    for b in range(a,n_k):
        for c in range(b,n_k):
            if not test_bin(a,b,c): continue
            if a==b and a==c and b==c:
                Delta_abc[i] = 6.
            elif a==b or a==c or b==c:
                Delta_abc[i] = 2.
            else:
                Delta_abc[i] = 1.
            i += 1

########################### CONSTRUCT FISHER MATRIX ###########################

# First check if Fisher matrix already exists
full_fish_file_name = outdir+'%s_mean%d_%s_full-fish_alpha_beta_k%.3f_%.3f_%.3f_l%d.npy'%(string,N_mc,weight_type,k_min,k_max,dk,lmax)

if os.path.exists(full_fish_file_name):
    print("\nFisher matrix already computed; exiting!\n")
    sys.exit()

# Compute mean Fisher matrix < tilde-phi > C^-1 < phi >

print("\n## Computing mean Fisher matrix contribution")

sum_Cinv_phi_alpha_file_name = lambda a,b,c,l_i: mcdir+'sum_%s_unif%d_%s_Cinv-phi^alpha_map%d,%d,%d,%d_k%.3f_%.3f_%.3f_l%d.npz'%(string,N_mc,weight_type,a,b,c,l_i,k_min,k_max,dk,lmax)
sum_tilde_phi_alpha_file_name = lambda a,b,c,l_i: mcdir+'sum_%s_unif%d_%s_tilde-phi^alpha_map%d,%d,%d,%d_k%.3f_%.3f_%.3f_l%d.npz'%(string,N_mc,weight_type,a,b,c,l_i,k_min,k_max,dk,lmax)

def load_row_mean(alpha):
    ### Load a single row of the mean Fisher matrix, < phi_alpha > C^-1 < phi_beta >/12
    # Note that we include combinatoric factors in phi_alpha here

    l_i = alpha//(n_bins//n_l)
    a_index = (alpha%(n_bins//n_l))
    a,b,c = bins_index[a_index]

    this_row = np.zeros(n_bins)
    infile = np.load(sum_tilde_phi_alpha_file_name(a,b,c,l_i))
    if infile['ct']!=N_mc:
        print(np.sort(infile['its']))
        print(alpha)
        raise Exception("Wrong number of tilde-phi bias simulations computed! (%d of %d)"%(infile['ct'],N_mc))
    mean_tilde_phi_alpha = infile['dat']
    infile.close()

    for beta in range(alpha,n_bins): # compute only upper triangle and diagonal by symmetry

        l_j = beta//(n_bins//n_l)
        b_index = (beta%(n_bins//n_l))
        a2,b2,c2 = bins_index[b_index]

        infile = np.load(sum_Cinv_phi_alpha_file_name(a2,b2,c2,l_j))
        if infile['ct']!=N_mc:
            print(np.sort(infile['its']))
            print(beta)
            raise Exception("Wrong number of Cinv-phi bias simulations computed! (%d of %d)"%(infile['ct'],N_mc))
        mean_Cinv_phi_beta = infile['dat']
        infile.close()
        this_row[beta] = np.real(np.sum(mean_tilde_phi_alpha*mean_Cinv_phi_beta)/12.)
        del mean_Cinv_phi_beta
    del mean_tilde_phi_alpha

    return this_row

import multiprocessing as mp, tqdm
# Find out how many processes to use
n_proc = mp.cpu_count()

print("\nComputing Fisher matrix on %d cores"%n_proc)
p = mp.Pool(processes=n_proc)

# Multiprocess file loading with tqdm for timings
mean_fisher = np.array(list(tqdm.tqdm(p.imap(load_row_mean,range(n_bins)),total=n_bins)))
p.close()
p.join()
print("Multi-processed Fisher matrix computation complete")

# Add symmetry factor
Delta_abc_all = np.hstack([Delta_abc for _ in range(n_l)])
mean_fisher *= 4./np.outer(Delta_abc_all,Delta_abc_all)

## Add in conjugate symmetry
for alpha in range(n_bins):
    for beta in range(alpha+1,n_bins):
        mean_fisher[beta,alpha] = mean_fisher[alpha,beta]

# Compute full Fisher matrix contribution, < tilde-phi C^-1 phi >
print("\n### Computing full Fisher matrix contribution")

### Define file names
def fish_file_name(bias_sim):
    return mcdir+'%s_unif%d_%s_fish_alpha_beta_k%.3f_%.3f_%.3f_l%d.npy'%(string,bias_sim,weight_type,k_min,k_max,dk,lmax)

# Iterate over simulations and normalize correctly
full_fisher = np.zeros((n_bins,n_bins))
for i in range(1,N_mc+1):
    full_fisher += np.load(fish_file_name(i))/N_mc

full_fisher -= mean_fisher
np.save(full_fish_file_name,full_fisher)

duration = time.time()-init
print("## Saved Fisher matrix to %s. Exiting after %d seconds (%d minutes)\n"%(full_fish_file_name,duration,duration//60))
sys.exit()
