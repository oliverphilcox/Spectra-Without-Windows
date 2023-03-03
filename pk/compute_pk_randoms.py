# compute_pk_randoms.py (Oliver Philcox, 2021)
### Compute the power spectrum of survey data with FKP or ML weightings
### This computes the q-bar and F_ab terms from Gaussian distributed randoms (independent of the survey geometry)
### All input parameters (specifying the mask, k-cuts etc.) are specified in a given .param file

# Import modules
from nbodykit.lab import *
import numpy as np, sys, os, time, configparser, shutil
from scipy.interpolate import interp1d
# custom definitions
sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/../src')
from opt_utilities import load_data, load_randoms, load_nbar, load_MAS, grid_data, load_coord_grids, compute_spherical_harmonic_functions, compute_filters, ft, ift
from covariances_pk import applyC_alpha, applyN

# Read command line arguments
if len(sys.argv)!=3:
    raise Exception("Need to specify random iteration and parameter file!")
else:
    rand_it = int(sys.argv[1]) # which random catalog
    paramfile = str(sys.argv[2]) # ngc or sgc

################################ INPUT PARAMETERS ##############################

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
    from covariances_pk import applyCinv_fkp
elif weight_type=='ML':
    wtype = 1
    from covariances_pk import applyCinv
else:
    raise Exception("weights must be 'FKP' or 'ML'")

# Binning parameters
k_min, k_max, dk = float(config['pk-binning']['k_min']),float(config['pk-binning']['k_max']),float(config['pk-binning']['dk'])
assert k_max>k_min
assert dk>0
assert k_min>=0
lmax = int(config['pk-binning']['lmax'])
assert (lmax//2)*2==lmax, "l-max must be even!"

## Directories
outdir =  str(config['directories']['output'])
mcdir = str(config['directories']['monte_carlo'])

# Redshifts
ZMIN, ZMAX = float(config['sample']['z_min']), float(config['sample']['z_max'])

# Fiducial cosmological parameters
h_fid, OmegaM_fid = float(config['parameters']['h_fid']), float(config['parameters']['OmegaM_fid'])

# Survey dimensions
boxsize_grid = np.array(config.getlist('sample','box'),dtype=float)
grid_3d = np.array(config.getlist('pk-binning','grid'),dtype=int)

# Number of MC simulations
N_mc = int(config['settings']['N_mc'])
if rand_it>N_mc: raise Exception("Simulation number cannot be greater than number of bias sims!")

if wtype==1:
    # Fiducial power spectrum input (for ML weights)
    pk_input_file = str(config['settings']['fiducial_pk'])

# Testing parameters
include_pix, rand_nbar, low_mem = config.getboolean('settings','include_pix'), config.getboolean('settings','rand_nbar'), config.getboolean('settings','low_mem')

if low_mem:
    from covariances_pk import applyC_alpha_single
    tmpdir = str(config['directories']['temporary'])+str('%d/'%(rand_it))

    # Remove crud from a previous run
    if os.path.exists(tmpdir): shutil.rmtree(tmpdir)
    # Create directory
    if not os.path.exists(tmpdir): os.makedirs(tmpdir)

# Create directories
if not os.path.exists(outdir): os.makedirs(outdir)
if not os.path.exists(mcdir): os.makedirs(mcdir)

# Summarize parameters
print("\n###################### PARAMETERS ######################\n")
print("Data-type: %s"%string)
print("Random iteration: %d"%rand_it)
print("Weight-Type: %s"%weight_type)
if rand_nbar:
    print("n-bar: from randoms")
else:
    print("n-bar: from mask")
print("Forward model pixellation: %d"%include_pix)
print("\nk-min: %.3f"%k_min)
print("k-max: %.3f"%k_max)
print("dk: %.3f"%dk)
print("l-max: %d"%lmax)
print("\nFiducial h = %.3f"%h_fid)
print("Fiducial Omega_m = %.3f"%OmegaM_fid)
print("\nOutput Directory: %s"%outdir)
print("\nBias / Fisher Directory: %s"%mcdir)
if low_mem:
    print("\nTemporary Directory: %s"%tmpdir)
print("\n########################################################")

init = time.time()

################################# LOAD DATA ####################################

# First check that the simulation hasn't already been analyzed
bias_file_name = mcdir+'%s%d_%s_pk_q-bar_a_k%.3f_%.3f_%.3f_l%d.npy'%(string,rand_it,weight_type,k_min,k_max,dk,lmax)
fish_file_name = mcdir+'%s%d_%s_pk_fish_a_k%.3f_%.3f_%.3f_l%d.npy'%(string,rand_it,weight_type,k_min,k_max,dk,lmax)
combined_bias_file_name = outdir + 'bias-pk_%s%d_%s_k%.3f_%.3f_%.3f_l%d.npy'%(string,N_mc,weight_type,k_min,k_max,dk,lmax)
combined_fish_file_name = outdir + 'fisher-pk_%s%d_%s_k%.3f_%.3f_%.3f_l%d.npy'%(string,N_mc,weight_type,k_min,k_max,dk,lmax)

if os.path.exists(bias_file_name) and os.path.exists(fish_file_name):
    print("Simulation already completed; exiting!\n")
    sys.exit()

if os.path.exists(combined_bias_file_name) and os.path.exists(combined_fish_file_name):
    print("Full Fisher matrices already completed; exiting!\n")
    sys.exit()

print("\n## Loading %s random iteration %d with %s weights"%(string,rand_it,weight_type))

### Load fiducial cosmology for co-ordinate conversions (in nbodykit)
cosmo_coord = cosmology.Cosmology(h=h_fid).match(Omega0_m = OmegaM_fid)

# Generate nbar field (same density as no. randoms), filling in zeros to allow for MAS bleed
nbar_A = load_nbar(config, 'pk', 1.)
nbar_A[nbar_A==0] = min(nbar_A[nbar_A!=0])*0.01

# Gaussian sample to create random field
np.random.seed(rand_it)
diff = np.random.normal(loc=0.,scale=np.sqrt(nbar_A),size=nbar_A.shape)

# Compute alpha for nbar rescaling
print("\nLoading data")
data_true = load_data(1,config,cosmo_coord,fkp_weights=False)
rand_true = load_randoms(config,cosmo_coord,fkp_weights=False)
alpha_ran = (data_true['WEIGHT'].sum()/rand_true['WEIGHT'].sum()).compute()
shot_fac = ((data_true['WEIGHT']**2.).mean().compute()+alpha_ran*(rand_true['WEIGHT']**2.).mean().compute())/rand_true['WEIGHT'].mean().compute()
print("Data-to-random ratio: %.3f, shot-noise factor: %.3f"%(alpha_ran,shot_fac))

if rand_nbar:
    print("Loading nbar from random particles")
    nbar_rand, density = grid_data(data_true, rand_true, boxsize_grid, grid_3d, MAS='TSC', return_randoms=True, return_norm=False)[1:]
else:
    # load density mesh (used to define coordinate arrays)
    density = grid_data(data_true, rand_true, boxsize_grid,grid_3d,MAS='TSC',return_randoms=False,return_norm=False)[1]
del rand_true, data_true

# Load pre-computed n(r) map (from mask and n(z), not discrete particles)
print("Loading nbar from mask")
nbar_mask = load_nbar(config, 'pk', alpha_ran)

# Load grids in real and Fourier space
k_grids, r_grids = load_coord_grids(boxsize_grid, grid_3d, density)
k_norm = np.sqrt(np.sum(k_grids**2.,axis=0))
k_grids /= (1e-12+k_norm)
r_grids /= (1e-12+np.sqrt(np.sum(r_grids**2.,axis=0)))
del density

# Load MAS grids
MAS_mat = load_MAS(boxsize_grid, grid_3d)

# For weightings, we should use a smooth nbar always.
nbar_weight = nbar_mask.copy()
if rand_nbar:
    nbar = nbar_rand.copy()
    del nbar_rand
else:
    nbar = nbar_mask.copy()
del nbar_mask

# Cell volume
v_cell = 1.*boxsize_grid.prod()/(1.*grid_3d.prod())

############################## GRID DEFINITIONS ################################

# Compute spherical harmonic fields in real and Fourier-space
Y_lms = compute_spherical_harmonic_functions(lmax)

if wtype==1:
    # Load fit to input P(k)
    pk_input = np.loadtxt(pk_input_file)
    fid_pk_interp = interp1d(pk_input[:,0],pk_input[:,1:].T)
    pk_map = fid_pk_interp(k_norm)[:lmax//2+1]

# Compute k-space filters
k_filters = compute_filters(k_min,k_max,dk)
n_k = int((k_max-k_min)/dk)

### True inverse covariance
def applyAinv(input_map):
    """Apply true A^{-1} to the input map (no MAS effects needed!)."""
    return input_map/nbar_A

######################### COMPUTE FISHER + BIAS ################################

## Compute C^-1 a and A^-1 a for random map a
print("\n## Computing C^-1 of map assuming %s weightings"%weight_type)
if wtype==0:
    Cinv_diff = np.real(applyCinv_fkp(diff,nbar_weight,MAS_mat,v_cell,shot_fac,include_pix=include_pix))
else:
    Cinv_diff = np.real(applyCinv(diff,nbar_weight,MAS_mat,pk_map,Y_lms,k_grids,r_grids,v_cell,shot_fac,rel_tol=1e-6,verb=1,max_it=30,include_pix=include_pix))

Ainv_diff = applyAinv(diff)
del diff

# Compute N A^-1 a
N_Ainv_a = applyN(Ainv_diff,nbar,MAS_mat,v_cell,shot_fac,include_pix=include_pix)

### Compute C^-1 C_a C^-1 a and C_a A^-1 a
if low_mem:
    n_bins = n_k*(lmax//2+1)
    print("## Computing C_a C^-1 of map assuming %s weightings\n"%weight_type)
    for alpha in range(n_bins):
        # Compute C_a A^-1 a
        this_C_a_Ainv_diff = applyC_alpha_single(Ainv_diff,nbar,MAS_mat,Y_lms,k_grids,r_grids,v_cell,k_filters,k_norm,alpha//n_k,alpha%n_k,include_pix=include_pix,data=False)
        np.save(tmpdir+'C_a_Ai_%d.npy'%(alpha),this_C_a_Ainv_diff)
        del this_C_a_Ainv_diff
else:
    print("## Computing C_a C^-1 of map assuming %s weightings\n"%weight_type)
    C_a_Cinv_diff = applyC_alpha(Cinv_diff,nbar,MAS_mat,Y_lms,k_grids,r_grids,v_cell,k_filters,k_norm,n_k,lmax,include_pix=include_pix,data=False)
    C_a_Ainv_diff = applyC_alpha(Ainv_diff,nbar,MAS_mat,Y_lms,k_grids,r_grids,v_cell,k_filters,k_norm,n_k,lmax,include_pix=include_pix,data=False)

    n_bins = len(C_a_Cinv_diff)

    # Compute C^-1 of maps
    print("## Computing C^-1 C_a C^-1 of map assuming %s weightings\n"%weight_type)
    if wtype==0:
        Cinv_C_a_Cinv_diff = [applyCinv_fkp(C_a_Cinv_diff[alpha],nbar_weight,MAS_mat,v_cell,shot_fac,include_pix=include_pix) for alpha in range(n_bins)]
    else:
        Cinv_C_a_Cinv_diff = [applyCinv(C_a_Cinv_diff[alpha],nbar_weight,MAS_mat,pk_map,Y_lms,k_grids,r_grids,v_cell,shot_fac,rel_tol=1e-6,verb=0,max_it=30,include_pix=include_pix) for alpha in range(n_bins)]
    del Cinv_diff, nbar, k_norm, nbar_weight, MAS_mat, Y_lms, k_grids, r_grids

del Ainv_diff

### Compute Fisher matrix and bias term, saving each element in turn
print("## Computing Fisher and bias term")
fish = np.zeros((n_bins,n_bins))
q_bias = np.zeros(n_bins)

for alpha in range(n_bins):
    if (alpha+1)%5==0: print("On bin %d of %d"%(alpha+1,n_bins))

    if low_mem:
        ### Compute C_a C^-1 a on the fly
        tmp_map = applyC_alpha_single(Cinv_diff,nbar,MAS_mat,Y_lms,k_grids,r_grids,v_cell,k_filters,k_norm,alpha//n_k,alpha%n_k,include_pix=include_pix,data=False)

        ### Compute C^-1 C_a C^-1 a on the fly
        if wtype==0:
            this_Cinv_C_a_Cinv_diff = applyCinv_fkp(tmp_map,nbar_weight,MAS_mat,v_cell,shot_fac,include_pix=include_pix)
        else:
            this_Cinv_C_a_Cinv_diff = applyCinv(tmp_map,nbar_weight,MAS_mat,pk_map,Y_lms,k_grids,r_grids,Yr_lm,v_cell,shot_fac,rel_tol=1e-6,verb=0,max_it=30,include_pix=include_pix) # C^-1.x
        del tmp_map
    else:
        this_Cinv_C_a_Cinv_diff = Cinv_C_a_Cinv_diff[alpha]

    # Compute bias term
    q_bias[alpha] = 0.5*np.real_if_close(np.sum(this_Cinv_C_a_Cinv_diff*N_Ainv_a))
    
    # Compute Fisher term
    for beta in range(n_bins):
        if low_mem:
            fish[alpha,beta] = 0.5*np.real_if_close(np.sum(this_Cinv_C_a_Cinv_diff*np.load(tmpdir+'C_a_Ai_%d.npy'%beta)))
        else:
            fish[alpha,beta] = 0.5*np.real_if_close(np.sum(this_Cinv_C_a_Cinv_diff*C_a_Ainv_diff[beta]))
    del this_Cinv_C_a_Cinv_diff

if not low_mem: del C_a_Ainv_diff, Cinv_C_a_Cinv_diff
del N_Ainv_a, Cinv_diff, nbar, k_norm, nbar_weight, MAS_mat, Y_lms, k_grids, r_grids

## Symmetrize matrix
fish = 0.5*(fish+fish.T)

############################### SAVE AND EXIT ##################################

## Save output
np.save(bias_file_name,q_bias)
np.save(fish_file_name,fish)

# Remove temporary files if necessary
if low_mem:
    print("Removing temporary directory")
    if os.path.exists(tmpdir): shutil.rmtree(tmpdir)

duration = time.time()-init
print("\n## Saved output to %s and %s. Exiting after %d seconds (%d minutes)\n"%(bias_file_name,fish_file_name,duration,duration//60))
sys.exit()
