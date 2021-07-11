# compute_pk_boss_randoms.py (Oliver Philcox, 2021)
### Compute the power spectrum of BOSS or Patchy data with FKP or ML weightings
### This computes the q-bar and F_ab terms from uniformly distributed randoms (independent of the survey geometry)
### We use the BOSS covariances for this; compute_pk_randoms.py is used for Patchy covariances.

# Import modules
from nbodykit.lab import *
import sys, os, copy, time, pyfftw
import numpy as np
from scipy.interpolate import interp1d
# custom definitions
sys.path.append('../src')
from opt_utilities import load_data, load_randoms, load_MAS, load_nbar, grid_data, grid_uniforms, load_coord_grids, compute_spherical_harmonics, compute_filters, ft, ift
from covariances_pk import applyC_alpha, applyN

# Read command line arguments
if len(sys.argv)!=6:
    raise Exception("Need to specify random iteration, patch, z-type, weight-type and grid factor!")
else:
    rand_it = int(sys.argv[1])
    patch = str(sys.argv[2]) # ngc or sgc
    z_type = str(sys.argv[3]) # z1 or z3
    wtype = int(sys.argv[4]) # 0 for FKP, 1 for ML
    grid_factor = float(sys.argv[5])

################################ INPUT PARAMETERS ##############################

## k-space binning
k_min = 0.0
k_max = 0.26
dk = 0.005
lmax = 4

## Cosmological parameters for co-ordinate conversions
h_fid = 0.676
OmegaM_fid = 0.31

# Whether to forward-model pixellation effects.
include_pix = False
# If true, use nbar(r) from the random particles instead of the mask / n(z) distribution.
rand_nbar = True

# Directories
outdir = '/projects/QUIJOTE/Oliver/pk_opt_patchy/' # to hold output Fisher matrices

if wtype==1:
    # Fiducial power spectrum input
    pk_input_file = '/projects/QUIJOTE/Oliver/bk_opt/patchy_%s_%s_pk_fid_k_0.00_0.30.txt'%(patch,z_type)

#### In principle, nothing below here needs to be altered for BOSS

# Redshifts
if z_type=='z1':
    ZMIN = 0.2
    ZMAX = 0.5
    z = 0.38
elif z_type=='z3':
    ZMIN = 0.5
    ZMAX  = 0.75
    z = 0.61
else:
    raise Exception("Wrong z-type")

# Load survey dimensions
if z_type=='z1' and patch=='ngc':
    boxsize_grid = np.array([1350,2450,1400])
    grid_3d = np.asarray(np.asarray([252.,460.,260.])/grid_factor,dtype=int)
elif z_type=='z1' and patch=='sgc':
    boxsize_grid = np.array([1000,1900,1100])
    grid_3d = np.asarray(np.asarray([190.,360.,210.])/grid_factor,dtype=int)
elif z_type=='z3' and patch=='ngc':
    boxsize_grid = np.array([1800,3400,1900])
    grid_3d = np.asarray(np.asarray([340.,650.,360.])/grid_factor,dtype=int)
elif z_type=='z3' and patch=='sgc':
    boxsize_grid = np.array([1000,2600,1500])
    grid_3d = np.asarray(np.asarray([190.,500.,280.])/grid_factor,dtype=int)
else:
    raise Exception("Wrong z-type / patch")

# Create directories
if not os.path.exists(outdir): os.makedirs(outdir)

# Covariance matrix parameters
if wtype==0:
    weight_str = 'fkp'
    from covariances_pk import applyCinv_fkp
elif wtype==1:
    weight_str = 'ml'
    from covariances_pk import applyCinv
else:
    raise Exception("Incorrect weight type!")

# Summarize parameters
print("\n###################### PARAMETERS ######################\n")
print("Random iteration: %d"%rand_it)
print("Grid-Factor: %.1f"%grid_factor)
print("Weight-Type: %s"%weight_str)
print("\nPatch: %s"%patch)
print("Redshift-type: %s"%z_type)
if rand_nbar:
    print("n-bar: from randoms (BOSS)")
else:
    print("n-bar: from mask (BOSS)")
print("Forward model pixellation: %d"%include_pix)
print("\nk-min: %.3f"%k_min)
print("k-max: %.3f"%k_max)
print("dk: %.3f"%dk)
print("\nFiducial h = %.3f"%h_fid)
print("Fiducial Omega_m = %.3f"%OmegaM_fid)
print("\nOutput Directory: %s"%outdir)
print("\n########################################################")

init = time.time()

################################# LOAD DATA ####################################

print("\n## Analyzing random iteration %d for %s %s with %s weights and grid-factor %.1f assuming BOSS geometry"%(rand_it,patch,z_type,weight_str,grid_factor))

# First check that the simulation hasn't already been analyzed
bias_file_name = outdir+'boss%d_%s_%s_%s_g%.1f_pk_q-bar_a_k%.3f_%.3f_%.3f.npy'%(rand_it,patch,z_type,weight_str,grid_factor,k_min,k_max,dk)
fish_file_name = outdir+'boss%d_%s_%s_%s_g%.1f_pk_fish_a_k%.3f_%.3f_%.3f.npy'%(rand_it,patch,z_type,weight_str,grid_factor,k_min,k_max,dk)

if os.path.exists(bias_file_name) and os.path.exists(fish_file_name):
    print("Output already exists; exiting!\n")
    sys.exit()

### Load fiducial cosmology for co-ordinate conversions (in nbodykit)
cosmo_coord = cosmology.Cosmology(h=h_fid).match(Omega0_m = OmegaM_fid)

# Load a uniform random sample for data
nbar_unif = 1e-3
data = UniformCatalog(nbar_unif,boxsize_grid,seed=rand_it)
print("Created %d uniform randoms"%len(data))

# Assign to a grid
diff = grid_uniforms(data, nbar_unif, boxsize_grid, grid_3d, MAS='TSC')
shot_fac_unif = 1.
del data

# Compute alpha for nbar rescaling
data_true = load_data(-1,ZMIN,ZMAX,cosmo_coord,patch=patch,fkp_weights=False)
rand_true = load_randoms(-1,ZMIN,ZMAX,cosmo_coord,patch=patch,fkp_weights=False)
alpha_ran = (np.sum(data_true['WEIGHT'])/np.sum(rand_true['WEIGHT'])).compute()
shot_fac = (np.mean(data_true['WEIGHT']**2.).compute()+alpha_ran*np.mean(rand_true['WEIGHT']**2.).compute())/np.mean(rand_true['WEIGHT']).compute()
norm = 1./np.asarray(alpha_ran*np.sum(rand_true['NBAR']*rand_true['WEIGHT']*rand_true['WEIGHT_FKP']**2.))
print("Data: alpha_ran = %.3f, shot_factor: %.3f"%(alpha_ran,shot_fac))

if rand_nbar:
    print("Loading nbar from random particles")
    nbar_rand, density = grid_data(data_true, rand_true, boxsize_grid,grid_3d,MAS='TSC',return_randoms=True,return_norm=False)[1:]
else:
    # load density mesh (used to define coordinate arrays)
    density = grid_data(data_true, rand_true, boxsize_grid,grid_3d,MAS='TSC',return_randoms=False,return_norm=False)[1]

# Compute renormalization factor to match gridded randoms
renorm2 = np.asarray(alpha_ran*((rand_true['NBAR']*rand_true['WEIGHT'])).sum())
del rand_true, data_true

# Load pre-computed n(r) map (from mask and n(z), not discrete particles)
print("Loading nbar from mask")
nbar_mask = load_nbar(-1, patch, z_type, ZMIN, ZMAX, grid_factor, alpha_ran)

# Load grids in real and Fourier space
k_grids, r_grids = load_coord_grids(boxsize_grid, grid_3d, density)
k_norm = np.sqrt(np.sum(k_grids**2.,axis=0))
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

# Apply renormalization factors
print("Renormalization factor: %.3f"%(renorm2/(np.sum(nbar**2.)*v_cell)))
nbar *= np.sqrt(renorm2/(np.sum(nbar**2.)*v_cell))
nbar_weight *= np.sqrt(renorm2/(np.sum(nbar_weight**2.)*v_cell))

############################## GRID DEFINITIONS ################################

# Compute spherical harmonic fields in real and Fourier-space
Yk_lm, Yr_lm = compute_spherical_harmonics(lmax,k_grids,r_grids)

if wtype==1:
    # Load fit to Patchy P(k)
    pk_input = np.loadtxt(pk_input_file)
    fid_pk_interp = interp1d(pk_input[:,0],pk_input[:,1:].T)
    pk_map = fid_pk_interp(k_norm)[:lmax//2+1]

del r_grids, k_grids

# Compute k-space filters
k_filters = compute_filters(k_min,k_max,dk,k_norm)
n_k = len(k_filters)

### True inverse covariance
def applyCinv_unif(input_map):
    """Apply true C^{-1} to the uniform map including MAS effects."""
    return ift(ft(input_map)*MAS_mat**2)/nbar_unif*v_cell/shot_fac_unif

######################### COMPUTE FISHER + BIAS ################################

## Compute C^-1 a and A^-1 a for random map a
print("\n## Computing C^-1 of map assuming %s weightings"%weight_str)
if wtype==0:
    Cinv_diff = applyCinv_fkp(diff,nbar_weight,MAS_mat,v_cell,shot_fac,include_pix=include_pix)
else:
    Cinv_diff = applyCinv(diff,nbar_weight,MAS_mat,pk_map,Yk_lm,Yr_lm,v_cell,shot_fac,rel_tol=1e-6,verb=1,max_it=30,include_pix=include_pix)

Ainv_diff = applyCinv_unif(diff)
del diff

# Compute N A^-1 a
N_Ainv_a = applyN(Ainv_diff,nbar,MAS_mat,v_cell,shot_fac,include_pix=include_pix)

### Compute C_a C^-1 a
print("## Computing C_a C^-1 of map assuming %s weightings\n"%weight_str)
C_a_Cinv_diff = applyC_alpha(Cinv_diff,nbar,MAS_mat,Yk_lm,Yr_lm,v_cell,k_filters,lmax,include_pix=include_pix,data=False)
C_a_Ainv_diff = applyC_alpha(Ainv_diff,nbar,MAS_mat,Yk_lm,Yr_lm,v_cell,k_filters,lmax,include_pix=include_pix,data=False)

del Cinv_diff, Ainv_diff, nbar

### Compute C^-1 C_a C^-1 a
print("## Computing C^-1 C_a C^-1 of map assuming %s weightings"%weight_str)
n_bins = len(C_a_Cinv_diff)
Cinv_C_a_Cinv_diff = []
for alpha in range(n_bins):

    if (alpha+1)%5==0: print("On bin %d of %d"%(alpha+1,n_bins))
    if wtype==0:
        tmp_map = applyCinv_fkp(C_a_Cinv_diff[alpha],nbar_weight,MAS_mat,v_cell,shot_fac,include_pix=include_pix)
    else:
        tmp_map = applyCinv(C_a_Cinv_diff[alpha],nbar_weight,MAS_mat,pk_map,Yk_lm,Yr_lm,v_cell,shot_fac,rel_tol=1e-6,verb=0,max_it=30,include_pix=include_pix) # C^-1.x

    Cinv_C_a_Cinv_diff.append(tmp_map)
    del tmp_map
del nbar_weight, MAS_mat, Yk_lm, Yr_lm

### Compute Fisher matrix
fish = np.zeros((n_bins,n_bins))

print("\n## Computing Fisher matrix")
for alpha in range(n_bins):

    if (alpha+1)%5==0: print("On bin %d of %d"%(alpha+1,n_bins))
    this_Cinv_C_a_Cinv_diff = Cinv_C_a_Cinv_diff[alpha]
    for beta in range(n_bins):
        fish[alpha,beta] = 0.5*np.real_if_close(np.sum(this_Cinv_C_a_Cinv_diff*C_a_Ainv_diff[beta]))

    del this_Cinv_C_a_Cinv_diff
del C_a_Ainv_diff

## Symmetrize matrix
fish = 0.5*(fish+fish.T)

### Compute bias term
q_bias = np.zeros(n_bins)

print("\n## Computing bias term")
for alpha in range(n_bins):
    q_bias[alpha] = 0.5*np.real_if_close(np.sum(Cinv_C_a_Cinv_diff[alpha]*N_Ainv_a))
del N_Ainv_a

############################### SAVE AND EXIT ##################################

## Save output
np.save(bias_file_name,q_bias)
np.save(fish_file_name,fish)

duration = time.time()-init
print("\n## Saved output to %s and %s. Exiting after %d seconds (%d minutes)\n\n"%(bias_file_name,fish_file_name,duration,duration//60))
sys.exit()
