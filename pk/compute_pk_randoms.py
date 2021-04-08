# compute_pk_randoms.py (Oliver Philcox, 2021)
### Compute the power spectrum of BOSS or Patchy data with FKP or ML weightings
### This computes the q-bar and F_ab terms from uniformly distributed randoms (independent of the survey geometry)

# Import modules
from nbodykit.lab import *
import sys, os, copy, time, pyfftw
import numpy as np
from scipy.interpolate import interp1d
# custom definitions
sys.path.append('../src')
from opt_utilities import load_data, load_randoms, load_MAS, load_nbar, grid_data, load_coord_grids, compute_spherical_harmonics, compute_filters, ft, ift, plotter
from covariances_pk import applyC_alpha, applyN

# Read command line arguments
if len(sys.argv)!=4:
    raise Exception("Need to specify random iteration, weight-type and grid factor!")
else:
    # If sim no = -1 the true BOSS data is used
    rand_it = int(sys.argv[1])
    wtype = int(sys.argv[2]) # 0 for FKP, 1 for ML
    grid_factor = float(sys.argv[3])

################################ INPUT PARAMETERS ##############################

## Simulation parameters
patch = 'ngc'
z_type = 'z1'

## k-space binning
k_min = 0.0
k_max = 0.41
dk = 0.005
lmax = 4

## Cosmological parameters for co-ordinate conversions
h_fid = 0.676
OmegaM_fid = 0.31

# Directories
outdir = '/projects/QUIJOTE/Oliver/pk_opt/' # to hold output Fisher matrices

# Fiducial power spectrum input
pk_input_file = '/projects/QUIJOTE/Oliver/bk_opt/patchy_%s_%s_pk_fid_k_0.00_0.30.txt'%(patch,z_type)

#### In principle, nothing below here needs to be altered for BOSS

# box dimensions (scaled from BOSS release)
if patch=='ngc' and z_type=='z1':
    boxsize_grid = np.array([1350,2450,1400])
    grid_3d = np.asarray(np.asarray([252.,460.,260.])/grid_factor,dtype=int)
else:
    raise Exception()

# Redshifts
if z_type=='z1':
    ZMIN = 0.2
    ZMAX = 0.5
    z = 0.38
else:
    raise Exception()

# Create directories
if not os.path.exists(outdir): os.makedirs(outdir)

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
print("\nk-min: %.3f"%k_min)
print("k-max: %.3f"%k_max)
print("dk: %.3f"%dk)
print("\nFiducial h = %.3f"%h_fid)
print("Fiducial Omega_m = %.3f"%OmegaM_fid)
print("\nOutput Directory: %s"%outdir)
print("\n########################################################")

init = time.time()

################################# LOAD DATA ####################################

print("\n## Analyzing random iteration %d for %s %s with %s weights and grid-factor %.1f"%(rand_it,patch,z_type,weight_str,grid_factor))

# First check that the simulation hasn't already been analyzed
bias_file_name = outdir+'patchy%d_%s_%s_%s_g%.1f_pk_q-bar_a_k%.3f_%.3f_%.3f.npy'%(rand_it,patch,z_type,weight_str,grid_factor,k_min,k_max,dk)
fish_file_name = outdir+'patchy%d_%s_%s_%s_g%.1f_pk_fish_a_k%.3f_%.3f_%.3f.npy'%(rand_it,patch,z_type,weight_str,grid_factor,k_min,k_max,dk)

if os.path.exists(bias_file_name) and os.path.exists(fish_file_name):
    print("Output already exists; exiting!\n")
    sys.exit()

### Load fiducial cosmology for co-ordinate conversions (in nbodykit)
cosmo_coord = cosmology.Cosmology(h=h_fid).match(Omega0_m = OmegaM_fid)

nbar_unif = 1e-4
# Load a uniform random sample for data and 50x randoms
data = UniformCatalog(nbar_unif,boxsize_grid)
data['NBAR'] = np.ones(len(data))
data['WEIGHT'] = np.ones(len(data))
data['WEIGHT_FKP'] = np.ones(len(data))
print("Generated %d data particles"%len(data))
randoms = UniformCatalog(nbar_unif*50,boxsize_grid)
randoms['NBAR'] = np.ones(len(randoms))
randoms['WEIGHT'] = np.ones(len(randoms))
randoms['WEIGHT_FKP'] = np.ones(len(randoms))
print("Generated %d random particles"%len(randoms))

# Assign to a grid
diff, density = grid_data(data, randoms, boxsize_grid,grid_3d,MAS='TSC',return_randoms=False,return_norm=False)

# Compute alpha rescaling and shot-noise factor
alpha_ran = np.sum(data['WEIGHT']).compute()/np.sum(randoms['WEIGHT']).compute()
shot_fac = (np.mean(data['WEIGHT']**2.).compute()+alpha_ran*np.mean(randoms['WEIGHT']**2.).compute())/np.mean(randoms['WEIGHT']).compute()
print("alpha = %.3f, shot_factor: %.3f"%(alpha_ran,shot_fac))

# Compute alpha for nbar rescaling
print("Computing Patchy alpha factor from weights")
if patch!='ngc' and ztype!='z1':
    raise Exception("NOT CONFIGURED FOR OTHER PATCHES!!")
data_w = load_data(1,ZMIN,ZMAX,cosmo_coord,weight_only=True).sum().compute()
rand_w = load_randoms(1,ZMIN,ZMAX,cosmo_coord,weight_only=True).sum().compute()
alpha_ran0 = data_w/rand_w
print("alpha_ran = %.3f"%alpha_ran0)

# Load pre-computed n(r) map (from mask and n(z), not discrete particles)
nbar = load_nbar(1, patch, z_type, ZMIN, ZMAX, grid_factor, alpha_ran0)

# Load grids in real and Fourier space
k_grids, r_grids = load_coord_grids(boxsize_grid, grid_3d, density)
k_norm = np.sqrt(np.sum(k_grids**2.,axis=0))
del density

############################## GRID DEFINITIONS ################################

# Cell volume
v_cell = 1.*boxsize_grid.prod()/(1.*grid_3d.prod())

# Load MAS grids, if appropriate
MAS_mat = load_MAS(boxsize_grid, grid_3d)

# Compute spherical harmonic fields in real and Fourier-space
Yk_lm, Yr_lm = compute_spherical_harmonics(lmax,k_grids,r_grids)

if wtype==1:
    # Load fit to Patchy P(k)
    pk_input = np.loadtxt(pk_input_file)
    fid_pk_interp = interp1d(pk_input[:,0],pk_input[:,1:].T)
    pk_map = fid_pk_interp(k_norm)

del r_grids, k_grids

# Compute k-space filters
k_filters = compute_filters(k_min,k_max,dk,k_norm)
n_k = len(k_filters)

############################# COVARIANCE FUNCTIONS #############################

### Define true random density
nbar_analyt = np.ones_like(diff)*nbar_unif

### True inverse covariance
def applyCinv_unif(input_map):
    """Apply true C^{-1} to the uniform map including MAS effects."""
    return ift(ft(ift(ft(input_map)*MAS_mat)/nbar_analyt)*MAS_mat)*v_cell/shot_fac

######################### COMPUTE FISHER + BIAS ################################

## Compute C^-1.x and A^-1.x
print("\n## Computing C^-1 of map assuming %s weightings"%weight_str)

if wtype==0:
    Cinv_diff = applyCinv_fkp(diff,nbar,MAS_mat,v_cell,shot_fac) # C^-1.x
else:
    Cinv_diff = applyCinv(diff,nbar,MAS_mat,pk_map,Yk_lm,Yr_lm,v_cell,shot_fac,rel_tol=1e-6,verb=1,max_it=50) # C^-1.x

Ainv_diff = applyCinv_unif(diff) # A^-1.x
del diff

# Compute N A^-1.x
N_Ainv_a = applyN(Ainv_diff,nbar,MAS_mat,v_cell,shot_fac)

### Compute C_a C^-1 x including MAS effects
print("## Computing C_a C^-1 of map assuming %s weightings\n"%weight_str)
C_a_Cinv_diff = applyC_alpha(Cinv_diff,nbar,MAS_mat,Yk_lm,Yr_lm,v_cell,k_filters,lmax)
C_a_Ainv_diff = applyC_alpha(Ainv_diff,nbar,MAS_mat,Yk_lm,Yr_lm,v_cell,k_filters,lmax)

del Cinv_diff, Ainv_diff

### Compute C^-1 C_a C^-1 x including MAS effects
print("## Computing C^-1 C_a C^-1 of map assuming %s weightings"%weight_str)
n_bins = len(C_a_Cinv_diff)
Cinv_C_a_Cinv_diff = []
for alpha in range(n_bins):

    if (alpha+1)%5==0: print("On bin %d of %d"%(alpha+1,n_bins))
    if wtype==0:
        tmp_map = applyCinv_fkp(C_a_Cinv_diff[alpha],nbar,MAS_mat,v_cell,shot_fac)
    else:
        tmp_map = applyCinv(C_a_Cinv_diff[alpha],nbar,MAS_mat,pk_map,Yk_lm,Yr_lm,v_cell,shot_fac,rel_tol=1e-4,verb=0,max_it=50) # C^-1.x

    Cinv_C_a_Cinv_diff.append(tmp_map)
    del tmp_map
del nbar, MAS_mat, Yk_lm, Yr_lm

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
