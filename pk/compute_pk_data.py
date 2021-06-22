# compute_pk_data.py (Oliver Philcox, 2021)
### Compute the power spectrum of BOSS or Patchy data with FKP or ML weightings
### This computes the q_alpha term from data and combines it with the Fisher matrix to compute the full windowless power spectrum estimate
### Note that compute_pk_randoms.py must be run on N_mc sims before this script begins in order to compute Fisher matrices and bias terms
### If the sim-no parameter is set to -1, this will compute the power spectrum of BOSS data, using a BOSS-specific Fisher matrix

# Import modules
from nbodykit.lab import *
import sys, os, copy, time, pyfftw
import numpy as np
from scipy.interpolate import interp1d
# custom definitions
sys.path.append('../src')
from opt_utilities import load_data, load_randoms, load_MAS, load_nbar, grid_data, load_coord_grids, compute_spherical_harmonics, compute_filters, ft, ift, plotter
from covariances_pk import applyC_alpha

# Read command line arguments
if len(sys.argv)!=6:
    raise Exception("Need to specify sim number, patch, z-type, weight-type and grid factor!")
else:
    # If sim no = -1 the true BOSS data is used
    sim_no = int(sys.argv[1])
    patch = str(sys.argv[2]) # ngc or sgc
    z_type = str(sys.argv[3]) # z1 or z3
    wtype = int(sys.argv[4]) # 0 for FKP, 1 for ML
    grid_factor = float(sys.argv[5])

############################### INPUT PARAMETERS ###############################

## k-space binning
k_min = 0.0
k_max = 0.26
dk = 0.005
lmax = 4

## Cosmological parameters for co-ordinate conversions
h_fid = 0.676
OmegaM_fid = 0.31

# Number of Monte Carlo sims used
<<<<<<< HEAD
N_mc = 50

# Whether to forward-model pixellation effects.
include_pix = True
# If true, use nbar(r) from the random particles instead of the mask / n(z) distribution.
rand_nbar = True

# Directories
outdir = '/projects/QUIJOTE/Oliver/pk_opt_patchy/' # to hold output Fisher matrices and power spectra
=======
N_mc = 100

# Whether to forward-model pixellation effects.
include_pix = False
# If true, use nbar(r) from the random particles instead of the mask / n(z) distribution.
rand_nbar = False

# Directories
outdir = '/projects/QUIJOTE/Oliver/pk_opt_patchy5/' # to hold output Fisher matrices and power spectra
>>>>>>> c3992fbb0e5f95d0f96bd056cf1bfa0995eb7218

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
if sim_no==-1:
    print("BOSS Data")
else:
    print("Simulation: %d"%sim_no)
print("Grid-Factor: %.1f"%grid_factor)
print("Weight-Type: %s"%weight_str)
print("\nPatch: %s"%patch)
print("Redshift-type: %s"%z_type)
if rand_nbar:
    print("n-bar: from randoms")
else:
    print("n-bar: from mask")
print("Forward model pixellation: %d"%include_pix)
print("\nk-min: %.3f"%k_min)
print("k-max: %.3f"%k_max)
print("dk: %.3f"%dk)
print("\nFiducial h = %.3f"%h_fid)
print("Fiducial Omega_m = %.3f"%OmegaM_fid)
print("\nN_mc: %d"%N_mc)
print("Output Directory: %s"%outdir)
print("\n########################################################")

init = time.time()

################################## LOAD DATA ###################################

# Check if simulation has already been analyzed
<<<<<<< HEAD
if sim_no!=-1:
    pk_file_name = outdir + 'pk_patchy%d_%s_%s_%s_N%d_g%.1f_k%.3f_%.3f_%.3f.npy'%(sim_no,patch,z_type,weight_str,N_mc,grid_factor,k_min,k_max,dk)
else:
    pk_file_name = outdir + 'pk_boss_%s_%s_%s_N%d_g%.1f_k%.3f_%.3f_%.3f.npy'%(patch,z_type,weight_str,N_mc,grid_factor,k_min,k_max,dk)
if os.path.exists(pk_file_name):
    print("Simulation has already been computed; exiting!")
    sys.exit()

# Check if relevant Fisher / bias simulations exist
if sim_no==-1:
    root = 'boss'
else:
    root = 'patchy'

bias_file_name = lambda bias_sim: outdir+'%s%d_%s_%s_%s_g%.1f_pk_q-bar_a_k%.3f_%.3f_%.3f.npy'%(root,bias_sim,patch,z_type,weight_str,grid_factor,k_min,k_max,dk)
fish_file_name = lambda bias_sim: outdir+'%s%d_%s_%s_%s_g%.1f_pk_fish_a_k%.3f_%.3f_%.3f.npy'%(root,bias_sim,patch,z_type,weight_str,grid_factor,k_min,k_max,dk)
combined_bias_file_name = outdir + 'bias_%s%d_%s_%s_%s_g%.1f_k%.3f_%.3f_%.3f.npy'%(root,N_mc,patch,z_type,weight_str,grid_factor,k_min,k_max,dk)
combined_fish_file_name = outdir + 'fisher_%s%d_%s_%s_%s_g%.1f_k%.3f_%.3f_%.3f.npy'%(root,N_mc,patch,z_type,weight_str,grid_factor,k_min,k_max,dk)

if not (os.path.exists(combined_bias_file_name) and os.path.exists(combined_fish_file_name)):
    for i in range(1,N_mc+1):
        if not os.path.exists(bias_file_name(i)):
            raise Exception("Bias term %d not found"%i)
        if not os.path.exists(fish_file_name(i)):
            raise Exception("Fisher matrix %d not found"%i)

# Start computation
if sim_no!=-1:
    print("\n## Analyzing %s %s simulation %d with %s weights and grid-factor %.1f"%(patch,z_type,sim_no,weight_str,grid_factor))
else:
=======
if sim_no!=-1:
    pk_file_name = outdir + 'pk_patchy%d_%s_%s_%s_N%d_g%.1f_k%.3f_%.3f_%.3f.npy'%(sim_no,patch,z_type,weight_str,N_mc,grid_factor,k_min,k_max,dk)
else:
    pk_file_name = outdir + 'pk_boss_%s_%s_%s_N%d_g%.1f_k%.3f_%.3f_%.3f.npy'%(patch,z_type,weight_str,N_mc,grid_factor,k_min,k_max,dk)
if os.path.exists(pk_file_name):
    print("Simulation has already been computed; exiting!")
    sys.exit()

# Check if relevant Fisher / bias simulations exist
if sim_no==-1:
    root = 'boss'
else:
    root = 'patchy'

bias_file_name = lambda bias_sim: outdir+'%s%d_%s_%s_%s_g%.1f_pk_q-bar_a_k%.3f_%.3f_%.3f.npy'%(root,bias_sim,patch,z_type,weight_str,grid_factor,k_min,k_max,dk)
fish_file_name = lambda bias_sim: outdir+'%s%d_%s_%s_%s_g%.1f_pk_fish_a_k%.3f_%.3f_%.3f.npy'%(root,bias_sim,patch,z_type,weight_str,grid_factor,k_min,k_max,dk)
combined_bias_file_name = outdir + 'bias_%s%d_%s_%s_%s_g%.1f_k%.3f_%.3f_%.3f.npy'%(root,N_mc,patch,z_type,weight_str,grid_factor,k_min,k_max,dk)
combined_fish_file_name = outdir + 'fisher_%s%d_%s_%s_%s_g%.1f_k%.3f_%.3f_%.3f.npy'%(root,N_mc,patch,z_type,weight_str,grid_factor,k_min,k_max,dk)

if not (os.path.exists(combined_bias_file_name) and os.path.exists(combined_fish_file_name)):
    for i in range(1,N_mc+1):
        if not os.path.exists(bias_file_name(i)):
            raise Exception("Bias term %d not found"%i)
        if not os.path.exists(fish_file_name(i)):
            raise Exception("Fisher matrix %d not found"%i)

# Start computation
if sim_no!=-1:
    print("\n## Analyzing %s %s simulation %d with %s weights and grid-factor %.1f"%(patch,z_type,sim_no,weight_str,grid_factor))
else:
>>>>>>> c3992fbb0e5f95d0f96bd056cf1bfa0995eb7218
    print("\n## Analyzing %s %s BOSS data with %s weights and grid-factor %.1f"%(patch,z_type,weight_str,grid_factor))

### Load fiducial cosmology for co-ordinate conversions (in nbodykit)
cosmo_coord = cosmology.Cosmology(h=h_fid).match(Omega0_m = OmegaM_fid)

# Load data and paint to grid
data = load_data(sim_no,ZMIN,ZMAX,cosmo_coord,patch=patch,fkp_weights=False);
randoms = load_randoms(sim_no,ZMIN,ZMAX,cosmo_coord,patch=patch,fkp_weights=False);
if rand_nbar:
<<<<<<< HEAD
    print("Loading nbar from random particles")
=======
>>>>>>> c3992fbb0e5f95d0f96bd056cf1bfa0995eb7218
    diff, nbar_rand, density = grid_data(data, randoms, boxsize_grid,grid_3d,MAS='TSC',return_randoms=True,return_norm=False)
else:
    diff, density = grid_data(data, randoms, boxsize_grid,grid_3d,MAS='TSC',return_randoms=False,return_norm=False)

# Compute alpha rescaling and shot-noise factor
alpha_ran = np.sum(data['WEIGHT']).compute()/np.sum(randoms['WEIGHT']).compute()
shot_fac = (np.mean(data['WEIGHT']**2.).compute()+alpha_ran*np.mean(randoms['WEIGHT']**2.).compute())/np.mean(randoms['WEIGHT']).compute()
norm = 1./np.asarray(alpha_ran*np.sum(randoms['NBAR']*randoms['WEIGHT']*randoms['WEIGHT_FKP']**2.))
print("alpha = %.3f, shot_factor: %.3f"%(alpha_ran,shot_fac))
del data, randoms

# Load pre-computed n(r) map (from mask and n(z), not discrete particles)
print("Loading nbar from mask")
<<<<<<< HEAD
nbar_mask = load_nbar(sim_no, patch, z_type, ZMIN, ZMAX, grid_factor, alpha_ran)
=======
nbar_mask = load_nbar(sim_no, patch, z_type, ZMIN, ZMAX, grid_factor, alpha_ran, hr=True)
>>>>>>> c3992fbb0e5f95d0f96bd056cf1bfa0995eb7218

# Load grids in real and Fourier space
k_grids, r_grids = load_coord_grids(boxsize_grid, grid_3d, density)
k_norm = np.sqrt(np.sum(k_grids**2.,axis=0))
del density

# Load MAS grids
MAS_mat = load_MAS(boxsize_grid, grid_3d)

# For weightings, we should use a smooth nbar always.
<<<<<<< HEAD
nbar_weight = nbar_mask.copy()
=======
#nbar_weight = nbar_mask.copy()
nbar_weight = load_nbar(sim_no, patch, z_type, ZMIN, ZMAX, grid_factor, alpha_ran, z_only=True)
>>>>>>> c3992fbb0e5f95d0f96bd056cf1bfa0995eb7218
if rand_nbar:
    nbar = nbar_rand.copy()
    del nbar_rand
else:
    nbar = nbar_mask.copy()
del nbar_mask

############################ GRID DEFINITIONS ##################################

# Cell volume
v_cell = 1.*boxsize_grid.prod()/(1.*grid_3d.prod())

<<<<<<< HEAD
# Compute renormalization factor (not currently used)
rescale_fac = 1./np.sqrt(np.sum(nbar**2)*v_cell*norm)
print("Rescale factor: %.4e"%rescale_fac)
=======
# Compute renormalization factor
rescale_fac = 1./np.sqrt(np.sum(nbar**2)*v_cell*norm)
print("Rescale factor: %.4e"%rescale_fac)
np.save(outdir+'sim%d_g%d_rescale.npy'%(sim_no,grid_factor),rescale_fac)
>>>>>>> c3992fbb0e5f95d0f96bd056cf1bfa0995eb7218

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

################################# COMPUTE q_alpha ##############################

## Compute C^-1[d]
print("\n## Computing C-inverse of data and associated computations assuming %s weightings\n"%weight_str)
if wtype==0:
<<<<<<< HEAD
    Cinv_diff = applyCinv_fkp(diff,nbar_weight,MAS_mat,v_cell,shot_fac,include_pix=include_pix) # C^-1.x
else:
    Cinv_diff = applyCinv(diff,nbar_weight,MAS_mat,pk_map,Yk_lm,Yr_lm,v_cell,shot_fac,rel_tol=1e-6,verb=1,max_it=30,include_pix=include_pix) # C^-1.x
=======
    Cinv_diff = applyCinv_fkp(diff,nbar_weight,MAS_mat,v_cell,shot_fac,use_MAS=include_pix) # C^-1.x
else:
    Cinv_diff = applyCinv(diff,nbar_weight,MAS_mat,pk_map,Yk_lm,Yr_lm,v_cell,shot_fac,rel_tol=1e-6,verb=1,max_it=50,use_MAS=include_pix) # C^-1.x
>>>>>>> c3992fbb0e5f95d0f96bd056cf1bfa0995eb7218
    del pk_map
del diff, nbar_weight

## Now compute C_a C^-1 d including MAS effects
<<<<<<< HEAD
C_a_Cinv_diff = applyC_alpha(Cinv_diff,nbar,MAS_mat,Yk_lm,Yr_lm,v_cell,k_filters,lmax,include_pix=include_pix,data=True)
=======
C_a_Cinv_diff = applyC_alpha(Cinv_diff,nbar,MAS_mat,Yk_lm,Yr_lm,v_cell,k_filters,lmax,use_MAS=include_pix,data=True)
>>>>>>> c3992fbb0e5f95d0f96bd056cf1bfa0995eb7218
del nbar, MAS_mat, Yk_lm, Yr_lm

## Compute q_alpha
q_alpha = np.zeros(len(C_a_Cinv_diff))

for i in range(len(C_a_Cinv_diff)):
    q_alpha[i] = np.real_if_close(np.sum(Cinv_diff*C_a_Cinv_diff[i]))/2.

############################## LOAD FISHER MATRIX ##############################

## First load in Fisher matrix and bias term
try:
    bias = np.load(combined_bias_file_name)
    fish = np.load(combined_fish_file_name)
except IOError:
    print("Loading bias term and Fisher matrix from uniform simulations")

    fish = 0.
    bias = 0.
    for i in range(1,N_mc+1):
        fish += np.load(fish_file_name(i))
        bias += np.load(bias_file_name(i))
    fish /= N_mc
    bias /= N_mc

    # Save combined bias term
    np.save(combined_bias_file_name,bias)
    print("Computed bias term from %d simulations and saved to %s\n"%(N_mc,combined_bias_file_name))

    # Save combined Fisher matrix
    np.save(combined_fish_file_name,fish)
    print("Computed Fisher matrix from %d simulations and saved to %s\n"%(N_mc,combined_fish_file_name))

###################### COMPUTE POWER SPECTRUM AND SAVE #########################

p_alpha = np.inner(np.linalg.inv(fish),q_alpha-bias)

with open(pk_file_name,"w+") as output:
    if sim_no==-1:
        output.write("####### Power Spectrum of BOSS #############")
    else:
        output.write("####### Power Spectrum of Patchy Simulation %d #############"%sim_no)
    output.write("\n# Patch: %s"%patch)
    output.write("\n# z-type: %s"%z_type)
    output.write("\n# Weights: %s"%weight_str)
    output.write("\n# Fiducial Omega_m: %.3f"%OmegaM_fid)
    output.write("\n# Fiducial h: %.3f"%h_fid)
    output.write("\n# Forward-model pixellation : %d"%include_pix)
    output.write("\n# Rando n-bar: %d"%rand_nbar)
    output.write("\n# Boxsize: [%.1f, %.1f, %.1f]"%(boxsize_grid[0],boxsize_grid[1],boxsize_grid[2]))
    output.write("\n# Grid: [%d, %d, %d]"%(grid_3d[0],grid_3d[1],grid_3d[2]))
    output.write("\n# k-binning: [%.3f, %.3f, %.3f]"%(k_min,k_max,dk))
    output.write("\n# Monte Carlo Simulations: %d"%N_mc)
    output.write("\n#")
    output.write("\n# Format: k | P0 | P2 | P4")
    output.write("\n############################################")

    for i in range(n_k):
        output.write('\n%.4f\t%.8e\t%.8e\t%.8e'%(k_min+(i+0.5)*dk,p_alpha[i],p_alpha[i+n_k],p_alpha[i+2*n_k]))

####################################### EXIT ###################################

duration = time.time()-init
print("## Saved output to %s. Exiting after %d seconds (%d minutes)\n\n"%(pk_file_name,duration,duration//60))
sys.exit()
