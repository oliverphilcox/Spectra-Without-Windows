# compute_pk_data.py (Oliver Philcox, 2021)
### Compute the power spectrum of BOSS or Patchy data with FKP or ML weightings
### This computes the q_alpha term from data and combines it with the Fisher matrix to compute the full windowless power spectrum estimate
### Note that compute_pk_randoms.py must be run on N_bias sims before this script begins in order to compute Fisher matrices and bias terms
### If the sim-no parameter is set to -1, this will compute the power spectrum of BOSS data

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
if len(sys.argv)!=4:
    raise Exception("Need to specify sim number, weight-type and grid factor!")
else:
    # If sim no = -1 the true BOSS data is used
    sim_no = int(sys.argv[1])
    wtype = int(sys.argv[2]) # 0 for FKP, 1 for ML
    grid_factor = int(sys.argv[3])

############################### INPUT PARAMETERS ###############################

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
    grid_3d = np.asarray([252,460,260],dtype=int)/grid_factor

# box dimensions (scaled from BOSS release)
if patch=='ngc' and z_type=='z1':
    boxsize_grid = np.array([1350,2450,1400])
    grid_3d = np.asarray([252,460,260],dtype=int)/grid_factor
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
print("Grid-Factor: %d"%grid_factor)
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

################################## LOAD DATA ###################################

if sim_no!=-1:
    print("\n## Analyzing %s %s simulation %d with %s weights and grid-factor %d"%(patch,z_type,sim_no,weight_str,grid_factor))
else:
    print("\n## Analyzing %s %s BOSS data with %s weights and grid-factor %d"%(patch,z_type,weight_str,grid_factor))

### Load fiducial cosmology for co-ordinate conversions (in nbodykit)
cosmo_coord = cosmology.Cosmology(h=h_fid).match(Omega0_m = OmegaM_fid)

# Load data and paint to grid
data = load_data(sim_no,ZMIN,ZMAX,cosmo_coord,fkp_weights=False);
if patch!='ngc' or ztype!='z1':
    raise Exception("NOT CONFIGURED FOR OTHER PATCHES!!")
randoms = load_randoms(sim_no,ZMIN,ZMAX,cosmo_coord,fkp_weights=False);
diff, density = grid_data(data, randoms, boxsize_grid,grid_3d,MAS='TSC',return_randoms=False,return_norm=False)

# Compute alpha rescaling and shot-noise factor
alpha_ran = np.sum(data['WEIGHT']).compute()/np.sum(randoms['WEIGHT']).compute()
shot_fac = (np.mean(data['WEIGHT']**2.).compute()+alpha_ran*np.mean(randoms['WEIGHT']**2.).compute())/np.mean(randoms['WEIGHT']).compute()
print("alpha = %.3f, shot_factor: %.3f"%(alpha_ran,shot_fac))
del data, randoms

# Load pre-computed n(r) map (from mask and n(z), not discrete particles)
nbar = load_nbar(sim_no, patch, z_type, ZMIN, ZMAX, grid_factor, alpha_ran)

# Load grids in real and Fourier space
k_grids, r_grids = load_coord_grids(boxsize_grid, grid_3d, density)
k_norm = np.sqrt(np.sum(k_grids**2.,axis=0))
del density

############################ GRID DEFINITIONS ##################################

# Cell volume
v_cell = 1.*boxsize_grid.prod()/(1.*grid_3d.prod())

# Load MAS grids
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

################################# COMPUTE q_alpha ##############################

## Compute C^-1[d]
print("\n## Computing C-inverse of data and associated computations assuming %s weightings\n"%weight_str)

if wtype==0:
    Cinv_diff = applyCinv_fkp(diff,nbar,MAS_mat,v_cell,shot_fac) # C^-1.x
else:
    Cinv_diff = applyCinv(diff,nbar,MAS_mat,pk_map,Yk_lm,Yr_lm,v_cell,shot_fac,rel_tol=1e-6,verb=1,max_it=50) # C^-1.x
    del pk_map

del diff

## Now compute C_a C^-1 d including MAS effects
C_a_Cinv_diff = applyC_alpha(Cinv_diff,nbar,MAS_mat,Yk_lm,Yr_lm,v_cell,k_filters,lmax)
del nbar, MAS_mat, Yk_lm, Yr_lm

## Compute q_a
q_a = np.zeros(len(C_a_Cinv_diff))

for i in range(len(C_a_Cinv_diff)):
    q_a[i] = np.real_if_close(np.sum(Cinv_diff*C_a_Cinv_diff[i]))/2.

############################## LOAD FISHER MATRIX ##############################

### Define file names
bias_file_name = outdir+'patchy%d_%s_%s_%s_g%d_pk_q-bar_a_k%.3f_%.3f_%.3f.npy'%(bias_sim,patch,z_type,weight_str,grid_factor,k_min,k_max,dk)
fish_file_name = outdir+'patchy%d_%s_%s_%s_g%d_pk_fish_a_k%.3f_%.3f_%.3f.npy'%(bias_sim,patch,z_type,weight_str,grid_factor,k_min,k_max,dk)
if sim_no!=-1:
    pk_file_name = out_dir + 'pk_patchy%d_%s_%s_%s_g%d_k%.3f_%.3f_%.3f.npy'%(sim_no,patch,z_type,weight_str,grid_factor,k_min,k_max,dk)
else:
    pk_file_name = out_dir + 'pk_boss_%s_%s_%s_g%d_k%.3f_%.3f_%.3f.npy'%(patch,z_type,weight_str,grid_factor,k_min,k_max,dk)
combined_bias_file_name = out_dir + 'bias_patchy%d_%s_%s_%s_g%d_k%.3f_%.3f_%.3f.npy'%(N_bias,patch,z_type,weight_str,grid_factor,k_min,k_max,dk)
combined_fish_file_name = out_dir + 'fisher_patchy%d_%s_%s_%s_g%d_k%.3f_%.3f_%.3f.npy'%(N_bias,patch,z_type,weight_str,grid_factor,k_min,k_max,dk)

## First load in Fisher matrix and bias term
try:
    bias = np.load(combined_fish_file_name)
    fish = np.load(combined_fish_file_name)
except IOError:
    print("Loading bias term and Fisher matrix from uniform simulations")

    fish = 0.
    bias = 0.
    for i in range(1,N_bias+1):
        try:
            fish += np.load(fish_file_name(i))
        except IOError:
            raise Exception("Fisher matrix %d not found!"%i)
        try:
            bias += np.load(bias_file_name(i))
        except IOError:
            raise Exception("Bias term %d not found!"%i)
    fish /= N_bias
    bias /= N_bias

    # Save combined bias term
    np.save(combined_bias_file_name,bias)
    print("Computed bias term from %d simulations and saved to %s\n"%(N_bias,combined_bias_file_name))

    # Save combined Fisher matrix
    np.save(combined_fish_file_name,fish)
    print("Computed Fisher matrix from %d simulations and saved to %s\n"%(N_bias,combined_fish_file_name))

###################### COMPUTE POWER SPECTRUM AND SAVE #########################

p_alpha = np.matmul(np.linalg.inv(fish),q_alpha-bias)

## Save output
if sim_no==-1:
    file_name = outdir+'boss_%s_%s_%s_g%d_pk_q_a_k%.3f_%.3f_%.3f.npy'%(patch,z_type,weight_str,grid_factor,k_min,k_max,dk)
else:
    file_name = outdir+'patchy%d_%s_%s_%s_g%d_pk_q_a_k%.3f_%.3f_%.3f.npy'%(sim_no,patch,z_type,weight_str,grid_factor,k_min,k_max,dk)
np.save(file_name,q_a)

with open(pk_file_name,"w+") as output:
    if sim_no==-1:
        output.write("####### Power Spectrum of BOSS #############")
    else:
        output.write("####### Power Spectrum of Patchy Simulation %d #############"%sim_no)
    output.write("\n# Patch: %s"%patch)
    output.write("\n# z-type: %s"%z_type)
    output.write("\n# Weights: %s"%weight_str)
    output.write("\n# Fiducial Omega_m: %.3f"%Omfid)
    output.write("\n# Fiducial h: %.3f"%hfid)
    output.write("\n# Boxsize: [%.1f, %.1f, %.1f]"%(boxsize_grid[0],boxsize_grid[1],boxsize_grid[2]))
    output.write("\n# Grid: [%d, %d, %d]"%(grid_3d[0],grid_3d[1],grid_3d[2]))
    output.write("\n# k-binning: [%.3f, %.3f, %.3f]"%(k_min,k_cut,dk))
    output.write("\n#")
    output.write("\n# Format: k | P0 | P2 | P4")
    output.write("\n############################################")

    for i in range(len(k_good)):
        output.write('\n%.4f\t%.8e\t%.8e\t%.8e'%(k_good[i],pk[i,0],pk[i,1],pk[i,2]))

####################################### EXIT ###################################

duration = time.time()-init
print("## Saved output to %s. Exiting after %d seconds (%d minutes)\n\n"%(pk_file_name,duration,duration//60))
sys.exit()
