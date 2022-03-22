# compute_pk_data.py (Oliver Philcox, 2021)
### Compute the power spectrum of survey data with FKP or ML weightings
### This computes the q_alpha term from data and combines it with the Fisher matrix to compute the full windowless power spectrum estimate
### Note that compute_pk_randoms.py must be run on N_mc sims before this script begins in order to compute Fisher matrices and bias terms

# Import modules
from nbodykit.lab import *
import numpy as np, sys, os, time, configparser
from scipy.interpolate import interp1d
# custom definitions
sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/../src')
from opt_utilities import load_data, load_randoms, load_MAS, load_nbar, grid_data, load_coord_grids, compute_spherical_harmonic_functions, compute_filters, ft, ift
from covariances_pk import applyC_alpha

# Read command line arguments
if len(sys.argv)!=3:
    raise Exception("Need to specify simulation number and parameter file")
else:
    # Use sim_no = -1 if the simulation is unnumbered
    sim_no = int(sys.argv[1])
    paramfile = str(sys.argv[2])

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

# Redshifts
ZMIN, ZMAX = float(config['sample']['z_min']), float(config['sample']['z_max'])

# Fiducial cosmological parameters
h_fid, OmegaM_fid = float(config['parameters']['h_fid']), float(config['parameters']['OmegaM_fid'])

# Survey dimensions
boxsize_grid = np.array(config.getlist('sample','box'),dtype=float)
grid_3d = np.array(config.getlist('pk-binning','grid'),dtype=int)

# Number of MC simulations
N_mc = int(config['settings']['N_mc'])

# Testing parameters
include_pix, rand_nbar = config.getboolean('settings','include_pix'), config.getboolean('settings','rand_nbar')

if wtype==1:
    # Fiducial power spectrum input (for ML weights)
    pk_input_file = str(config['settings']['fiducial_pk'])

# Summarize parameters
print("\n###################### PARAMETERS ######################\n")
if sim_no>=0:
    print("Simulation Number: %d"%sim_no)
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
print("\nN_mc: %d"%N_mc)
print("Output Directory: %s"%outdir)
print("\n########################################################")

init = time.time()

################################## LOAD DATA ###################################

# Check if simulation has already been analyzed
if sim_no!=-1:
    pk_file_name = outdir + 'pk_%s%d_%s_N%d_k%.3f_%.3f_%.3f_l%d.txt'%(string,sim_no,weight_type,N_mc,k_min,k_max,dk,lmax)
else:
    pk_file_name = outdir + 'pk_%s_N%d_k%.3f_%.3f_%.3f_l%d.txt'%(weight_type,N_mc,k_min,k_max,dk,lmax)
if os.path.exists(pk_file_name):
    print("Simulation has already been computed; exiting!")
    sys.exit()

# Check if relevant Fisher / bias simulations exist
bias_file_name = lambda rand_it: outdir+'%s%d_%s_pk_q-bar_a_k%.3f_%.3f_%.3f_l%d.npy'%(string,rand_it,weight_type,k_min,k_max,dk,lmax)
fish_file_name = lambda rand_it: outdir+'%s%d_%s_pk_fish_a_k%.3f_%.3f_%.3f_l%d.npy'%(string,rand_it,weight_type,k_min,k_max,dk,lmax)
combined_bias_file_name = outdir + 'bias_%s%d_%s_k%.3f_%.3f_%.3f_l%d.npy'%(string,N_mc,weight_type,k_min,k_max,dk,lmax)
combined_fish_file_name = outdir + 'fisher_%s%d_%s_k%.3f_%.3f_%.3f_l%d.npy'%(string,N_mc,weight_type,k_min,k_max,dk,lmax)

if not (os.path.exists(combined_bias_file_name) and os.path.exists(combined_fish_file_name)):
    for i in range(1,N_mc+1):
        if not os.path.exists(bias_file_name(i)):
            raise Exception("Bias term %d not found"%i)
        if not os.path.exists(fish_file_name(i)):
            raise Exception("Fisher matrix %d not found"%i)

# Start computation
if sim_no!=-1:
    print("\n## Analyzing %s simulation %d with %s weights"%(string, sim_no, weight_type))
else:
    print("\n## Analyzing %s with %s weights"%(string, weight_type))

### Load fiducial cosmology for co-ordinate conversions (in nbodykit)
cosmo_coord = cosmology.Cosmology(h=h_fid).match(Omega0_m = OmegaM_fid)

# Load data and paint to grid
data = load_data(sim_no,config,cosmo_coord,fkp_weights=False)
randoms = load_randoms(config,cosmo_coord,fkp_weights=False)
if rand_nbar:
    print("Loading nbar from random particles")
    diff, nbar_rand, density = grid_data(data, randoms, boxsize_grid,grid_3d,MAS='TSC',return_randoms=True,return_norm=False)
else:
    diff, density = grid_data(data, randoms, boxsize_grid,grid_3d,MAS='TSC',return_randoms=False,return_norm=False)

# Compute alpha rescaling and shot-noise factor
alpha_ran = data['WEIGHT'].sum().compute()/randoms['WEIGHT'].sum().compute()
shot_fac = ((data['WEIGHT']**2.).mean().compute()+alpha_ran*(randoms['WEIGHT']**2.).mean().compute())/randoms['WEIGHT'].mean().compute()
print("alpha = %.3f, shot_factor: %.3f"%(alpha_ran,shot_fac))

# Load pre-computed n(r) map (from mask and n(z), not discrete particles)
print("\nLoading nbar from mask")
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

############################ GRID DEFINITIONS ##################################

# Compute spherical harmonic functions
Y_lms = compute_spherical_harmonic_functions(lmax)

if wtype==1:
    # Load fit to P(k)
    pk_input = np.loadtxt(pk_input_file)
    fid_pk_interp = interp1d(pk_input[:,0],pk_input[:,1:].T)
    pk_map = fid_pk_interp(k_norm)[:lmax//2+1]

# Compute k-space filters
k_filters = compute_filters(k_min,k_max,dk)
n_k = int((k_max-k_min)/dk)

################################# COMPUTE q_alpha ##############################

## Compute C^-1[d]
print("\n## Computing C-inverse of data and associated computations assuming %s weightings\n"%weight_type)
if wtype==0:
    Cinv_diff = applyCinv_fkp(diff,nbar_weight,MAS_mat,v_cell,shot_fac,include_pix=include_pix) # C^-1.x
else:
    Cinv_diff = applyCinv(diff,nbar_weight,MAS_mat,pk_map,Y_lms,k_grids,r_grids,v_cell,shot_fac,rel_tol=1e-6,verb=1,max_it=30,include_pix=include_pix) # C^-1.x
    del pk_map
del diff, nbar_weight

## Now compute C_a C^-1 d including MAS effects
C_a_Cinv_diff = applyC_alpha(Cinv_diff,nbar,MAS_mat,Y_lms,k_grids,r_grids,v_cell,k_filters,k_norm,n_k,lmax,include_pix=include_pix,data=True)
del nbar, MAS_mat, Y_lms,k_grids,r_grids, k_norm

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
    print("Computed bias term from %d realizations and saved to %s"%(N_mc,combined_bias_file_name))

    # Save combined Fisher matrix
    np.save(combined_fish_file_name,fish)
    print("Computed Fisher matrix from %d realizations and saved to %s\n"%(N_mc,combined_fish_file_name))

###################### COMPUTE POWER SPECTRUM AND SAVE #########################

p_alpha = np.inner(np.linalg.inv(fish),q_alpha-bias)

with open(pk_file_name,"w+") as output:
    if sim_no==-1:
        output.write("####### Power Spectrum of %s #############"%string)
    else:
        output.write("####### Power Spectrum of %s Simulation %d #############"%(string,sim_no))
    output.write("\n# Weights: %s"%weight_type)
    output.write("\n# Fiducial Omega_m: %.3f"%OmegaM_fid)
    output.write("\n# Fiducial h: %.3f"%h_fid)
    output.write("\n# Forward-model pixellation : %d"%include_pix)
    output.write("\n# Random n-bar: %d"%rand_nbar)
    output.write("\n# Boxsize: [%.1f, %.1f, %.1f]"%(boxsize_grid[0],boxsize_grid[1],boxsize_grid[2]))
    output.write("\n# Grid: [%d, %d, %d]"%(grid_3d[0],grid_3d[1],grid_3d[2]))
    output.write("\n# k-binning: [%.3f, %.3f, %.3f]"%(k_min,k_max,dk))
    output.write("\n# l-max: %d"%lmax)
    output.write("\n# Monte Carlo Simulations: %d"%N_mc)
    output.write("\n#")
    if lmax==0:
        output.write("\n# Format: k | P0(k)")
    elif lmax==2:
        output.write("\n# Format: k | P0(k) | P2(k)")
    elif lmax==4:
        output.write("\n# Format: k | P0(k) | P2(k) | P4(k)")
    else:
        output.write("\n# Format: k | P_{multipoles}(k)")
    output.write("\n############################################")

    for i in range(n_k):
        output.write('\n%.4f'%(k_min+(i+0.5)*dk))
        for l_i in range(lmax//2+1):
            output.write('\t%.8e'%(p_alpha[i+l_i*n_k]))

####################################### EXIT ###################################

duration = time.time()-init
print("## Saved output to %s. Exiting after %d seconds (%d minutes)\n"%(pk_file_name,duration,duration//60))
sys.exit()