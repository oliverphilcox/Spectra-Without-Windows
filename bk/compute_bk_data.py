# compute_bk_data.py (Oliver Philcox, 2021)
### Compute the component g^a maps for the binned bispectrum of survey data with FKP or ML weightings
### These are then used to compute the full windowless bispectrum estimate
### Note compute_bk_randoms.py and compute_bk_fisher.py must be run before this script to compute the Fisher matrix

# Import modules
from nbodykit.lab import *
import sys, os, time, configparser
import numpy as np
from scipy.interpolate import interp1d
# custom definitions
sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/../src')
from opt_utilities import load_MAS, grid_data, load_coord_grids, compute_spherical_harmonic_functions, compute_filters, ft, ift, load_data, load_randoms, load_nbar

# Read command line arguments
if len(sys.argv)!=3:
    raise Exception("Need to specify simulation number and parameter file")
else:
    # Use sim_no = -1 if the simulation is unnumbered
    sim_no = int(sys.argv[1])
    paramfile = str(sys.argv[2]) # parameter file

########################### INPUT PARAMETERS ###########################

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
mcdir =  str(config['directories']['monte_carlo'])
outdir =  str(config['directories']['output'])

# Redshifts
ZMIN, ZMAX = float(config['sample']['z_min']), float(config['sample']['z_max'])

# Fiducial cosmological parameters
h_fid, OmegaM_fid = float(config['parameters']['h_fid']), float(config['parameters']['OmegaM_fid'])

# Survey dimensions
boxsize_grid = np.array(config.getlist('sample','box'),dtype=float)
grid_3d = np.array(config.getlist('bk-binning','grid'),dtype=int)

# Number of MC simulations
N_mc = int(config['settings']['N_mc'])

# Testing parameters
include_pix, rand_nbar, use_qbar = config.getboolean('settings','include_pix'), config.getboolean('settings','rand_nbar'), config.getboolean('settings','use_qbar')

if not use_qbar: print("CAUTION: Not including linear term in bispectrum estimator!\n")

if wtype==1:
    # Fiducial power spectrum input (for ML weights)
    pk_input_file = str(config['settings']['fiducial_pk'])

# Covariance matrix parameters
if wtype==1:
    lmax_pk = 4
    from covariances_pk import applyCinv
else:
    from covariances_pk import applyCinv_fkp

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

########################### LOAD DATA ###########################

# Check if simulation has already been analyzed
if sim_no!=-1:
    p_alpha_file_name = outdir + 'bk_%s%d_%s_N%d_k%.3f_%.3f_%.3f_l%d.txt'%(string,sim_no,weight_type,N_mc,k_min,k_max,dk,lmax)
else:
    p_alpha_file_name = outdir + 'bk_%s%s_N%d_k%.3f_%.3f_%.3f_l%d.txt'%(string,weight_type,N_mc,k_min,k_max,dk,lmax)
    

if os.path.exists(p_alpha_file_name):
    print("Simulation has already been computed; exiting!")
    sys.exit()

if sim_no!=-1:
    print("\n## Loading %s simulation %d with %s weights"%(string,sim_no,weight_type))
else:
    print("\n## Loading %s with %s weights"%(string,weight_type))

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
del data, randoms

# Load pre-computed n(r) map (from mask and n(z), not discrete particles)
print("Loading nbar from mask")
nbar_mask = load_nbar(config, 'bk', alpha_ran)

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

########################### GRID DEFINITIONS ###########################

if wtype==1:
    # Compute spherical harmonic fields in real and Fourier-space
    Y_lms = compute_spherical_harmonic_functions(np.maximum([lmax_pk,lmax]))

    # Load fit to P(k)
    pk_input = np.loadtxt(pk_input_file)
    fid_pk_interp = interp1d(pk_input[:,0],pk_input[:,1:].T)
    pk_map = fid_pk_interp(k_norm)[:lmax_pk//2+1]
else:
    # Load spherical harmonics
    Y_lms = compute_spherical_harmonic_functions(lmax)

# Compute k-space filters
k_filters = compute_filters(k_min,k_max,dk)
n_k = int((k_max-k_min)/dk)
n_l = lmax//2+1

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

########################### COMPUTE g_a ###########################

print("\n## Computing g-a maps assuming %s weightings"%weight_type)

# Compute C^-1 d
if wtype==0:
    Cinv_diff = applyCinv_fkp(diff,nbar_weight,MAS_mat,v_cell,shot_fac,include_pix=include_pix)
else:
    Cinv_diff = applyCinv(diff,nbar_weight,MAS_mat,pk_map,Y_lms,k_grids,r_grids,v_cell,shot_fac,rel_tol=1e-6,verb=1,max_it=30,include_pix=include_pix)
    del pk_map
del diff, nbar_weight

## Compute g^a_l functions
all_g_a = []
for l_i in range(0,n_l):
    this_g_a = []

    # Compute Sum_m Y_lm(k)FT[Y_lm(r)n(r)[C^-1d](r)]
    f_nx_l = 0.
    for m_i in range(len(Y_lms[l_i])):

        # First compute FT[nC^-1d * Y_lm(r)]
        if include_pix:
            ft_nCinv_d_lm = ft(ift(ft(Cinv_diff)/MAS_mat)*nbar*Y_lms[l_i][m_i](*r_grids))
        else:
            ft_nCinv_d_lm = ft(Cinv_diff*nbar*Y_lms[l_i][m_i](*r_grids))*MAS_mat

        # Now compute contribution to sum
        f_nx_l += ft_nCinv_d_lm*Y_lms[l_i][m_i](*k_grids)

    # Normalize
    f_nx = 4.*np.pi/(4.*l_i+1.)*f_nx_l

    # Apply k-binning
    for i in range(n_k):
        this_g_a.append(ift(k_filters(i,k_norm)*f_nx))
    all_g_a.append(this_g_a)
del f_nx, f_nx_l, ft_nCinv_d_lm, k_filters, k_norm, k_grids, r_grids, Y_lms
del Cinv_diff, nbar, MAS_mat

########################### COMPUTE q_alpha ###########################

print("\n## Computing q_alpha quantity in %d bins assuming %s weightings"%(n_bins,weight_type))

# Define Delta_alpha parameter
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

def bias_term(a,b,l_i):
    """Load bias term < g_a tilde-g_b^l > or < g_a^l tilde-g_b >. The ell is affixed to whichever of a or b is larger."""
    a0 = min([a,b])
    a1 = max([a,b])
    
    filename = mcdir+'%s%d_%s_bias_map%d,%d,%d_k%.3f_%.3f_%.3f_l%d.npz'%(string,N_mc,weight_type,a0,a1,l_i,k_min,k_max,dk,lmax)
    infile = np.load(filename)
    if infile['ct']!=N_mc: raise Exception("Wrong number of bias simulations computed! (%d of %d)"%(infile['ct'],N_mc))
    return infile['dat']

# Iterate over all possible triangles, creating q_alpha
q_alpha = []
for a in range(n_k):
    print("On primary k-bin %d of %d"%(a+1,n_k))
    g_a = all_g_a[0][a]
    for b in range(a,n_k):
        g_b = all_g_a[0][b]
        for c in range(b,n_k):
            # c is largest, by definition, so includes the Legendre polynomial
            if not test_bin(a,b,c): continue
            
            q_out = np.zeros(lmax//2+1)
            for l_i in range(lmax//2+1):
                gl_c = all_g_a[l_i][c]

                ## Analyze this bin
                if use_qbar:
                    tmp_q = np.sum(g_a*g_b*gl_c-g_a*bias_term(b,c,l_i)-g_b*bias_term(c,a,l_i)-gl_c*bias_term(a,b,0))
                else:
                    tmp_q = np.sum(g_a*g_b*gl_c)
                q_out[l_i] = np.real_if_close(tmp_q)
            q_alpha.append(np.real_if_close(q_out))

# Add symmetry factor
q_alpha = np.asarray(q_alpha)/Delta_abc[:,None]
q_alpha = np.concatenate([q_alpha[:,l_i] for l_i in range(lmax//2+1)])

########################### CONSTRUCT FISHER MATRIX ###########################

full_fish_file_name = outdir+'%s_mean%d_%s_full-fish_alpha_beta_k%.3f_%.3f_%.3f_l%d.npy'%(string,N_mc,weight_type,k_min,k_max,dk,lmax)

if not os.path.exists(full_fish_file_name):
    raise Exception("Fisher matrix has not been computed; exiting!")

print("\n## Loading Fisher matrix from file")
full_fisher = np.load(full_fish_file_name)

########################### COMPUTE p_alpha ###########################

print("\n## Computing p_alpha in %d bins assuming %s weightings"%(n_bins,weight_type))

# Create output
p_alpha = np.matmul(np.linalg.inv(full_fisher),q_alpha)

########################### SAVE & EXIT ###########################

with open(p_alpha_file_name,"w+") as output:
    if sim_no!=-1:
        output.write("####### Bispectrum of %s Simulation %d #############"%(string,sim_no))
    else:
        output.write("####### Bispectrum of %s #############"%(string))
    output.write("\n# Weights: %s"%weight_type)
    output.write("\n# Fiducial Omega_m: %.3f"%OmegaM_fid)
    output.write("\n# Fiducial h: %.3f"%h_fid)
    output.write("\n# Boxsize: [%.1f, %.1f, %.1f]"%(boxsize_grid[0],boxsize_grid[1],boxsize_grid[2]))
    output.write("\n# Grid: [%d, %d, %d]"%(grid_3d[0],grid_3d[1],grid_3d[2]))
    output.write("\n# k-binning: [%.3f, %.3f, %.3f]"%(k_min,k_max,dk))
    output.write("\n# l-max: %d"%lmax)
    output.write("\n# Monte Carlo Simulations: %d"%N_mc)
    output.write("\n#")
    if lmax==0:
        output.write("\n# Format: k1 | k2 | k3 | B0(k1,k2,k3)")
    elif lmax==2:
        output.write("\n# Format: k1 | k2 | k3 | B0(k1,k2,k3) | B2(k1,k2,k3)")
    elif lmax==4:
        output.write("\n# Format: k1 | k2 | k3 | B(k1,k2,k3) | B2(k1,k2,k3) | B4(k1,k2,k3)")
    else:
        output.write("\n# Format: k1 | k2 | k3 | B_{multipoles}(k1,k2,k3)")
    output.write("\n############################################")

    k_av = np.arange(k_min,k_max,dk)+dk/2.

    for i in range(n_bins//n_l):
        a,b,c = bins_index[i]
        output.write('\n%.4f\t%.4f\t%.4f'%(k_av[a],k_av[b],k_av[c]))
        for l_i in range(lmax//2+1):
            output.write('\t%.8e'%(p_alpha[l_i*(n_bins//n_l)+i]))

duration = time.time()-init
print("## Saved bispectrum estimates to %s. Exiting after %d seconds (%d minutes)\n"%(p_alpha_file_name,duration,duration//60))
sys.exit()
