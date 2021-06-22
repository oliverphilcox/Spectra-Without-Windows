# compute_bk_randoms.py (Oliver Philcox, 2021)
### Compute the component g^a maps for the binned bispectrum of uniformly distributed random particles.
### These are then used to compute phi_alpha and C^-1 phi_alpha maps and added to Monte Carlo averages.
### Note that we assume a Patchy geometry always here.

# Import modules
from nbodykit.lab import *
import sys, os, copy, time, pyfftw, shutil, fasteners, numpy as np
from scipy.interpolate import interp1d
# custom definitions
sys.path.append('../src')
from opt_utilities import load_data, load_randoms, load_MAS, load_nbar, grid_data, grid_uniforms, load_coord_grids, compute_spherical_harmonics, compute_filters, ft, ift, plotter

# Read command line arguments
if len(sys.argv)!=6:
    raise Exception("Need to specify random iteration, patch, z-type, weight-type and grid factor!")
else:
    rand_it = int(sys.argv[1])
    patch = str(sys.argv[2]) # ngc or sgc
    z_type = str(sys.argv[3]) # z1 or z3
    wtype = int(sys.argv[4]) # 0 for FKP, 1 for ML
    grid_factor = float(sys.argv[5])

############################### INPUT PARAMETERS ###############################

## Number of Monte Carlo simulations used
N_mc = 50

## k-space binning
k_min = 0.00
k_max = 0.16
dk = 0.01

## Cosmological parameters for co-ordinate conversions
h_fid = 0.676
OmegaM_fid = 0.31

# Whether to forward-model pixellation effects.
include_pix = False
# If true, use nbar(r) from the random particles instead of the mask / n(z) distribution.
rand_nbar = True

## Directories
tmpdir = '/tmp/phi_alpha%d_%.1f/'%(rand_it,grid_factor) # to hold temporary output (should be large)
mcdir = '/projects/QUIJOTE/Oliver/bk_opt_production5a/summed_phi_alpha/' # to hold intermediate sums (should be large)

if wtype==1:
    # Fiducial power spectrum input (for ML weights)
    pk_input_file = '/projects/QUIJOTE/Oliver/bk_opt/patchy_%s_%s_pk_fid_k_0.00_0.30.txt'%(patch,z_type)

#### In principle, nothing below here needs to be altered for BOSS

if rand_it>N_mc: raise Exception("Simulation number cannot be greater than number of bias sims!")

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
if not os.path.exists(mcdir): os.makedirs(mcdir)

# Covariance matrix parameters
if wtype==1:
    lmax = 4
    weight_str = 'ml'
    from covariances_pk import applyCinv
elif wtype==0:
    weight_str = 'fkp'
    from covariances_pk import applyCinv_fkp
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
    print("n-bar: from randoms")
else:
    print("n-bar: from mask")
print("Forward model pixellation: %d"%include_pix)
print("\nk-min: %.3f"%k_min)
print("k-max: %.3f"%k_max)
print("dk: %.3f"%dk)
print("\nFiducial h = %.3f"%h_fid)
print("Fiducial Omega_m = %.3f"%OmegaM_fid)
print("\nMonte Carlo Directory: %s"%mcdir)
print("Temporary Directory: %s"%tmpdir)
print("\n########################################################")

init = time.time()

################################### LOAD DATA ##################################

### First check if we actually need to compute this simulation
fish_file_name = mcdir+'patchy_unif%d_%s_%s_%s_g%.1f_fish_alpha_beta_k%.3f_%.3f_%.3f.npy'%(rand_it,patch,z_type,weight_str,grid_factor,k_min,k_max,dk)
if os.path.exists(fish_file_name):
    print("Simulation already completed!")
    sys.exit();

print("\n## Loading random iteration %d for %s %s with %s weights and grid-factor %.1f"%(rand_it,patch,z_type,weight_str,grid_factor))

# Clean any crud from a previous run
if os.path.exists(tmpdir): shutil.rmtree(tmpdir)
os.makedirs(tmpdir)

### Load fiducial cosmology for co-ordinate conversions (in nbodykit)
cosmo_coord = cosmology.Cosmology(h=h_fid).match(Omega0_m = OmegaM_fid)

nbar_unif = 1e-3
# Load a uniform random sample for data
data = UniformCatalog(nbar_unif,boxsize_grid,seed=rand_it)
print("Created %d uniform randoms"%len(data))

# Assign to a grid
print("Assigning to grid")
diff = grid_uniforms(data, nbar_unif, boxsize_grid,grid_3d,MAS='TSC')
shot_fac_unif = 1.
del data

# Compute alpha for nbar rescaling
print("Loading data")
data_true = load_data(1,ZMIN,ZMAX,cosmo_coord,patch=patch,fkp_weights=False)
rand_true = load_randoms(1,ZMIN,ZMAX,cosmo_coord,patch=patch,fkp_weights=False)
alpha_ran = (data_true['WEIGHT'].sum()/rand_true['WEIGHT'].sum()).compute()
shot_fac = ((data_true['WEIGHT']**2.).mean().compute()+alpha_ran*(rand_true['WEIGHT']**2.).mean().compute())/rand_true['WEIGHT'].mean().compute()
norm = 1./np.asarray(alpha_ran*(rand_true['NBAR']*rand_true['WEIGHT']*rand_true['WEIGHT_FKP']**2.).mean().compute())
print("Data: alpha_ran = %.3f, shot_factor: %.3f"%(alpha_ran,shot_fac))

if rand_nbar:
    print("Loading nbar from random particles")
    nbar_rand, density = grid_data(data_true, rand_true, boxsize_grid,grid_3d,MAS='TSC',return_randoms=True,return_norm=False)[1:]
else:
    # load density mesh (used to define coordinate arrays)
    density = grid_data(data_true, rand_true, boxsize_grid,grid_3d,MAS='TSC',return_randoms=False,return_norm=False)[1]

del rand_true, data_true

# Load pre-computed n(r) map (from mask and n(z), not discrete particles)
print("Loading nbar from mask")
nbar_mask = load_nbar(1, patch, z_type, ZMIN, ZMAX, grid_factor, alpha_ran, z_only=True)

# Load grids in real and Fourier space
k_grids, r_grids = load_coord_grids(boxsize_grid, grid_3d, density)
k_norm = np.sqrt(np.sum(k_grids**2.,axis=0))
del density

# For weightings, we should use a smooth nbar always.
nbar_weight = nbar_mask.copy()
if rand_nbar:
    nbar = nbar_rand.copy()
    del nbar_rand
else:
    nbar = nbar_mask.copy()
del nbar_mask

############################ GRID DEFINITIONS ##################################

# Cell volume
v_cell = 1.*boxsize_grid.prod()/(1.*grid_3d.prod())

# Load MAS grids
MAS_mat = load_MAS(boxsize_grid, grid_3d)

if wtype==1:
    # Compute spherical harmonic fields in real and Fourier-space
    Yk_lm, Yr_lm = compute_spherical_harmonics(lmax,k_grids,r_grids)
    del r_grids, k_grids

    # Load fit to Patchy P(k)
    pk_input = np.loadtxt(pk_input_file)
    fid_pk_interp = interp1d(pk_input[:,0],pk_input[:,1:].T)
    pk_map = fid_pk_interp(k_norm)

# Compute k-space filters
k_filters = compute_filters(k_min,k_max,dk,k_norm)
n_k = len(k_filters)

def test_bin(a,b,c,tol=1e-6):
    """Test bin to see if it satisfies triangle conditions, being careful of numerical overlaps."""
    k_lo = np.arange(k_min,k_max,dk)
    k_hi = k_lo+dk
    ct = 0
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
            n_bins += 1

############################ COVARIANCE FUNCTIONS ##############################

### Define true random density
nbar_analyt = np.ones_like(diff)*nbar_unif

### True inverse covariance of uniform randoms
def applyCinv_unif(input_map):
    """Apply true C^{-1} to the uniform map including MAS effects."""
    return ift(ft(input_map)*MAS_mat**2)/nbar_analyt*v_cell/shot_fac_unif

############################## COMPUTE g_a #####################################

## Compute FT[n*H^-1[d](r)] (necessary part of C_a[x] derivative)
print("\n## Computing g-a maps assuming %s weightings"%weight_str)

# Compute H^-1.a
if wtype==0:
    Cinv_diff = applyCinv_fkp(diff,nbar_weight,MAS_mat,v_cell,shot_fac,include_pix=include_pix)
else:
    Cinv_diff = applyCinv(diff,nbar_weight,MAS_mat,pk_map,Yk_lm,Yr_lm,v_cell,shot_fac,rel_tol=1e-6,verb=1,max_it=50,include_pix=include_pix) # C^-1.x

Ainv_diff = applyCinv_unif(diff) # A^-1.a
del diff

# Now compute FT[nC^-1d], optionally including MAS matrix operations
if include_pix:
    ft_nCinv_a = ft(ift(ft(Cinv_diff)/MAS_mat)*nbar)
    ft_nAinv_a = ft(ift(ft(Ainv_diff)/MAS_mat)*nbar)
else:
    ft_nCinv_a = ft(Cinv_diff*nbar)
    ft_nAinv_a = ft(Ainv_diff*nbar)

del Cinv_diff, Ainv_diff

# Compute g_a and tilde-g_a maps
all_g_a = []
all_tilde_g_a = []
for i in range(n_k):
    all_g_a.append(ift(k_filters[i]*ft_nCinv_a))
    all_tilde_g_a.append(ift(k_filters[i]*ft_nAinv_a))

del ft_nCinv_a, ft_nAinv_a

############## COMPUTE < tilde-g_alpha g_beta > contribution ###################

print("\n## Computing < tilde-g-a g-b > contribution assuming %s weightings"%weight_str)

bias_ab_file_name = lambda a,b: mcdir+'patchy%d_%s_%s_%s_g%.1f_bias_map%d,%d_k%.3f_%.3f_%.3f.npz'%(N_mc,patch,z_type,weight_str,grid_factor,a,b,k_min,k_max,dk)

# Iterate over bins
for a in range(n_k):
    print("On primary k-bin %d of %d"%(a+1,n_k))
    tg_a = all_tilde_g_a[a]
    g_a = all_g_a[a]

    for b in range(a,n_k):
        tmp_av = 0.5*(tg_a*all_g_a[b]+g_a*all_tilde_g_a[b])

        # Save the product
        # Note that this is stored for a <= b only by symmetry.
        looptime = time.time()

        infile_name = bias_ab_file_name(a,b)
        lock = fasteners.InterProcessLock(infile_name+'.lock')  # for processes

        # Be careful that only one script adds to the file at once!
        with lock:
            if os.path.exists(infile_name):
                infile = np.load(infile_name)
                bias_ab = infile['dat']
                ct_ab = infile['ct']
                its = list(infile['its'])
            else:
                # first iteration!
                bias_ab = 0.
                ct_ab = 0
                its = []
            if rand_it not in its:
                np.savez(infile_name,dat=bias_ab+tmp_av/N_mc,ct=ct_ab+1,its=its+[rand_it])

##################### COMPUTE unsymmetrized phi_alpha ##########################

print("\n## Computing unsymmetrized phi-alpha maps assuming %s weightings"%weight_str)

tmp_phi_alpha_file_name = lambda a,b,c: tmpdir+'tmp_patchy_unif%d_%s_%s_%s_g%.1f_phi^alpha_map%d,%d,%d_k%.3f_%.3f_%.3f.npy'%(rand_it,patch,z_type,weight_str,grid_factor,a,b,c,k_min,k_max,dk)
tmp_tilde_phi_alpha_file_name = lambda a,b,c: tmpdir+'tmp_patchy_unif%d_%s_%s_%s_g%.1f_tilde-phi^alpha_map%d,%d,%d_k%.3f_%.3f_%.3f.npy'%(rand_it,patch,z_type,weight_str,grid_factor,a,b,c,k_min,k_max,dk)

def compute_unsymmetrized_phi(a):
    #### Compute the unsymmetrized phi maps for all betas given some alpha
    #### These are saved to some temporary directory which is later removed
    # Load first map
    g_a = all_g_a[a]
    tg_a = all_tilde_g_a[a]
    for b in range(a,n_k):
        # Compute FT[g^a[m]g^b[m]]
        ft_g_ab = ft(g_a*all_g_a[b])
        ft_tg_ab = ft(tg_a*all_tilde_g_a[b])
        for c in range(n_k):
            # Compute n(r) * IFT[Theta^c(k)FT[g^a[m]g^b[m]]], optionally with MAS corrections
            if include_pix:
                phi_alpha = np.real_if_close(ift(ft(ift(k_filters[c]*ft_g_ab)*nbar)/MAS_mat)/v_cell)
            else:
                phi_alpha = np.real_if_close(ift(k_filters[c]*ft_g_ab)*nbar/v_cell)
            np.save(tmp_phi_alpha_file_name(a,b,c),phi_alpha)
            del phi_alpha

            # Repeat for A^-1 weighted field
            if include_pix:
                tilde_phi_alpha = np.real_if_close(ift(ft(ift(k_filters[c]*ft_tg_ab)*nbar)/MAS_mat)/v_cell)
            else:
                tilde_phi_alpha = np.real_if_close(ift(k_filters[c]*ft_tg_ab)*nbar/v_cell)
            np.save(tmp_tilde_phi_alpha_file_name(a,b,c),tilde_phi_alpha)
            del tilde_phi_alpha
        del ft_g_ab, ft_tg_ab
    del g_a, tg_a

for i in range(n_k):
    print("On primary k-bin %d of %d"%(i+1,n_k))
    compute_unsymmetrized_phi(i)

del all_g_a, all_tilde_g_a

############## COMPUTE symmetrized phi_alpha and C^-1.phi_alpha ################

print("\n## Computing symmetrized phi-alpha maps and C^-1 phi_alpha assuming %s weightings"%weight_str)

sum_Cinv_phi_alpha_file_name = lambda a,b,c: mcdir+'sum_patchy_unif%d_%s_%s_%s_g%.1f_Cinv-phi^alpha_map%d,%d,%d_k%.3f_%.3f_%.3f.npz'%(N_mc,patch,z_type,weight_str,grid_factor,a,b,c,k_min,k_max,dk)
sum_tilde_phi_alpha_file_name = lambda a,b,c: mcdir+'sum_patchy_unif%d_%s_%s_%s_g%.1f_tilde-phi^alpha_map%d,%d,%d_k%.3f_%.3f_%.3f.npz'%(N_mc,patch,z_type,weight_str,grid_factor,a,b,c,k_min,k_max,dk)
Cinv_phi_alpha_file_name = lambda a,b,c: tmpdir+'patchy_%s_%s_%s_g%.1f_Cinv-phi^alpha_map%d,%d,%d_k%.3f_%.3f_%.3f.npy'%(patch,z_type,weight_str,grid_factor,a,b,c,k_min,k_max,dk)
tilde_phi_alpha_file_name = lambda a,b,c: tmpdir+'patchy_%s_%s_%s_g%.1f_tilde-phi^alpha_map%d,%d,%d_k%.3f_%.3f_%.3f.npy'%(patch,z_type,weight_str,grid_factor,a,b,c,k_min,k_max,dk)

def analyze_phi(index):

    ### Perform symmetrization for a single element, loading phi_alpha from file
    ### Also compute A^-1 phi_alpha and C^-1 phi_alpha
    ### This adds to a global average map
    a,b,c = bins_index[index]

    ### 1a. Load tilde-phi_alpha
    # Note that we only stored phi_{uvw} for u<=v by symmetry
    tilde_phi_alpha = np.load(tmp_tilde_phi_alpha_file_name(min([a,b]),max([a,b]),c))+0.j
    tilde_phi_alpha += np.load(tmp_tilde_phi_alpha_file_name(min([c,a]),max([c,a]),b))
    tilde_phi_alpha += np.load(tmp_tilde_phi_alpha_file_name(min([b,c]),max([b,c]),a))

    ### 1b. Save to temporary disk
    np.save(tilde_phi_alpha_file_name(a,b,c),tilde_phi_alpha)

    ### 1c. Add to global average
    looptime = time.time()

    infile_name = sum_tilde_phi_alpha_file_name(a,b,c)
    lock = fasteners.InterProcessLock(infile_name+'.lock')  # for processes

    # Be careful that only one script adds to the file at once!
    with lock:
        if os.path.exists(infile_name):
            infile = np.load(infile_name)
            this_sum_tilde_phi_alpha = infile['dat']
            ct_alpha1 = infile['ct']
            its = list(infile['its'])
        else:
            this_sum_tilde_phi_alpha = 0.
            ct_alpha1 = 0
            its = []
        if rand_it not in its:
            np.savez(infile_name,dat=this_sum_tilde_phi_alpha+tilde_phi_alpha/N_mc,ct=ct_alpha1+1,its=its+[rand_it])
    del tilde_phi_alpha

    ### 2a. Load phi_alpha
    phi_alpha = np.load(tmp_phi_alpha_file_name(min([a,b]),max([a,b]),c))+0.j
    phi_alpha += np.load(tmp_phi_alpha_file_name(min([c,a]),max([c,a]),b))
    phi_alpha += np.load(tmp_phi_alpha_file_name(min([b,c]),max([b,c]),a))

    ### 2b. Compute C^-1 phi_alpha
    if wtype==0:
        Cinv_phi_alpha = applyCinv_fkp(phi_alpha,nbar_weight,MAS_mat,v_cell,shot_fac,include_pix=include_pix)
    else:
        Cinv_phi_alpha = applyCinv(phi_alpha,nbar_weight,MAS_mat,pk_map,Yk_lm,Yr_lm,v_cell,shot_fac,rel_tol=1e-4,verb=0,max_it=50,include_pix=include_pix)
    del phi_alpha

    ### 2c. Save to temporary disk
    np.save(Cinv_phi_alpha_file_name(a,b,c),Cinv_phi_alpha)

    ### 2d. Add to global average
    looptime = time.time()

    infile_name = sum_Cinv_phi_alpha_file_name(a,b,c)
    lock = fasteners.InterProcessLock(infile_name+'.lock')  # for processes

    # Be careful that only one script adds to the file at once!
    with lock:
        if os.path.exists(infile_name):
            infile = np.load(infile_name)
            this_sum_Cinv_phi_alpha = infile['dat']
            ct_alpha2 = infile['ct']
            its = list(infile['its'])
        else:
            this_sum_Cinv_phi_alpha = 0.
            ct_alpha2 = 0
            its = []
        if rand_it not in its:
            np.savez(infile_name,dat=this_sum_Cinv_phi_alpha+Cinv_phi_alpha/N_mc,ct=ct_alpha2+1,its=its+[rand_it])

    # ### 2d. Add to global average
    del Cinv_phi_alpha

for i in range(n_bins):
    print("On index %d of %d"%(i+1,n_bins))
    analyze_phi(i)

###################### COMPUTE Fisher matrix contribution ######################

print("\n### Computing Fisher matrix contribution in %d bins satisfying triangle conditions"%(n_bins))

# Define Delta_alpha parameter
Delta_abc = np.zeros(n_bins)
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

def load_row(alpha):
    ### Load a single row of the Fisher matrix, (phi_alpha C^-1 phi_beta)/12
    # Note that we include combinatoric factors in phi_alpha here
    # phi_alphas are loaded from disk for this (too expensive to hold them all in memory!)

    this_row = np.zeros(n_bins)
    tilde_phi_alpha = np.load(tilde_phi_alpha_file_name(*bins_index[alpha]))

    for beta in range(alpha,n_bins): # compute diagonal by symmetry
        Cinv_phi_beta = np.load(Cinv_phi_alpha_file_name(*bins_index[beta]))
        this_row[beta] = np.real(np.sum(tilde_phi_alpha*Cinv_phi_beta)/12.)
        del Cinv_phi_beta
    del tilde_phi_alpha

    return this_row

## Now construct output from individual rows
fisher_output = np.zeros((n_bins,n_bins))
for alpha in range(n_bins):
    print("On row %d of %d"%(alpha+1,n_bins))
    fisher_output[alpha] = load_row(alpha)

# Add symmetry factor
fisher_output *= 4./np.outer(Delta_abc,Delta_abc)

## Add in conjugate symmetry and save
for alpha in range(n_bins):
    for beta in range(alpha+1,n_bins):
        fisher_output[beta,alpha] = fisher_output[alpha,beta]

np.save(fish_file_name,fisher_output)

########################### DELETE temporary files #############################

print("Removing temporary directory")
if os.path.exists(tmpdir): shutil.rmtree(tmpdir)

################################### EXIT #######################################

duration = time.time()-init
print("\n### Exiting after %d seconds (%d minutes)\n\n"%(duration,duration//60))
sys.exit()
