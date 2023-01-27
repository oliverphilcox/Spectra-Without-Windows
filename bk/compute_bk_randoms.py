# compute_bk_randoms.py (Oliver Philcox, 2021)
### Compute the component g^a maps for the binned bispectrum (even) multipoles of uniformly distributed random particles.
### These are then used to compute phi_alpha and C^-1 phi_alpha maps and added to Monte Carlo averages.
### All input parameters (specifying the mask, k-cuts etc.) are specified in a given .param file

# Import modules
from nbodykit.lab import *
import numpy as np, sys, os, time, shutil, fasteners, configparser
from scipy.interpolate import interp1d
# custom definitions
sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/../src')
from opt_utilities import load_data, load_randoms, load_nbar, load_MAS, grid_data, grid_uniforms, load_coord_grids, compute_spherical_harmonic_functions, compute_filters, ft, ift

# Read command line arguments
if len(sys.argv)!=3:
    raise Exception("Need to specify random iteration and parameter file!")
else:
    rand_it = int(sys.argv[1]) # random iteration
    paramfile = str(sys.argv[2]) # parameter file

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
    lmax_pk = 4
    from covariances_pk import applyCinv
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
tmpdir = str(config['directories']['temporary'])+str('%d/'%(rand_it))
mcdir =  str(config['directories']['monte_carlo'])

# Redshifts
ZMIN, ZMAX = float(config['sample']['z_min']), float(config['sample']['z_max'])

# Fiducial cosmological parameters
h_fid, OmegaM_fid = float(config['parameters']['h_fid']), float(config['parameters']['OmegaM_fid'])

# Survey dimensions
boxsize_grid = np.array(config.getlist('sample','box'),dtype=float)
grid_3d = np.array(config.getlist('bk-binning','grid'),dtype=int)

# Number of MC simulations
N_mc = int(config['settings']['N_mc'])
if rand_it>N_mc: raise Exception("Simulation number cannot be greater than number of bias sims!")

if wtype==1:
    # Fiducial power spectrum input (for ML weights)
    pk_input_file = str(config['settings']['fiducial_pk'])

# Testing parameters
include_pix, rand_nbar = config.getboolean('settings','include_pix'), config.getboolean('settings','rand_nbar')

# Create directories
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
print("\nMonte Carlo Directory: %s"%mcdir)
print("Temporary Directory: %s"%tmpdir)
print("\n########################################################")

init = time.time()

################################### LOAD DATA ##################################

### First check if we actually need to compute this simulation
fish_file_name = mcdir+'%s_unif%d_%s_fish_alpha_beta_k%.3f_%.3f_%.3f_l%d.npy'%(string,rand_it,weight_type,k_min,k_max,dk,lmax)
if os.path.exists(fish_file_name):
    print("Simulation already completed; exiting!\n")
    sys.exit();

print("\n## Loading %s random iteration %d with %s weights\n"%(string,rand_it,weight_type))

# Clean any crud from a previous run
if os.path.exists(tmpdir): shutil.rmtree(tmpdir)
os.makedirs(tmpdir)

### Load fiducial cosmology for co-ordinate conversions (in nbodykit)
cosmo_coord = cosmology.Cosmology(h=h_fid).match(Omega0_m = OmegaM_fid)

# Load data to get co-ordinate grids and random properties
print("Loading data")
data_true = load_data(1,config,cosmo_coord,fkp_weights=False)
rand_true = load_randoms(config,cosmo_coord,fkp_weights=False)
alpha_ran = (data_true['WEIGHT'].sum()/rand_true['WEIGHT'].sum()).compute()
shot_fac = ((data_true['WEIGHT']**2.).mean().compute()+alpha_ran*(rand_true['WEIGHT']**2.).mean().compute())/rand_true['WEIGHT'].mean().compute()
print("Data-to-random ratio: %.3f, shot-noise factor: %.3f"%(alpha_ran,shot_fac))

if rand_nbar:
    print("\nLoading nbar from random particles")
    nbar_rand, density = grid_data(data_true, rand_true, boxsize_grid, grid_3d, MAS='TSC', return_randoms=True, return_norm=False)[1:]
else:
    # load density mesh (used to define coordinate arrays)
    density = grid_data(data_true, rand_true, boxsize_grid, grid_3d, MAS='TSC', return_randoms=False, return_norm=False)[1]
del rand_true, data_true

# Load pre-computed n(r) map (from mask and n(z), not discrete particles)
print("Loading nbar from mask")
nbar_mask = load_nbar(config, 'bk', alpha_ran)

# Cell volume
v_cell = 1.*boxsize_grid.prod()/(1.*grid_3d.prod())

# Generate nbar field (same density as no. randoms), filling in zeros to allow for MAS bleed
nbar_A = load_nbar(config, 'bk', 1.)
nbar_A[nbar_A==0] = min(nbar_A[nbar_A!=0])*0.01

# Gaussian sample to create random field
np.random.seed(rand_it)
diff = np.random.normal(loc=0.,scale=np.sqrt(nbar_A),size=nbar_A.shape)

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

############################ GRID DEFINITIONS ##################################

if wtype==1:
    # Compute spherical harmonic fields in real and Fourier-space
    Y_lms = compute_spherical_harmonic_functions(np.maximum([lmax_pk,lmax]))

    # Load fit to Patchy P(k)
    pk_input = np.loadtxt(pk_input_file)
    fid_pk_interp = interp1d(pk_input[:,0],pk_input[:,1:].T)
    pk_map = fid_pk_interp(k_norm)[:lmax_pk//2+1]
else:
    Y_lms = compute_spherical_harmonic_functions(lmax)

# Compute k-space filters
k_filters = compute_filters(k_min,k_max,dk)
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
n_l = lmax//2+1
for a in range(n_k):
    for b in range(a,n_k):
        for c in range(b,n_k):
            if not test_bin(a,b,c): continue
            bins_index.append([a,b,c])
            n_bins += n_l

### True inverse covariance
def applyAinv(input_map):
    """Apply true A^{-1} to the input map (no MAS effects needed!)."""
    return input_map/nbar_A

############################## COMPUTE g_a #####################################

print("\n## Computing g-a maps assuming %s weightings"%weight_type)
# Compute C^-1 a
if wtype==0:
    Cinv_diff = applyCinv_fkp(diff,nbar_weight,MAS_mat,v_cell,shot_fac,include_pix=include_pix)
else:
    Cinv_diff = applyCinv(diff,nbar_weight,MAS_mat,pk_map,Y_lms,k_grids,r_grids,v_cell,shot_fac,rel_tol=1e-6,verb=1,max_it=30,include_pix=include_pix)

Ainv_diff = applyAinv(diff) # A^-1 a
del diff

## Compute g^a_l functions
all_g_a = []
all_tilde_g_a = []

if include_pix:
    nCinv = ift(ft(Cinv_diff)/MAS_mat)*nbar
    nAinv = ift(ft(Ainv_diff)/MAS_mat)*nbar
else:
    nCinv = nbar*Cinv_diff
    nAinv = nbar*Ainv_diff

for l_i in range(0,n_l):
    this_g_a = []
    this_tilde_g_a = []

    # Compute Sum_m Y_lm(k)FT[Y_lm(r)n(r)[C^-1d](r)]
    ft_nCinv_a = 0.
    ft_nAinv_a = 0.

    for m_i in range(len(Y_lms[l_i])):
        # Compute contribution to sum
        ft_nCinv_a += ft(nCinv*Y_lms[l_i][m_i](*r_grids))*Y_lms[l_i][m_i](*k_grids)
        ft_nAinv_a += ft(nAinv*Y_lms[l_i][m_i](*r_grids))*Y_lms[l_i][m_i](*k_grids)

    # Normalize
    ft_nCinv_a *= 4.*np.pi/(4.*l_i+1.)
    ft_nAinv_a *= 4.*np.pi/(4.*l_i+1.)

    # Apply k-binning
    for i in range(n_k):
        this_g_a.append(ift(k_filters(i,k_norm)*ft_nCinv_a))
        this_tilde_g_a.append(ift(k_filters(i,k_norm)*ft_nAinv_a))
    all_g_a.append(this_g_a)
    all_tilde_g_a.append(this_tilde_g_a)

del Cinv_diff, Ainv_diff, ft_nCinv_a, ft_nAinv_a
############## COMPUTE < tilde-g_alpha g_beta > CONTRIBUTION ###################

print("\n## Computing < tilde-g-a g-b > contribution assuming %s weightings"%weight_type)

bias_ab_file_name = lambda a,b,l_i: mcdir+'%s%d_%s_bias_map%d,%d,%d_k%.3f_%.3f_%.3f_l%d.npz'%(string,N_mc,weight_type,a,b,l_i,k_min,k_max,dk,lmax)

# Iterate over bins
for a in range(n_k):
    print("On primary k-bin %d of %d"%(a+1,n_k))

    for b in range(a,n_k):
        # b > a, so only this term contains ell
        for l_i in range(n_l):

            # Save the product (note that this is stored for a <= b only by symmetry).
            infile_name = bias_ab_file_name(a,b,l_i)
            lock = fasteners.InterProcessLock(infile_name+'.lock')  # for processes

            # Be careful that only one script adds to the file at once!
            with lock:
                if os.path.exists(infile_name):
                    infile = np.load(infile_name)
                    its = list(infile['its'])
                    bias_ab = infile['dat']
                    ct_ab = infile['ct']
                else:
                    # first iteration!
                    bias_ab = 0.
                    ct_ab = 0
                    its = []
                if rand_it not in its:
                    # Add to file
                    tmp_av = 0.5*(all_tilde_g_a[0][a]*all_g_a[l_i][b]+all_g_a[0][a]*all_tilde_g_a[l_i][b])
                    np.savez(infile_name,dat=bias_ab+tmp_av/N_mc,ct=ct_ab+1,its=its+[rand_it])

############## COMPUTE phi_alpha AND C^-1 phi_alpha ################

def compute_unsymmetrized_phi1_single(a,b,c,l_i):
    #### Compute the unsymmetrized phi1 maps for a single a,b,c,l
    #### These have Legendre polynomial only in a or b
    # b >= a, so only the second term contains ell
    # Compute IFT[FT[g^a_0[m]g^b_l[m]]*Theta_c(k)
    k_filt = k_filters(c,k_norm)
    g_ab_c = ift(ft(all_g_a[0][a]*all_g_a[l_i][b])*k_filt)
    tg_ab_c = ift(ft(all_tilde_g_a[0][a]*all_tilde_g_a[l_i][b])*k_filt)

    # Compute n(r) * IFT[Theta^c(k)FT[g^a[m]g^b[m]]], optionally with MAS corrections
    # Also repeat for A^-1 weighted field
    if include_pix:
        phi_alpha = np.real(ift(ft(g_ab_c*nbar)/MAS_mat)/v_cell)
        tilde_phi_alpha = np.real(ift(ft(tg_ab_c*nbar)/MAS_mat)/v_cell)
    else:
        phi_alpha = np.real(g_ab_c*nbar/v_cell)
        tilde_phi_alpha = np.real(tg_ab_c*nbar/v_cell)

    return phi_alpha, tilde_phi_alpha

def compute_unsymmetrized_phi2_single(a,b,c,l_i):
    #### Compute the unsymmetrized phi2 maps for a single a,b,c,l
    #### These have Legendre polynomial only in c
    # Load first map
    # a, b have ell = 0
    # Compute FT[g^a_0[m]g^b_0[m]]
    kfilt = k_filters(c,k_norm)
    ft_g_ab = ft(all_g_a[0][a]*all_g_a[0][b])*kfilt
    ft_tg_ab = ft(all_tilde_g_a[0][a]*all_tilde_g_a[0][b])*kfilt
    del kfilt

    # Compute n(r)*Y_lm(r)*IFT[Y_lm(k)*Theta^c(k)FT[g^a[m]g^b[m]]], optionally with MAS corrections
    phi_alpha = 0.
    tilde_phi_alpha = 0.
    for m_i in range(len(Y_lms[l_i])):
        Ylm = Y_lms[l_i][m_i]
        if include_pix:
            phi_alpha += np.real(ift(ft(ift(Ylm(*k_grids)*ft_g_ab)*nbar)/MAS_mat)/v_cell*Ylm(*r_grids))
            tilde_phi_alpha += np.real(ift(ft(ift(Ylm(*k_grids)*ft_tg_ab)*nbar)/MAS_mat)/v_cell*Ylm(*r_grids))
        else:
            ylmn = Ylm(*r_grids)*nbar/v_cell
            phi_alpha += np.real(ift(Ylm(*k_grids)*ft_g_ab)*ylmn)
            tilde_phi_alpha += np.real(ift(Ylm(*k_grids)*ft_tg_ab)*ylmn)

    # Add normalization factor
    phi_alpha *= 4.*np.pi/(4.*l_i+1.)
    tilde_phi_alpha *= 4.*np.pi/(4.*l_i+1.)

    return phi_alpha, tilde_phi_alpha

print("\n## Computing phi-alpha maps and C^-1 phi_alpha assuming %s weightings"%weight_type)

sum_Cinv_phi_alpha_file_name = lambda a,b,c,l_i: mcdir+'sum_%s_unif%d_%s_Cinv-phi^alpha_map%d,%d,%d,%d_k%.3f_%.3f_%.3f_l%d.npz'%(string,N_mc,weight_type,a,b,c,l_i,k_min,k_max,dk,lmax)
sum_tilde_phi_alpha_file_name = lambda a,b,c,l_i: mcdir+'sum_%s_unif%d_%s_tilde-phi^alpha_map%d,%d,%d,%d_k%.3f_%.3f_%.3f_l%d.npz'%(string,N_mc,weight_type,a,b,c,l_i,k_min,k_max,dk,lmax)
Cinv_phi_alpha_file_name = lambda a,b,c,l_i: tmpdir+'%s_%s_Cinv-phi^alpha_map%d,%d,%d,%d_k%.3f_%.3f_%.3f_l%d.npy'%(string,weight_type,a,b,c,l_i,k_min,k_max,dk,lmax)
tilde_phi_alpha_file_name = lambda a,b,c,l_i: tmpdir+'%s_%s_tilde-phi^alpha_map%d,%d,%d,%d_k%.3f_%.3f_%.3f_l%d.npy'%(string,weight_type,a,b,c,l_i,k_min,k_max,dk,lmax)

def analyze_phi(index):

    l_i = index//(n_bins//n_l)
    a_index = (index%(n_bins//n_l))
    a,b,c = bins_index[a_index]

    ### Check if the outputs already exists
    check = 0
    if os.path.exists(tilde_phi_alpha_file_name(a,b,c,l_i)):
        check += 1
    infile_name = sum_tilde_phi_alpha_file_name(a,b,c,l_i)
    lock = fasteners.InterProcessLock(infile_name+'.lock')  # for processes
    with lock:
        if os.path.exists(infile_name):
            if rand_it in np.load(infile_name)['its']:
                check += 1
    if os.path.exists(Cinv_phi_alpha_file_name(a,b,c,l_i)):
        check += 1
    infile_name = sum_Cinv_phi_alpha_file_name(a,b,c,l_i)
    lock = fasteners.InterProcessLock(infile_name+'.lock')  # for processes
    with lock:
        if os.path.exists(infile_name):
            if rand_it in np.load(infile_name)['its']:
                check += 1
    if check==4:
        print("All files exist; continuing")
        return

    ### Now start properly

    ### Perform symmetrization for a single element, loading phi_alpha from file
    ### Also compute A^-1 phi_alpha and C^-1 phi_alpha
    ### This adds to a global average map

    ### 1a. Load tilde-phi_alpha
    # Note that we only stored phi_{uvw} for u<=v by symmetry
    # Assemble matrices, using symmetries if possible

    if l_i==0:
        # use simpler form in this case!
        phi_abc, tilde_phi_abc = compute_unsymmetrized_phi1_single(a,b,c,l_i)
    else:
        phi_abc, tilde_phi_abc = compute_unsymmetrized_phi2_single(a,b,c,l_i)
    if b==c and l_i==0:
        phi_acb, tilde_phi_acb = phi_abc, tilde_phi_abc
    else:
        phi_acb, tilde_phi_acb = compute_unsymmetrized_phi1_single(a,c,b,l_i)
    if a==b:
        phi_bca, tilde_phi_bca = phi_acb, tilde_phi_acb
    else:
        if c==a and l_i==0:
            phi_bca, tilde_phi_bca = phi_abc, tilde_phi_abc
        else:
            phi_bca, tilde_phi_bca = compute_unsymmetrized_phi1_single(b,c,a,l_i)

    tilde_phi_alpha = tilde_phi_abc + tilde_phi_acb + tilde_phi_bca

    ### 1b. Save to temporary disk
    if not os.path.exists(tilde_phi_alpha_file_name(a,b,c,l_i)):
        np.save(tilde_phi_alpha_file_name(a,b,c,l_i),tilde_phi_alpha)
    del tilde_phi_abc, tilde_phi_acb, tilde_phi_bca

    ### 1c. Add to global average
    infile_name = sum_tilde_phi_alpha_file_name(a,b,c,l_i)
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
    phi_alpha = phi_abc + phi_acb + phi_bca
    del phi_abc, phi_acb, phi_bca

    ### 2b. Compute C^-1 phi_alpha
    if wtype==0:
        Cinv_phi_alpha = np.real(applyCinv_fkp(phi_alpha,nbar_weight,MAS_mat,v_cell,shot_fac,include_pix=include_pix))
    else:
        Cinv_phi_alpha = np.real(applyCinv(phi_alpha,nbar_weight,MAS_mat,pk_map,Y_lms,k_grids,r_grids,v_cell,shot_fac,rel_tol=1e-6,verb=0,max_it=30,include_pix=include_pix))
    del phi_alpha

    ### 2c. Save to temporary disk
    if not os.path.exists(Cinv_phi_alpha_file_name(a,b,c,l_i)):
        np.save(Cinv_phi_alpha_file_name(a,b,c,l_i),Cinv_phi_alpha)

    ### 2d. Add to global average
    infile_name = sum_Cinv_phi_alpha_file_name(a,b,c,l_i)
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
    del Cinv_phi_alpha

for i in range(n_bins):
    print("On index %d of %d"%(i+1,n_bins))
    analyze_phi(i)

if wtype==1:
    del k_grids, r_grids, Y_lms, pk_map

###################### COMPUTE FISHER MATRIX CONTRIBUTION ######################

# Compute Delta_abc
print("Computing Delta_{abc} normalization")
Delta_abc = np.zeros(n_bins)

# First compute ell=0 elements
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

# Now compute ell > 0 elements, if required
if n_l>1:

    # Define discrete binning functions
    bins = [ift(k_filters(a,k_norm)) for a in range(n_k)]

    i_min,i = n_bins//n_l,0
    for l_i in range(1,n_l):
        for a in range(n_k):
            for b in range(a,n_k):
                for c in range(b,n_k):
                    if not test_bin(a,b,c): continue
                    if a!=b and b!=c:
                        Delta_abc[i+i_min] = 1.
                    elif a==b and b!=c:
                        Delta_abc[i+i_min] = 2.
                    elif a!=b and b==c:
                        Nabc_0 = np.sum(bins[a]*bins[c]**2.).real
                        Dl_c = [ift(k_filters(c,k_norm)*Y_lms[l_i][m_i](*k_grids)) for m_i in range(len(Y_lms[l_i]))]
                        Nabc_l = np.sum([np.sum(bins[a]*Dl_c[m_i]*Dl_c[m_i]).real for m_i in range(len(Y_lms[l_i]))])*4.*np.pi/(4.*l_i+1.)                        
                        Delta_abc[i+i_min] = 1.+Nabc_l/Nabc_0
                    elif a==b and b==c:
                        Nabc_0 = np.sum(bins[c]**3.).real
                        Dl_c = [ift(k_filters(c,k_norm)*Y_lms[l_i][m_i](*k_grids)) for m_i in range(len(Y_lms[l_i]))]
                        Nabc_l = np.sum([np.sum(bins[a]*Dl_c[m_i]*Dl_c[m_i]).real for m_i in range(len(Y_lms[l_i]))])*4.*np.pi/(4.*l_i+1.)                        
                        Delta_abc[i+i_min] = 2.*(1.+2.*Nabc_l/Nabc_0)    
                    i += 1

    del bins, Dl_c
 
print("\n### Computing Fisher matrix contribution in %d bins satisfying triangle conditions"%(n_bins))

def load_row(alpha):
    ### Load a single row of the Fisher matrix, (phi_alpha C^-1 phi_beta)/12
    # Note that we include combinatoric factors in phi_alpha here
    # phi_alphas are loaded from disk for this (too expensive to hold them all in memory!)

    l_i = alpha//(n_bins//n_l)
    a_index = (alpha%(n_bins//n_l))

    this_row = np.zeros(n_bins)
    tilde_phi_alpha = np.load(tilde_phi_alpha_file_name(bins_index[a_index][0],bins_index[a_index][1],bins_index[a_index][2],l_i))

    for beta in range(alpha,n_bins): # compute diagonal by symmetry

        l_j = beta//(n_bins//n_l)
        b_index = (beta%(n_bins//n_l))

        Cinv_phi_beta = np.load(Cinv_phi_alpha_file_name(bins_index[b_index][0],bins_index[b_index][1],bins_index[b_index][2],l_j))
        this_row[beta] = np.sum(tilde_phi_alpha*Cinv_phi_beta)/12.
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
print("\n### Exiting after %d seconds (%d minutes)\n"%(duration,duration//60))
sys.exit()
