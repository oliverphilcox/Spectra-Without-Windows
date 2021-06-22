# compute_bk_data.py (Oliver Philcox, 2021)
### Compute the component g^a maps for the binned bispectrum of BOSS or Patchy data with FKP or ML weightings
### These are then used to compute the full windowless bispectrum estimate
### Note compute_bk_randoms.py must be run on N_mc sims before this script to compute Fisher matrix contributions
### If the sim-no parameter is set to -1, this will compute the bispectrum of BOSS data

# Import modules
from nbodykit.lab import *
import sys, os, copy, time, pyfftw
import numpy as np
from scipy.interpolate import interp1d
# custom definitions
sys.path.append('../src')
from opt_utilities import load_data, load_randoms, load_MAS, load_nbar, grid_data, load_coord_grids, compute_spherical_harmonics, compute_filters, ft, ift, plotter

# Read command line arguments
if len(sys.argv)!=6:
<<<<<<< HEAD
    raise Exception("Need to specify simulation number, patch, z-type, weight-type and grid factor!")
=======
    raise Exception("Need to specify random iteration, weight-type and grid factor!")
>>>>>>> c3992fbb0e5f95d0f96bd056cf1bfa0995eb7218
else:
    # If sim no = -1 the true BOSS data is used
    sim_no = int(sys.argv[1])
    patch = str(sys.argv[2]) # ngc or sgc
    z_type = str(sys.argv[3]) # z1 or z3
    wtype = int(sys.argv[4]) # 0 for FKP, 1 for ML
    grid_factor = float(sys.argv[5])

########################### INPUT PARAMETERS ###########################

## k-space binning
<<<<<<< HEAD
k_min = 0.00
=======
k_min = 0.0
>>>>>>> c3992fbb0e5f95d0f96bd056cf1bfa0995eb7218
k_max = 0.16
dk = 0.01

## Cosmological parameters for co-ordinate conversions
h_fid = 0.676
OmegaM_fid = 0.31

## Number of Monte Carlo simulations used
N_mc = 50

# Whether to forward-model pixellation effects.
include_pix = False
# If true, use nbar(r) from the random particles instead of the mask / n(z) distribution.
<<<<<<< HEAD
rand_nbar = True

use_qbar = True
if not use_qbar:
    print("Not subtracting q-bar pieces!")

## Directories
mcdir = '/projects/QUIJOTE/Oliver/bk_opt_production5a/summed_phi_alpha/' # to hold intermediate sums (should be large)
outdir = '/projects/QUIJOTE/Oliver/bk_opt_production5a/bk_estimates/' # to hold output bispectra
=======
rand_nbar = False

## Directories
mcdir = '/projects/QUIJOTE/Oliver/bk_opt2/summed_phi_alpha/' # to hold intermediate sums (should be large)
outdir = '/projects/QUIJOTE/Oliver/bk_opt2/bk_estimates/' # to hold output bispectra
>>>>>>> c3992fbb0e5f95d0f96bd056cf1bfa0995eb7218

if wtype==1:
    # Fiducial power spectrum input (for ML weights)
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
print("Monte Carlo Directory: %s"%mcdir)
print("Output Directory: %s"%outdir)
print("\n########################################################")

init = time.time()

########################### LOAD DATA ###########################

# Check if simulation has already been analyzed
<<<<<<< HEAD
if sim_no!=-1:
    p_alpha_file_name = outdir + 'bk_patchy%d_%s_%s_%s_N%d_g%.1f_k%.3f_%.3f_%.3f.txt'%(sim_no,patch,z_type,weight_str,N_mc,grid_factor,k_min,k_max,dk)
else:
    p_alpha_file_name = outdir + 'bk_boss_%s_%s_%s_N%d_g%.1f_k%.3f_%.3f_%.3f.txt'%(patch,z_type,weight_str,N_mc,grid_factor,k_min,k_max,dk)

if os.path.exists(p_alpha_file_name):
    print("Simulation has already been computed; exiting!")
    sys.exit()

if sim_no!=-1:
    print("\n## Loading %s %s simulation %d with %s weights and grid-factor %.1f"%(patch,z_type,sim_no,weight_str,grid_factor))
else:
=======
if sim_no!=-1:
    p_alpha_file_name = outdir + 'bk_patchy%d_%s_%s_%s_N%d_g%.1f_k%.3f_%.3f_%.3f.txt'%(sim_no,patch,z_type,weight_str,N_mc,grid_factor,k_min,k_max,dk)
else:
    p_alpha_file_name = outdir + 'bk_boss_%s_%s_%s_N%d_g%.1f_k%.3f_%.3f_%.3f.txt'%(patch,z_type,weight_str,N_mc,grid_factor,k_min,k_max,dk)

if os.path.exists(p_alpha_file_name):
    print("Simulation has already been computed; exiting!")
    sys.exit()

if sim_no!=-1:
    print("\n## Loading %s %s simulation %d with %s weights and grid-factor %.1f"%(patch,z_type,sim_no,weight_str,grid_factor))
else:
>>>>>>> c3992fbb0e5f95d0f96bd056cf1bfa0995eb7218
    print("\n## Loading %s %s BOSS data with %s weights and grid-factor %.1f"%(patch,z_type,weight_str,grid_factor))

### Load fiducial cosmology for co-ordinate conversions (in nbodykit)
cosmo_coord = cosmology.Cosmology(h=h_fid).match(Omega0_m = OmegaM_fid)

# Load data and paint to grid
data = load_data(sim_no,ZMIN,ZMAX,cosmo_coord,patch=patch,fkp_weights=False);
randoms = load_randoms(sim_no,ZMIN,ZMAX,cosmo_coord,patch=patch,fkp_weights=False);
if rand_nbar:
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
<<<<<<< HEAD
nbar_mask = load_nbar(sim_no, patch, z_type, ZMIN, ZMAX, grid_factor, alpha_ran, z_only=True)
=======
nbar_mask = load_nbar(sim_no, patch, z_type, ZMIN, ZMAX, grid_factor, alpha_ran)
>>>>>>> c3992fbb0e5f95d0f96bd056cf1bfa0995eb7218

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

########################### GRID DEFINITIONS ###########################

# Cell volume
v_cell = 1.*boxsize_grid.prod()/(1.*grid_3d.prod())

# Load MAS grids
MAS_mat = load_MAS(boxsize_grid, grid_3d)

if wtype==1:
    # Compute spherical harmonic fields in real and Fourier-space
    Yk_lm, Yr_lm = compute_spherical_harmonics(lmax,k_grids,r_grids)

    # Load fit to Patchy P(k)
    pk_input = np.loadtxt(pk_input_file)
    fid_pk_interp = interp1d(pk_input[:,0],pk_input[:,1:].T)
    pk_map = fid_pk_interp(k_norm)

del r_grids, k_grids

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

########################### COMPUTE g_a ###########################

## Compute FT[n*H^-1[d](r)] (necessary part of C_a[x] derivative)
print("\n## Computing g-a maps assuming %s weightings"%weight_str)

# Compute H^-1.d
if wtype==0:
<<<<<<< HEAD
    Cinv_diff = applyCinv_fkp(diff,nbar_weight,MAS_mat,v_cell,shot_fac,include_pix=include_pix)
else:
    Cinv_diff = applyCinv(diff,nbar_weight,MAS_mat,pk_map,Yk_lm,Yr_lm,v_cell,shot_fac,rel_tol=1e-6,verb=1,max_it=50,include_pix=include_pix)
=======
    Cinv_diff = applyCinv_fkp(diff,nbar_weight,MAS_mat,v_cell,shot_fac,use_MAS=include_pix)
else:
    Cinv_diff = applyCinv(diff,nbar_weight,MAS_mat,pk_map,Yk_lm,Yr_lm,v_cell,shot_fac,rel_tol=1e-6,verb=1,max_it=50,use_MAS=include_pix)
>>>>>>> c3992fbb0e5f95d0f96bd056cf1bfa0995eb7218
    del pk_map, Yk_lm, Yr_lm
del diff, nbar_weight

# Now compute FT[nH^-1d], optionally including MAS matrix operations
if include_pix:
    ft_nCinv_d = ft(ift(ft(Cinv_diff)/MAS_mat)*nbar)
else:
    # nb: still need MAS here else Bk scales as MAS^{-3}!
    ft_nCinv_d = ft(Cinv_diff*nbar)*MAS_mat
del Cinv_diff, nbar, MAS_mat

## Compute g^a functions
all_g_a = []
for i in range(n_k):
    all_g_a.append(ift(k_filters[i]*ft_nCinv_d))
del ft_nCinv_d, k_filters

########################### COMPUTE q_alpha ###########################

print("\n## Computing q_alpha quantity in %d bins assuming %s weightings"%(n_bins,weight_str))

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

if sim_no==-1:
    root = 'boss'
else:
    root = 'patchy'

def bias_term(a,b):
    a0 = min([a,b])
    a1 = max([a,b])

    filename = mcdir+'%s%d_%s_%s_%s_g%.1f_bias_map%d,%d_k%.3f_%.3f_%.3f.npz'%(root,N_mc,patch,z_type,weight_str,grid_factor,a0,a1,k_min,k_max,dk)
    infile = np.load(filename)
    if infile['ct']!=N_mc: raise Exception("Wrong number of bias simulations computed! (%d of %d)"%(infile['ct'],N_mc))

    return infile['dat']

# Iterate over all possible triangles, creating q_alpha
q_alpha = []
for a in range(n_k):
    print("On primary k-bin %d of %d"%(a+1,n_k))
    g_a = all_g_a[a]
    for b in range(a,n_k):
        g_b = all_g_a[b]
        for c in range(b,n_k):
            if not test_bin(a,b,c): continue
            g_c = all_g_a[c]

            ## Analyze this bin
<<<<<<< HEAD
            if use_qbar:
                tmp_q = np.sum(g_a*g_b*g_c-g_a*bias_term(b,c)-g_b*bias_term(c,a)-g_c*bias_term(a,b))
            else:
                tmp_q = np.sum(g_a*g_b*g_c)
=======
            tmp_q = np.sum(g_a*g_b*g_c-g_a*bias_term(b,c)-g_b*bias_term(c,a)-g_c*bias_term(a,b))
>>>>>>> c3992fbb0e5f95d0f96bd056cf1bfa0995eb7218
            q_alpha.append(np.real_if_close(tmp_q))

# Add symmetry factor
q_alpha = np.asarray(q_alpha)/Delta_abc

########################### COMPUTE Fisher matrix ###########################

full_fish_file_name = outdir+'%s_mean%d_%s_%s_%s_g%.1f_full-fish_alpha_beta_k%.3f_%.3f_%.3f.npy'%(root,N_mc,patch,z_type,weight_str,grid_factor,k_min,k_max,dk)

if not os.path.exists(full_fish_file_name):

    # Compute mean Fisher matrix < tilde-phi > C^-1 < phi >

    print("\n## Computing mean Fisher matrix contribution")

    sum_Cinv_phi_alpha_file_name = lambda a,b,c: mcdir+'sum_%s_unif%d_%s_%s_%s_g%.1f_Cinv-phi^alpha_map%d,%d,%d_k%.3f_%.3f_%.3f.npz'%(root,N_mc,patch,z_type,weight_str,grid_factor,a,b,c,k_min,k_max,dk)
    sum_tilde_phi_alpha_file_name = lambda a,b,c: mcdir+'sum_%s_unif%d_%s_%s_%s_g%.1f_tilde-phi^alpha_map%d,%d,%d_k%.3f_%.3f_%.3f.npz'%(root,N_mc,patch,z_type,weight_str,grid_factor,a,b,c,k_min,k_max,dk)

    def load_row_mean(alpha):
        ### Load a single row of the mean Fisher matrix, < phi_alpha > C^-1 < phi_beta >/12
        # Note that we include combinatoric factors in phi_alpha here

        this_row = np.zeros(n_bins)
        infile = np.load(sum_tilde_phi_alpha_file_name(*bins_index[alpha]))
        if infile['ct']!=N_mc:
            print(np.sort(infile['its']))
            print(alpha)
            raise Exception("Wrong number of tilde-phi bias simulations computed! (%d of %d)"%(infile['ct'],N_mc))
        mean_tilde_phi_alpha = infile['dat']
        infile.close()

        for beta in range(alpha,n_bins): # compute only upper triangle and diagonal by symmetry
            infile = np.load(sum_Cinv_phi_alpha_file_name(*bins_index[beta]))
            if infile['ct']!=N_mc:
                print(np.sort(infile['its']))
                print(beta)
                raise Exception("Wrong number of Cinv-phi bias simulations computed! (%d of %d)"%(infile['ct'],N_mc))
            mean_Cinv_phi_beta = infile['dat']
            infile.close()
<<<<<<< HEAD
            this_row[beta] = np.real(np.sum(mean_tilde_phi_alpha*mean_Cinv_phi_beta)/12.)
=======
            this_row[beta] = np.real_if_close(np.sum(mean_tilde_phi_alpha*mean_Cinv_phi_beta)/12.)
>>>>>>> c3992fbb0e5f95d0f96bd056cf1bfa0995eb7218
            del mean_Cinv_phi_beta
        del mean_tilde_phi_alpha

        return this_row

    ## Construct fisher matrix mean contribution from individual rows
    mean_fisher = np.zeros((n_bins,n_bins))
    # Iterate over rows
    for i in range(n_bins):
        if i%5==0: print("On index %d of %d"%(i+1,n_bins))
        mean_fisher[i] = load_row_mean(i)

    # Add symmetry factor
    mean_fisher *= 4./np.outer(Delta_abc,Delta_abc)

    ## Add in conjugate symmetry
    for alpha in range(n_bins):
        for beta in range(alpha+1,n_bins):
            mean_fisher[beta,alpha] = mean_fisher[alpha,beta]

    # Compute full Fisher matrix contribution, < tilde-phi C^-1 phi >
    print("\n### Computing full Fisher matrix contribution")

    ### Define file names
    def fish_file_name(bias_sim):
<<<<<<< HEAD
        return mcdir+'%s_unif%d_%s_%s_%s_g%.1f_fish_alpha_beta_k%.3f_%.3f_%.3f.npy'%(root,bias_sim,patch,z_type,weight_str,grid_factor,k_min,k_max,dk)

    # Iterate over simulations and normalize correctly
    full_fisher = np.zeros((n_bins,n_bins))
    for i in range(1,N_mc+1):
=======
        return mcdir+'%s_unif%d_%s_%s_%s_g%.1f_fish_alpha_beta_k%.3f_%.3f_%.3f.npy'%(root,N_mc,patch,z_type,weight_str,grid_factor,k_min,k_max,dk)

    # Iterate over simulations and normalize correctly
    full_fisher = np.zeros((n_bins,n_bins))
    for i in range(N_mc):
>>>>>>> c3992fbb0e5f95d0f96bd056cf1bfa0995eb7218
        full_fisher += np.load(fish_file_name(i))/N_mc

    full_fisher -= mean_fisher
    np.save(full_fish_file_name,full_fisher)

else:
    print("\n## Loading Fisher matrix from file")
    full_fisher = np.load(full_fish_file_name)

########################### COMPUTE p_alpha ###########################

print("\n## Computing p_alpha in %d bins assuming %s weightings"%(n_bins,weight_str))

# Create output
p_alpha = np.matmul(np.linalg.inv(full_fisher),q_alpha)

########################### SAVE & EXIT ###########################

with open(p_alpha_file_name,"w+") as output:
    if sim_no==-1:
        output.write("####### Bispectrum of BOSS #############")
    else:
        output.write("####### Bispectrum of Patchy Simulation %d #############"%sim_no)
    output.write("\n# Patch: %s"%patch)
    output.write("\n# z-type: %s"%z_type)
    output.write("\n# Weights: %s"%weight_str)
    output.write("\n# Fiducial Omega_m: %.3f"%OmegaM_fid)
    output.write("\n# Fiducial h: %.3f"%h_fid)
    output.write("\n# Boxsize: [%.1f, %.1f, %.1f]"%(boxsize_grid[0],boxsize_grid[1],boxsize_grid[2]))
    output.write("\n# Grid: [%d, %d, %d]"%(grid_3d[0],grid_3d[1],grid_3d[2]))
    output.write("\n# k-binning: [%.3f, %.3f, %.3f]"%(k_min,k_max,dk))
    output.write("\n# Monte Carlo Simulations: %d"%N_mc)
    output.write("\n#")
    output.write("\n# Format: k1 | k2 | k3 | B(k1,k2,k3)")
    output.write("\n############################################")

    k_av = np.arange(k_min,k_max,dk)+dk/2.

    for i in range(n_bins):
        a,b,c = bins_index[i]
        output.write('\n%.4f\t%.4f\t%.4f\t%.8e'%(k_av[a],k_av[b],k_av[c],p_alpha[i]))

duration = time.time()-init
print("## Saved bispectrum estimates to %s. Exiting after %d seconds (%d minutes)\n"%(p_alpha_file_name,duration,duration//60))
sys.exit()
