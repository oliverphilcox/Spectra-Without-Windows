# generate_mask.py (Oliver Philcox, 2021)
#
### Compute a smooth background field n(\vec{r}) on the grid using the survey mask and n(z) functions for either the BOSS or Patchy geometry.
#
# This is a necessary input to the power spectrum and bispectrum estimators
# We use the Patchy geometry by default in the analysis (should be similar to BOSS)
#
# Inputs:
# - sim_type: 0 for BOSS geometry, 1 for Patchy geometry
# - patch: Sky region - ngc or sgc.
# - z_type: Redshift cut - z1 or z3.
# - grid_factor: Factor by which to reduce the pixel size. The Nyquist frequency is proportional to the inverse of this.
#
# Within the code we can also change the output directory, fiducial cosmology and data cuts. Note that the MANGLE mask for BOSS must be downloaded and specified.

################################# LOAD MODULES #################################
# If sim no = -1 the true BOSS data is used
import sys, os, copy, time, pyfftw, mangle
import numpy as np
from nbodykit.lab import *
from scipy.interpolate import interp1d, UnivariateSpline
from nbodykit import setup_logging, style
# custom definitions
sys.path.append('src/')
from opt_utilities import load_data, load_randoms, grid_data, load_coord_grids

# Read command line arguments
if len(sys.argv)!=5:
    raise Exception("Need to specify simulation type, patch, z-type and grid factor!")
else:
    sim_type = int(sys.argv[1]) # 0 = BOSS, 1 = Patchy
    patch = str(sys.argv[2]) # ngc or sgc
    z_type = str(sys.argv[3]) # z1 or z3
    grid_factor = float(sys.argv[4])

############################# INPUT PARAMETERS #################################

h_fid = 0.676
OmegaM_fid = 0.31

# Output directory (should be set in src/opt_utilities also)
outdir = '/projects/QUIJOTE/Oliver/boss_masks/'

# Input mask file
if patch=='ngc':
    maskfile = '/projects/QUIJOTE/Oliver/boss_masks/mask_DR12v5_CMASSLOWZTOT_North.ply'
elif patch=='sgc':
    maskfile = '/projects/QUIJOTE/Oliver/boss_masks/mask_DR12v5_CMASSLOWZTOT_South.ply'
else:
    raise Exception("Wrong input patch!")

### In principle, nothing below here needs to be changed

# Cosmology for co-ordinate conversions
cosmo_coord = cosmology.Cosmology(h=h_fid).match(Omega0_m=OmegaM_fid)

# Load redshift ranges
if z_type=='z1':
    ZMIN = 0.2
    ZMAX = 0.5
elif z_type=='z3':
    ZMIN = 0.5
    ZMAX = 0.75
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

init = time.time()

###################################### LOAD DATA ###############################

if sim_type==0:
    print("\n## Creating BOSS mask from %s %s reducing grid by factor of %.1f\n"%(patch,z_type,grid_factor))
    sim_no=-1
elif sim_type==1:
    print("\n## Creating Patchy mask from %s %s reducing grid by factor of %.1f\n"%(patch,z_type,grid_factor))
    sim_no=42 # not directly used
else:
    raise Exception("Wrong sim type!")

# Load data to get co-ordinate grids and random properties
data = load_data(sim_no,ZMIN,ZMAX,cosmo_coord,patch=patch,fkp_weights=False)
randoms = load_randoms(sim_no,ZMIN,ZMAX,cosmo_coord,patch=patch,fkp_weights=False)
diff,density = grid_data(data, randoms, boxsize_grid, grid_3d,return_randoms=False)
k_grids, r_grids = load_coord_grids(boxsize_grid,grid_3d,density)

# Compute \vec{r} grid
X_grid,Y_grid,Z_grid = r_grids
pos_grid = np.vstack([X_grid.ravel(),Y_grid.ravel(),Z_grid.ravel()]).T

# Convert to sky co-ordinates
print("## Creating RA, Dec, z co-ordinate grid")
ra_grid, dec_grid, z_grid = transform.CartesianToSky(pos_grid,cosmo_coord)
ra_all = ra_grid.reshape(X_grid.shape)
dec_all = dec_grid.reshape(X_grid.shape)
z_all = z_grid.reshape(X_grid.shape)

################################# COMPUTE N(Z) #################################

print("## Creating n(z) interpolator from random particles")
# Histogram the random n_z values including systematic weights
# This gives dN = n(z)*dz
random_z = randoms['Z'].compute()
random_w = randoms['WEIGHT'].compute()
nz,z = np.histogram(random_z,bins=100,weights=random_w,range=[ZMIN,ZMAX])
z_av = 0.5*(z[:-1]+z[1:])
rz = cosmo_coord.comoving_distance(z)

# Convert to volume density i.e. dN = n_V(z)dV
volz = 4.*np.pi/3.*(rz[1:]**3.-rz[:-1]**3.)
nbar_z_interp = UnivariateSpline(z_av,nz/volz,s=0.0000003,ext='zeros')

################################ DEFINE MASK ###################################

print("## Defining angular mask")

mask = mangle.Mangle(maskfile)

# Define n_V(z) on grid
nbar_z_grid = nbar_z_interp(z_all)
# Define angular mask on grid, removing any points outside mask area
ids = mask.get_polyids(ra_grid,dec_grid)
angular_weight_grid = mask.weights[ids]
angular_weight_grid[ids==-1] = 0.
angular_weights = angular_weight_grid.reshape(X_grid.shape)

# Define total mask
print("## Computing total mask")
nbar_mask = nbar_z_grid*angular_weights

# Normalize mask to random particle density
v_cell = 1.*grid_3d.prod()/(1.*boxsize_grid.prod())
nbar_mask *= np.sum(randoms['WEIGHT']).compute()*v_cell/np.sum(nbar_mask)

################################# SAVE AND EXIT ################################

if sim_type==0:
    file_name = outdir+'nbar_boss_%s_%s_z%.3f_%.3f_g%.1f'%(patch,z_type,ZMIN,ZMAX,grid_factor)
elif sim_type==1:
    file_name = outdir+'nbar_patchy_%s_%s_z%.3f_%.3f_g%.1f'%(patch,z_type,ZMIN,ZMAX,grid_factor)
np.save(file_name,nbar_mask)

duration = time.time()-init
print("Output map saved to %s. Exiting after %d seconds (%d minutes)\n\n"%(file_name,duration,duration//60))
