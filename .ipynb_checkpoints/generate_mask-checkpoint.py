# generate_mask.py (Oliver Philcox, 2021)
#
### Compute a smooth background field n(\vec{r}) on the grid using the survey mask and n(z) functions for a given survey geometry.
#
# This is a necessary input to the power spectrum and bispectrum estimators
# Note that this requires the manglepy (Molly Swanson) code to load the window function.
# Note that the MANGLE mask for the survey must be downloaded and specified in the parameter file.
# The code will compute the mask for either the power spectrum or the bispectrum: these are different only if the grid-size is different for the two.

################################# LOAD MODULES #################################
# Import modules
import os, sys, time, mangle, configparser, numpy as np
from nbodykit.lab import *
from scipy.interpolate import UnivariateSpline
# custom definitions
sys.path.append('src/')
from opt_utilities import load_data, load_randoms, grid_data, load_coord_grids

# Read command line arguments
if len(sys.argv)!=3:
    raise Exception("Need to specify spectrum type and parameter file!")
else:
    spec_type = str(sys.argv[1]) # power spectrum (`pk`) or bispectrum (`bk`)
    paramfile = str(sys.argv[2]) # parameter file
    
############################# INPUT PARAMETERS #################################

# Load input file, and read in a few crucial parameters
config = configparser.ConfigParser(interpolation=None,converters={'list': lambda x: [i.strip() for i in x.split(',')]})
if not os.path.exists(paramfile):
    raise Exception("Parameter file does not exist!")
config.read(paramfile)

string = config['sample']['type']

# Output directory
outdir = str(config['directories']['output'])

# Redshifts
ZMIN, ZMAX = float(config['sample']['z_min']), float(config['sample']['z_max'])

# Fiducial cosmological parameters
h_fid, OmegaM_fid = float(config['parameters']['h_fid']), float(config['parameters']['OmegaM_fid'])

# Load survey dimensions
boxsize_grid = np.array(config.getlist('sample','box'),dtype=float)
if spec_type=='pk':
    grid_3d = np.array(config.getlist('pk-binning','grid'),dtype=int)
elif spec_type=='bk':
    grid_3d = np.array(config.getlist('bk-binning','grid'),dtype=int)
else:
    raise Exception("Spectrum type must be 'pk' or 'bk'!")

# Input mask file
maskfile = str(config['catalogs']['mask_file'])
if not os.path.exists(maskfile):
    raise Exception("Mask file does not exist!")

# Create directories
if not os.path.exists(outdir): os.makedirs(outdir)

# Summarize parameters
print("\n###################### PARAMETERS ######################\n")
print("Data-type: %s"%string)
print("\nFiducial h = %.3f"%h_fid)
print("Fiducial Omega_m = %.3f"%OmegaM_fid)
print("Redshift = [%.3f, %.3f]"%(ZMIN,ZMAX))
print("\nBoxsize = [%.3e, %.3e, %.3e] Mpc/h"%(boxsize_grid[0],boxsize_grid[1],boxsize_grid[2]))
print("Grid = [%d, %d, %d]"%(grid_3d[0],grid_3d[1],grid_3d[2]))
print("Output Directory: %s"%outdir)
print("\n########################################################")

init = time.time()

###################################### LOAD DATA ###############################

outfile = outdir+'nbar_%sz%.3f_%.3f_g%d_%d_%d.npy'%(string,ZMIN,ZMAX,grid_3d[0],grid_3d[1],grid_3d[2])

### First check if mask file exists
if os.path.exists(outfile):
    print("Mask already completed; exiting!\n")
    sys.exit()

# Cosmology for co-ordinate conversions
cosmo_coord = cosmology.Cosmology(h=h_fid).match(Omega0_m=OmegaM_fid)

print("\n## Loading data and randoms")

# Load data to get co-ordinate grids and random properties
data = load_data(1,config,cosmo_coord,fkp_weights=False)

randoms = load_randoms(config,cosmo_coord,fkp_weights=False)
diff, density = grid_data(data, randoms, boxsize_grid,grid_3d,MAS='TSC',return_randoms=False,return_norm=False)
k_grids, r_grids = load_coord_grids(boxsize_grid, grid_3d, density)

# Compute \vec{r} grid
X_grid,Y_grid,Z_grid = r_grids
pos_grid = np.vstack([X_grid.ravel(),Y_grid.ravel(),Z_grid.ravel()]).T

# Convert to sky co-ordinates
print("\n## Creating RA, Dec, z co-ordinate grid")
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
nbar_z_interp = UnivariateSpline(z_av,nz/volz,s=0.0000001,ext='zeros')

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
rescale = np.sum(randoms['WEIGHT']).compute()*v_cell/np.sum(nbar_mask)
nbar_mask = nbar_mask*rescale

################################# SAVE AND EXIT ################################

np.save(outfile,nbar_mask)

duration = time.time()-init
print("Output map saved to %s. Exiting after %d seconds (%d minutes)\n"%(outfile,duration,duration//60))
