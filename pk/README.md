# Windowless BOSS Power Spectra

### Overview

To compute power spectra, there are two steps:
1. Run the ```compute_pk_randoms.py``` script to generate and analyze uniformly distributed particles. These are used to compute the Fisher matrix. This script should be run with 50-100 choices of input parameter ```rand_it```.
2. Run the ```compute_pk_data.py``` script to analyze a specific Patchy simulation or BOSS data. Step (1) must be computed before this is run.

The output power spectrum will be saved as ```pk_patchy{SIM_NO}....txt``` or ```pk_boss....txt``` in the specified output directory.

### Input Parameters
On the command line, the following parameters can be specified:
- ```rand_it```: Index of the uniform simulation to be created and analyzed.
- ```sim_no```: Patchy simulation number. If set to -1, the true BOSS data is analyzed.
- ```wtype```: Flag to indicate the type of weights. If set to 0, we use FKP-like weights, else ML weights if set to 1. Note that both weights give unwindowed, pixelation-corrected power spectra.
-  ```grid_factor```: Factor by which to inflate the pixel size, relative to the original BOSS release.

Within the code we can specify the following additional parameters:
- ```patch```: Which region of BOSS to use, either ```ngc``` or ```sgc```.
- ```z_type```: Which redshift region, either ```z1``` or ```z3```.
- `` `N_bias```: Number of Monte Carlo simulations to compute bias and Fisher matrices.
- ```k_min```, ```k_max```, ```dk```: Desired (linear) k-binning strategy. 
- ```l_max```: Maximum (even) multipole to compute.
- ```h_fid```, ```OmegaM_fid```: Fiducial parameters to use when converting redshifts and angles into Cartesian co-ordinates.
- ```outdir```: Output directory for saving power spectra. Additional auxilliary data including bias and Fisher matrix terms are also stored. 

### Data

In the [data/](data) directory, we give the raw unwindowed power spectrum measurements of BOSS and 2048 MultiDark-Patchy simulations. Further details of the input parameters can be found in the file headers.
