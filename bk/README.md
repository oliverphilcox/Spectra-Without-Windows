# Windowless BOSS Bispectra

**STILL IN PROGRESS**

### Overview

To compute bispectra, there are two steps:
1. Run the ```compute_bk_randoms.py``` script to generate and analyze uniformly distributed particles. These are used to compute the Fisher and average q-alpha term matrix. This script should be run with 50-100 choices of input parameter ```rand_it```.
2. Run the ```compute_bk_data.py``` script to analyze a specific Patchy simulation or BOSS data. Step (1) must be computed before this is run.

The output power spectrum will be saved as ```bk_patchy{SIM_NO}....txt``` or ```bk_boss....txt``` in the specified output directory.

### Input Parameters
On the command line, the following parameters can be specified:
- ```rand_it```: Index of the uniform simulation to be created and analyzed (for ```compute_bk_randoms.py```).
- ```sim_no```: Patchy simulation number (for ```compute_bk_data.py```). If set to -1, the true BOSS data is analyzed.
- ```patch```: Which region of BOSS to use, either ```ngc``` or ```sgc```.
- ```z_type```: Which redshift region, either ```z1``` or ```z3```.
- ```wtype```: Flag to indicate the type of weights. If set to 0, we use FKP-like weights, else maximum likelihood (ML) weights if set to 1. Note that both weights give unwindowed, pixelation-corrected power spectra.
-  ```grid_factor```: Factor by which to inflate the pixel size, relative to the original BOSS release.

Within the code we can specify the following additional parameters:
- ```N_mc```: Number of Monte Carlo simulations to compute bias and Fisher matrices.
- ```k_min```, ```k_max```, ```dk```: Desired (linear) k-binning strategy.
- ```h_fid```, ```OmegaM_fid```: Fiducial parameters to use when converting redshifts and angles into Cartesian co-ordinates.
- ```tmpdir```: Directory to hold temporary output for each uniform simulation. This should have fast I/O and be large. It will be cleaned at the end of the ```compute_bk_randoms.py``` script.
- ```mcdir```:  Directory to hold temporary Monte Carlo summations. This should be large, and can be removed after ```compute_bk_data.py``` has been run at least once.
- ```lockdir```: Directory to hold some flag files to ensure atomic file writing. This can be removed after the ```compute_bk_randoms.py``` routines are complete.
- ```outdir```: Output directory for saving bispectra. Additional auxilliary data including bias and Fisher matrix terms are also stored.

### Data

In the ```data/``` directory, we give the raw unwindowed bispectrum measurements of BOSS and 2048 MultiDark-Patchy simulations. Further details of the input parameters can be found in the file headers.
