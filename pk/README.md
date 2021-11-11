# Windowless BOSS Power Spectra

### Overview

To compute power spectra, there are two steps:
1. Run the ```compute_pk_randoms.py``` script to generate and analyze uniformly distributed particles. These are used to compute the Fisher matrix appropriate for Patchy simulations. This script should be run with 50-100 choices of input parameter ```rand_it```. We also provide the ```compute_pk_boss_randoms.py``` script to compute the analogous Fisher matrix for the BOSS data (which has a different random catalog).
2. Run the ```compute_pk_data.py``` script to analyze a specific Patchy simulation or BOSS data. Step (1) must be computed before this is run.

The output power spectrum will be saved as ```pk_patchy{SIM_NO}....txt``` or ```pk_boss....txt``` in the specified output directory. Note that the background density map ```n(r)``` must be generated before these scripts are run; this can be done using the [generate_mask.py](../generate_mask.py) script.

### Input Parameters
On the command line, the following parameters can be specified:
- ```rand_it```: Index of the uniform simulation to be created and analyzed (for ```compute_pk_randoms.py```).
- ```sim_no```: Patchy simulation number (for ```compute_pk_data.py```). If set to -1, the true BOSS data is analyzed.
- ```patch```: Which region of BOSS to use, either ```ngc``` or ```sgc```.
- ```z_type```: Which redshift region, either ```z1``` or ```z3```.
- ```wtype```: Flag to indicate the type of weights. If set to 0, we use FKP-like weights, else maximum likelihood (ML) weights if set to 1. Note that both weights give unwindowed, pixelation-corrected power spectra.
-  ```grid_factor```: Factor by which to inflate the pixel size, relative to the original BOSS release.

Within the code we can specify the following additional parameters:
- ```N_mc```: Number of Monte Carlo simulations to compute bias and Fisher matrices.
- ```k_min```, ```k_max```, ```dk```: Desired (linear) k-binning strategy.
- ```l_max```: Maximum (even) multipole to compute.
- ```h_fid```, ```OmegaM_fid```: Fiducial parameters to use when converting redshifts and angles into Cartesian co-ordinates.
- ```pk_input_file```: CSV file containing the fiducial power spectrum multipoles. This is only used for ML weights.
- ```outdir```: Output directory for saving power spectra. Additional auxilliary data including bias and Fisher matrix terms are also stored.
- ```include_pix```: If true, forward model the effects of pixellation on the density field. (NB: the pixel window function is still removed at leading order in both cases).
- ```rand_nbar```: If true, compute the background number density from random particles rather than from the survey mask.
- ```low_mem```: If true, save and reload temporary random files rather than holding them in memory. This gives a significant increase in computation time, but may be useful if memory is limited.

### Data

In the [data/](data) directory, we give the raw unwindowed power spectrum measurements of BOSS, 2048 MultiDark-Patchy simulations and 84 Nseries simulations. Further details of the input parameters can be found in the file headers. Note that we remove any k-bins that are not properly corrected for the survey geometry. The original k-binning limits given in the final header.
