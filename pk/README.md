# Windowless Power Spectra

### Overview

To compute power spectrum multipoles, there are two steps:
1. Run the ```compute_pk_randoms.py``` script to generate and analyze randomly distributed particles. These are used to compute the Fisher matrix appropriate for given survey geommetry. This script should be run with around 100 choices of input parameter ```rand_it```. 
2. Run the ```compute_pk_data.py``` script to analyze a specific simulation or dataset. Step (1) must be computed before this is run. 

The output power spectrum will be saved as ```pk_{TYPE}{SIM_NO}....txt``` in the specified output directory. Note that the background density map ```n(r)``` must be generated before these scripts are run; this can be done using the [generate_mask.py](../generate_mask.py) script.

### Input Parameters
On the command line, the following parameters can be specified:
- ```rand_it```: Index of the Gaussian-random simulation to be created and analyzed (for ```compute_pk_randoms.py```).
- ```sim_no```: Simulation number (for ```compute_pk_data.py```). If analyzing an unlabelled simulation, or observational data, this is set to -1.
- ```paramfile```: Parameter file specifying various settings, including the survey mask and dimensions. See [paramfiles](../paramfiles) for examples.

### Data

In the [data/](data) directory, we give the raw unwindowed bispectrum multipole measurements of BOSS, 2048 MultiDark-Patchy simulations and 84 Nseries simulations. Further details of the input parameters can be found in the file headers. Note that we remove any k-bins that are not properly corrected for the survey geometry. The original k-binning limits given in the final header.
