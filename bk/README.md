# Windowless Bispectra

### Overview

To compute bispectrum multipoles, there are two steps:
1. Run the ```compute_bk_randoms.py``` script to generate and analyze randomly distributed particles. These are used to compute the Fisher matrix and q-alpha term, encoding the survey geometry. This script should be run with around 100 choices of input parameter ```rand_it```.
3. Run the ```compute_bk_data.py``` script to analyze a specific simulation or dataset. Step (1) must be computed before this is run.

The output bispectra will be saved as ```bk_{TYPE}{SIM_NO}....txt``` in the specified output directory. Note that the background density map ```n(r)``` must be generated before these scripts are run; this can be done using the [generate_mask.py](../generate_mask.py) script.

### Input Parameters
On the command line, the following parameters can be specified:
- ```rand_it```: Index of the Gaussian-random simulation to be created and analyzed (for ```compute_bk_randoms.py```).
- ```sim_no```: Simulation number (for ```compute_bk_data.py```). If analyzing an unlabelled simulation, or observational data, this is set to -1.
- ```paramfile```: Parameter file specifying various settings, including the survey mask and dimensions. See [paramfiles](../paramfiles) for examples.

### Data

In the [data/](data) directory, we give the raw unwindowed bispectrum measurements of BOSS, 2048 MultiDark-Patchy simulations and 84 Nseries simulations. Further details of the input parameters can be found in the file headers. Note that we remove any bin triplets that are not properly corrected for the survey geometry. The original k-binning limits given in the final header. Note also that these results were run with an earlier version of the code, thus follow slightly different naming conventions.
