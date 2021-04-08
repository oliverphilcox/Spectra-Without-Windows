# BOSS Without Windows

This repository contains details of the window-free analysis of BOSS DR12 data. This uses the quadratic and cubic estimators described in [Philcox 2020](https://arxiv.org/abs/2012.09389), and Philcox 2021 (in prep.).

# TODO:
- Add other patches
- Add n-bar code
- Add compression code?

### Outline
- [pk/](pk): Analysis code to estimate unwindowed power spectra. We additionally supply the raw power spectrum measurements of BOSS and 2048 Patchy simulations.
- [bk/](bk): Analysis code to estimate unwindowed bispectra. We additionally supply the raw bispectrum measurements of BOSS and 2048 Patchy simulations.
- [src/](src): Various Python utilities used in the unwindowed estimators.

### Requirements
To run the analysis code one requires:
- python (2 or 3)
- numpy
- scipy
- sympy
- pyfftw
- nbodykit

We additionally use a number of the BOSS data products available on the BOSS [SAS](https://data.sdss.org/sas/dr12/boss/lss/). Once downloaded, the locations of these files should be specified in [this file](src/opt_utilities.py).

### Authors
- [Oliver Philcox](mailto:ohep2@cantab.ac.uk) (Princeton / IAS)

### Publications
Codes and data from this repository are used in the following publications:

- Philcox (2020, [arXiv](https://arxiv.org/abs/2012.09389)): Description of the unwindowed power spectrum estimators.
- Philcox (2021, in prep.): Description of the unwindowed bispectrum estimators.
