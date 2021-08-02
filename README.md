# BOSS Without Windows

This repository contains details of the window-free analysis of BOSS DR12 data. This uses the quadratic and cubic estimators described in [Philcox 2020](https://arxiv.org/abs/2012.09389), and [Philcox 2021](https://arxiv.org/abs/2107.06287).

## Outline
- [pk/](pk): Analysis code to estimate unwindowed power spectra. We additionally supply the raw power spectrum measurements of BOSS and 999 Patchy simulations.
- [bk/](bk): Analysis code to estimate unwindowed bispectra. We additionally supply the raw bispectrum measurements of BOSS and 999 Patchy simulations.
- [src/](src): Various Python utilities used in the unwindowed estimators.
- [generate_mask.py](generate_mask.py): Utility function to generate the background number density, n(r) from the survey mask and n(z) distribution. This is described in the code header.

### Requirements
To run the analysis code one requires:
- python (2 or 3)
- numpy
- scipy
- [sympy](https://www.sympy.org/en/index.html) (for generating spherical harmonics)
- [pyfftw](https://github.com/pyFFTW/pyFFTW) (for FFTs)
- [nbodykit](https://nbodykit.readthedocs.io/en/latest/) (for reading in data)
- [fasteners](https://pypi.org/project/fasteners/) (for providing file overwrites in bispectrum computation)
- [mangle](https://github.com/mollyswanson/manglepy) (for reading survey mask files)

We additionally use a number of the BOSS data products available on the BOSS [SAS](https://data.sdss.org/sas/dr12/boss/lss/). Once downloaded, the locations of these files should be specified in [this file](src/opt_utilities.py).

## Acknowledgements

### Authors
- [Oliver Philcox](mailto:ohep2@cantab.ac.uk) (Princeton / IAS)

### Publications
Codes and data from this repository are used in the following publications:

- Philcox (2020, [Phys Rev. D](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.103.103504), [arXiv](https://arxiv.org/abs/2012.09389)): Description of the unwindowed power spectrum estimators.
- Philcox (2021, submitted to Phys. Rev. D, [arXiv](https://arxiv.org/abs/2107.06287)): Description of the unwindowed bispectrum estimators.
