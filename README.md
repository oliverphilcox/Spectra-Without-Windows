# Spectra Without Windows

This repository contains details of the window-free analysis of non-uniform spectroscopic data. This uses the quadratic and cubic estimators described in [Philcox 2020](https://arxiv.org/abs/2012.09389), [Philcox 2021](https://arxiv.org/abs/2107.06287) and [Ivanov et al. 2023](https://arxiv.org/abs/2302.04414), and additionally contains the data generated for the BOSS power spectrum and bispectrum analysis of [Philcox & Ivanov 2021](https://arxiv.org/abs/2112.04515) and subsequent works.

## Outline
- [pk/](pk): Analysis code to estimate unwindowed power spectrum multipoles. We additionally supply the raw power spectrum measurements of BOSS, 2048 Patchy simulations and 84 Nseries simulations.
- [bk/](bk): Analysis code to estimate unwindowed bispectrum multipoles. We additionally supply the raw bispectrum measurements of BOSS and 2048 Patchy simulations and 84 Nseries simulations.
- [paramfiles/](paramfiles): Example parameter files.
- [src/](src): Various Python utilities used in the unwindowed estimators.
- [slurm/](slurm): Example SLURM submission scripts.
- [generate_mask.py](generate_mask.py): Utility function to generate the background number density, n(r) from the survey mask and n(z) distribution. This is described in the code header.

### Requirements
The scripts in this repository have the following dependencies:
- python (2 or 3)
- numpy
- scipy
- [sympy](https://www.sympy.org/en/index.html) (for generating spherical harmonics)
- [pyfftw](https://github.com/pyFFTW/pyFFTW) (for FFTs)
- [nbodykit](https://nbodykit.readthedocs.io/en/latest/) (for reading in data)
- [mangle](https://github.com/mollyswanson/manglepy) (for reading survey mask files)

When applying the code to the BOSS survey, we use a number of products available on the BOSS [SAS](https://data.sdss.org/sas/dr12/boss/lss/), which are referenced in the relevant [paramfiles](paramfiles).

## Acknowledgements

### Authors
- [Oliver Philcox](mailto:ohep2@cantab.ac.uk) (Columbia / Simons Foundation)

### Publications
Codes and data from this repository are used in the following publications:

- Philcox (2020, [Phys. Rev. D](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.103.103504), [arXiv](https://arxiv.org/abs/2012.09389)): Description of the unwindowed power spectrum estimators.
- Philcox (2021, [Phys. Rev. D](https://doi.org/10.1103/PhysRevD.104.123529), [arXiv](https://arxiv.org/abs/2107.06287)): Description of the unwindowed bispectrum estimators.
- Philcox & Ivanov (2021, [Phys. Rev. D](https://doi.org/10.1103/PhysRevD.105.043517), [arXiv](https://arxiv.org/abs/2112.04515)): Combined constraints on LambdaCDM from the BOSS power spectrum and bispectrum.
- Cabass et al. (2022, [arXiv](https://arxiv.org/abs/2201.07238)): Constraints on single-field inflation from the BOSS power spectrum and bispectrum.
- Cabass et al. (2022, [arXiv](https://arxiv.org/abs/2204.01781)): Constraints on multi-field inflation from the BOSS power spectrum and bispectrum.
- Nunes et al. (2022, [arXiv](https://arxiv.org/abs/2203.08093)): Constraints on dark-sector interactions from the BOSS galaxy power spectrum.
- Rogers et al. (2023, [arXiv](https://arxiv.org/abs/2301.08361)): Ultra-light axions and the S8 tension: joint constraints from the cosmic microwave background and galaxy clustering.
- Ivanov et al. (2023, [arXiv](https://arxiv.org/abs/2302.04414)): Cosmology with the Galaxy Bispectrum Multipoles: Optimal Estimation and Application to BOSS Data.

**NB**: This code formerly appeared as ``BOSS-Without-Windows``, with the BOSS survey specifications hardcoded. For posterity, the original version of the code can be found on this [branch](https://github.com/oliverphilcox/Spectra-Without-Windows/tree/boss-specific-code).
