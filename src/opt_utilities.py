# opt_utilities.py (Oliver Philcox, 2021)
### Various utility functions for the optimal estimators.

import sys, os, copy, time, pyfftw
import matplotlib.pyplot as plt
import numpy as np
from nbodykit.lab import *
from nbodykit import setup_logging, style
from scipy.interpolate import interp1d
import sympy as sp

#### Input directories (should be user-set)
boss_data_dir = '/projects/QUIJOTE/Oliver/boss_gal/' # location of galaxy files
patchyN_data_dir = '/projects/QUIJOTE/Oliver/patchy_ngc_mocks/' # location of Patchy ngc mocks
patchyS_data_dir = '/projects/QUIJOTE/Oliver/patchy_sgc_mocks/' # location of Patchy sgc mocks
mask_dir = '/projects/QUIJOTE/Oliver/boss_masks/' # location of nbar files, computed by generate_mask.py

########################### HANDLING DATA ###########################

def load_data(sim_no,ZMIN,ZMAX,cosmo,patch='ngc',fkp_weights=False,P_fkp=1e4,weight_only=False):
    """
    Load in BOSS/Patchy data with specified patch, z cut, and cosmology.
    
    Parameters
    ----------
    sim_no : int
        Simulation number. -1 gives BOSS, else Patchy is used
    ZMIN : float
        Minimum redshift
    ZMAX : float
        Maximum redshift
    cosmo : Cosmology
        Nbodykit Cosmology object
    patch : str
        Patch: 'ngc' or 'sgc'
    fkp_weights : bool
        Whether to add FKP weights (not usually needed)
    P_fkp : float
        FKP weight (default 1e4)
    weight_only : bool
        If true, return only the weights

    Returns
    -------
    data : Catalog
        Nbodykit catalog containing the data
    """
    if sim_no==-1:
        # BOSS FITS File
        if patch=='ngc':
            datfile = boss_data_dir+'/galaxy_DR12v5_CMASSLOWZTOT_North.fits'
        elif patch=='sgc':
            datfile = boss_data_dir+'/galaxy_DR12v5_CMASSLOWZTOT_South.fits'
        else:
            raise Exception("Wrong patch!")
        data = FITSCatalog(datfile)
    else:
        # Patchy input
        if patch=='ngc':
            datfile = patchyN_data_dir+'ngc_mocks/Patchy-Mocks-DR12NGC-COMPSAM_V6C_%s.dat'%str(sim_no).zfill(4)
        elif patch=='sgc':
            datfile = patchyS_data_dir+'sgc_mocks/Patchy-Mocks-DR12SGC-COMPSAM_V6C_%s.dat'%str(sim_no).zfill(4)
        else:
            raise Exception("Wrong patch!")
        data = CSVCatalog(datfile,['RA', 'DEC', 'Z', 'MSTAR', 'NBAR', 'BIAS', 'VETO FLAG', 'FIBER COLLISION'])

    valid = (data['Z'] > ZMIN)&(data['Z'] < ZMAX)
    data = data[valid]

    if sim_no==-1:
        print("Loaded %d %s galaxies from BOSS\n"%(len(data),patch))
    else:
        print("Loaded %d %s galaxies from simulation %d \n"%(len(data),patch,sim_no))

    if sim_no==-1:
        # Add completeness + systematic weights
        data['WEIGHT'] = data['WEIGHT_SYSTOT'] * (data['WEIGHT_NOZ'] + data['WEIGHT_CP'] - 1.)
        data['NBAR'] = data['NZ']
    else:
        # Add the completeness weights
        data['WEIGHT'] = 1.* data['VETO FLAG'] * data['FIBER COLLISION']

    if weight_only: return data['WEIGHT']

    # Convert to Cartesian co-ordinates using the fiducial cosmology
    data['Position'] = transform.SkyToCartesian(data['RA'], data['DEC'], data['Z'], cosmo=cosmo)

    if fkp_weights:
        print("Adding FKP weights!")
        if sim_no!=-1:
            data['WEIGHT_FKP'] = 1./(1.+P_fkp*data['NBAR'])
    else:
        print("No FKP weights!")
        data['WEIGHT_FKP'] = np.ones(len(data['NBAR']))
    return data


def load_randoms(sim_no,ZMIN,ZMAX,cosmo,patch='ngc',fkp_weights=False,P_fkp=1e4,weight_only=False):
    """Load in BOSS/Patchy randoms with specified patch, z cut, and cosmology.
    
    Parameters
    ----------
        sim_no : int
            Simulation number. -1 gives BOSS, else Patchy is used
        ZMIN : float
            Minimum redshift
        ZMAX : float
            Maximum redshift
        cosmo : Cosmology
            Nbodykit Cosmology object
        patch : str
            Patch: 'ngc' or 'sgc'
        fkp_weights : bool
            Whether to add FKP weights (not usually needed)
        P_fkp : float
            FKP weight (default 1e4)
        weight_only : bool
            If true, return only the weights

    Returns
    -------
        randoms : Catalog
            Nbodykit catalog containing the randoms
    """
    if sim_no!=-1:
        if patch=='ngc':
            randfile = patchyN_data_dir+'Patchy-Mocks-Randoms-DR12NGC-COMPSAM_V6C_x50.dat'
        elif patch=='sgc':
            randfile = patchyS_data_dir+'Patchy-Mocks-Randoms-DR12SGC-COMPSAM_V6C_x50.dat'
        else:
            raise Exception("Wrong patch!")
        randoms = CSVCatalog(randfile,['RA', 'DEC', 'Z', 'NBAR', 'BIAS', 'VETO FLAG', 'FIBER COLLISION'])
    else:
        if patch=='ngc':
            randfile = boss_data_dir+'random0_DR12v5_CMASSLOWZTOT_North.fits'
        elif patch=='sgc':
            randfile = boss_data_dir+'random0_DR12v5_CMASSLOWZTOT_South.fits'
        else:
            raise Exception("Wrong patch!")
        randoms = FITSCatalog(randfile)

    # Cut to required redshift range
    valid = (randoms['Z'] > ZMIN)&(randoms['Z'] < ZMAX)
    randoms = randoms[valid]

    print("Loaded %d randoms \n"%len(randoms))

    if sim_no==-1:
        randoms['WEIGHT'] = 1.*randoms['Weight'] # all unity here
        randoms['NBAR'] = randoms['NZ']
    else:
        randoms['WEIGHT'] = 1.*randoms['VETO FLAG']*randoms['FIBER COLLISION']

    if weight_only: return randoms['WEIGHT']

    # Convert to Cartesian co-ordinates using the fiducial cosmology
    randoms['Position'] = transform.SkyToCartesian(randoms['RA'], randoms['DEC'], randoms['Z'], cosmo=cosmo)

    if fkp_weights:
        print("Adding FKP weights!")
        if sim_no!=-1:
            randoms['WEIGHT_FKP'] = 1./(1.+P_fkp*randoms['NBAR'])
    else:
        print("No FKP weights!")
        randoms['WEIGHT_FKP'] = np.ones(len(randoms['NBAR']))
    return randoms

def load_nbar(sim_no,patch,z_type,ZMIN,ZMAX,grid_factor,alpha_ran):
    """Load the smooth n_bar field computed from the angular mask and n(z) function on the grid.
    This has no window effects since it does not involve particle samples.
    It is normalized by the alpha factor = Sum (data weights) / Sum (random weights).
    
    Parameters
    ----------
        sim_no : int
            Simulation number. -1 gives BOSS, else Patchy is used
        patch : str
            Patch: 'ngc' or 'sgc'
        z_type : str
            Redshift bin: 'z1' or 'z3'
        ZMIN : float
            Minimum redshift
        ZMAX : float
            Maximum redshift
        grid_factor : int
            Factor by which to reduce the grid size
        alpha_ran : float
            Ratio of summed data weights to summed random weights. This is used to normalize the nbar field.

    Returns
    -------
        nbar : ndarray
            Array contining the 3D number density field
    """
    if sim_no==-1:
        file_name = mask_dir+'nbar_boss_%s_%s_z%.3f_%.3f_g%.1f.npy'%(patch,z_type,ZMIN,ZMAX,grid_factor)
    else:
        file_name = mask_dir+'nbar_patchy_%s_%s_z%.3f_%.3f_g%.1f.npy'%(patch,z_type,ZMIN,ZMAX,grid_factor)
    if not os.path.exists(file_name):
        raise Exception("n_bar file '%s' has not been computed!"%file_name)

    nbar_map = np.load(file_name)*alpha_ran
    return nbar_map

def grid_data(data, randoms, boxsize_grid, grid_3d, MAS='TSC', return_randoms=True,return_norm=False):
    """Given two nbodykit catalogs, paint the data and randoms to a single mesh, with a defined mass assignment scheme.
    Returns (data - random) and (optionally) random fields.
    Note that the random field is from *discrete* data here. Use load_nbar to get the continuous version!
    
    Parameters
    ----------
        data : Catalog
            Nbodykit catalog containing the data
        randoms : Catalog
            Nbodykit catalog containing the randoms
        boxsize_grid : array
            3-vector specifying the box dimensions
        grid_3d : array
            3-vector specifying the grid dimensions
        MAS : str
            Mass assignment scheme: 'TSC' or 'CIC'
        return_randoms : bool
            If true, return also the discretized random map
        return_norm : bool
            If true, return also the normalization factor

    Returns
    -------
        diff : ndarray
            3D map of the difference between data and randoms
        density : ndarray
            Nbodykit map containing useful attributes
        rand : ndarray (optional)
            3D map of the random field
        norm : float (optional)
            Normalization factor
    """
    
    # combine the data and randoms into a single catalog
    fkp = FKPCatalog(data, randoms,BoxSize=boxsize_grid,nbar='NBAR')

    assert MAS=='TSC'
    mesh = fkp.to_mesh(Nmesh=grid_3d,fkp_weight='WEIGHT_FKP', comp_weight='WEIGHT', window='tsc')

    from pmesh.pm import ComplexField,RealField
    from nbodykit.algorithms.fftpower import project_to_basis, _find_unique_edges

    ### Extract [n_g - n_r]w density field, and normalize
    alpha_norm = data['WEIGHT'].sum()/randoms['WEIGHT'].sum()
    if return_norm:
        norm = 1./np.asarray(alpha_norm*(randoms['NBAR']*randoms['WEIGHT']*randoms['WEIGHT_FKP']**2.).sum())

    def get_compensation(mesh):
        toret = None
        try:
            compensation = mesh._get_compensation()
            toret = {'func':compensation[0][1], 'kind':compensation[0][2]}
        except ValueError:
            pass
        return toret

    ### SET UP FIELDS AND OUTPUT
    first = mesh
    second = mesh

    # clear compensation from the actions
    for source in [first, second]:
        source.actions[:] = []; source.compensated = False
        assert len(source.actions) == 0

    # compute the compensations
    compensation = {}
    for name, mesh in zip(['first', 'second'], [first, second]):
        compensation[name] = get_compensation(mesh)

    pm   = first.pm

    # paint the 1st FKP density field to the mesh (paints: data - alpha*randoms, essentially)
    density = first.compute(Nmesh=pm.Nmesh)

    diff = density.copy().value

    if return_randoms:
        # Also output the random file
        rrand = first['randoms'].to_real_field(normalize=False)*density.attrs['alpha'] #*446051.74/np.sum(randoms['WEIGHT'])
        vol_per_cell = (pm.BoxSize/pm.Nmesh).prod()
        rrand[:] /= vol_per_cell
        rand = rrand.value
        if return_norm:
            return diff, rand, density, norm
        else:
            return diff, rand, density
    else:
        if return_norm:
            return diff, density, norm
        else:
            return diff, density

def grid_uniforms(data, nbar_unif, boxsize_grid, grid_3d, MAS='TSC'):
    """Given a single unwindowed catalog, paint to a grid. Output is an overdensity field.
    Note that the random field is from *discrete* data here. Use load_nbar to get the continuous version!
    
    Parameters
    ----------
        data : Catalog
            Nbodykit catalog containing the data
        nbar_unif : ndarray
            Uniform number density field
        boxsize_grid : array
            3-vector specifying the box dimensions
        grid_3d : array
            3-vector specifying the grid dimensions
        MAS : str
            Mass assignment scheme: 'TSC' or 'CIC'

    Returns
    -------
        diff : ndarray
            3D map of the difference between data and randoms
    """
    # combine the data and randoms into a single catalog
    assert MAS=='TSC'
    mesh = data.to_mesh(Nmesh=grid_3d,BoxSize=boxsize_grid,resampler='tsc')

    from pmesh.pm import ComplexField,RealField
    from nbodykit.algorithms.fftpower import project_to_basis, _find_unique_edges

    ### SET UP FIELDS AND OUTPUT
    first = mesh
    pm = first.pm

    # paint the 1st FKP density field to the mesh (paints: data - alpha*randoms, essentially)
    density = first.compute(Nmesh=pm.Nmesh)

    # Normalize
    diff = density.copy().value
    diff *= len(data)/np.sum(diff)*grid_3d.prod()*1./boxsize_grid.prod()
    diff -= nbar_unif

    return diff

########################### COORDINATES AND MAPS ###########################

def compute_spherical_harmonic_functions(lmax):
    """Compute array of valid spherical harmonic functions.
    
    Parameters
    ----------
        lmax : int
            Maximum spherical harmonic degree

    Returns
    -------
        spherical_harmonics : list
            List of functions that generate real spherical harmonics for all even ell <= lmax and all m.
   
    """
    def get_real_Ylm(l, m):
        """
        Return a function that computes the real spherical harmonic of order (l,m). Taken from nbodykit.

        Parameters
        ----------
            l : int
                Spherical harmonic degree
            m : int
                Spherical harmonic order

        Returns
        -------
            Ylm : function
                Function that computes the real spherical harmonic of order (l,m)
        """

        # make sure l,m are integers
        l = int(l); m = int(m)

        # the relevant cartesian and spherical symbols
        x, y, z, r = sp.symbols('x y z r', real=True, positive=True)
        xhat, yhat, zhat = sp.symbols('xhat yhat zhat', real=True, positive=True)
        phi, theta = sp.symbols('phi theta')
        defs = [(sp.sin(phi), y/sp.sqrt(x**2+y**2)),
                (sp.cos(phi), x/sp.sqrt(x**2+y**2)),
                (sp.cos(theta), z/sp.sqrt(x**2 + y**2+z**2))]

        # the normalization factors
        if m == 0:
            amp = sp.sqrt((2*l+1) / (4*numpy.pi))
        else:
            amp = sp.sqrt(2*(2*l+1) / (4*numpy.pi) * sp.factorial(l-abs(m)) / sp.factorial(l+abs(m)))

        # the cos(theta) dependence encoded by the associated Legendre poly
        expr = (-1)**m * sp.assoc_legendre(l, abs(m), sp.cos(theta))

        # the phi dependence
        if m < 0:
            expr *= sp.expand_trig(sp.sin(abs(m)*phi))
        elif m > 0:
            expr *= sp.expand_trig(sp.cos(m*phi))

        # simplify
        expr = sp.together(expr.subs(defs)).subs(x**2 + y**2 + z**2, r**2)
        expr = amp * expr.expand().subs([(x/r, xhat), (y/r, yhat), (z/r, zhat)])
        Ylm = sp.lambdify((xhat,yhat,zhat), expr, 'numexpr')

        # attach some meta-data
        Ylm.expr = expr
        Ylm.l    = l
        Ylm.m    = m

        return Ylm

    Y_lm_out = []
    for l in np.arange(0,lmax+1,2):
        Y_m_out = []
        for m in np.arange(-l,l+1,1):
            Y_m_out.append(get_real_Ylm(l,m))
        Y_lm_out.append(np.asarray(Y_m_out))
    return Y_lm_out

def load_coord_grids(boxsize_grid, grid_3d, density):
    """Load Fourier- and real-space co-ordinate grids
    
    Parameters
    ----------
        boxsize_grid : array
            3-vector specifying the box dimensions
        grid_3d : array
            3-vector specifying the grid dimensions
        density : Nbodykit map  
            Density field from nbodykit

    Returns
    -------
        k_grid : ndarray
            Fourier-space co-ordinate grid (kx, ky, kz)
        r_grid : ndarray
            Real-space co-ordinate grid (x, y, z)
    """
    # Fourier-space coordinate grid
    kF = 2.*np.pi/(1.*boxsize_grid)
    middle_3d = np.asarray(grid_3d)/2
    kx, ky, kz = [np.asarray([kkk-grid_3d[i] if kkk>=middle_3d[i] else kkk for kkk in np.arange(grid_3d[i])])*kF[i] for i in range(3)]
    kNy = kF*grid_3d/2.
    k3y,k3x,k3z = np.meshgrid(ky,kx,kz)
    print("k_Nyquist = %.2f h/Mpc"%kNy.mean())

    # the real-space grid
    offset = density.attrs['BoxCenter']+0.5*boxsize_grid/grid_3d
    x_grid,y_grid,z_grid = [xx.real.astype('f8').ravel() + offset[ii].real for ii, xx in enumerate(density.slabs.optx)]
    r3y,r3x,r3z = np.meshgrid(y_grid,x_grid,z_grid)
    return np.asarray([k3x,k3y,k3z]),np.asarray([r3x,r3y,r3z])

def load_MAS(boxsize_grid, grid_3d):
    """Load the mass-assignment scheme (aka compensation) matrix.
    
    Parameters
    ----------
        boxsize_grid : array
            3-vector specifying the box dimensions
        grid_3d : array
            3-vector specifying the grid dimensions

    Returns
    -------
        MAS : ndarray
            Mass assignmment scheme matrix
    """
    kF = 2.*np.pi/(1.*boxsize_grid)
    middle_3d = np.asarray(grid_3d)/2
    kx, ky, kz = [np.asarray([kkk-grid_3d[i] if kkk>=middle_3d[i] else kkk for kkk in np.arange(grid_3d[i])])*kF[i] for i in range(3)]
    prefact = np.pi/np.asarray(grid_3d,dtype=np.float32)
    kkx = kx/kF[0]
    kky = ky/kF[1]
    kkz = kz/kF[2]
    def mas_1d(k):
        """TSC MAS scheme including first-order alias corrections (copied from nbodykit)"""
        s = np.sin(k)**2.
        v = 1./(1.-s+2./15*s**2.)**0.5
        return v

    MAS_arrx = mas_1d(prefact[0]*kkx)
    MAS_arry = mas_1d(prefact[1]*kky)
    MAS_arrz = mas_1d(prefact[2]*kkz)
    MAS_mat = np.meshgrid(MAS_arry,MAS_arrx,MAS_arrz)
    return MAS_mat[1]*MAS_mat[0]*MAS_mat[2]

def compute_filters(kmin,kmax,dk):
    """Load k-space filters, picking out k in [k_i-dk/2,k_i+dk/2] for each bin.
    
    Parameters
        ----------
        kmin : float
            Minimum k-value
        kmax : float
            Maximum k-value
        dk : float
            Width of each k-bin

    Returns
    -------
        filters : ndarray
            3-vector of boolean k-space filters
    """
    # define k-binning
    k_all = np.arange(kmin,kmax+dk,dk)
    k_lo = k_all[:-1]
    k_hi = k_all[1:]
    return lambda i, k_norm: np.logical_and(k_norm>=k_lo[i],k_norm<k_hi[i])

########################### FOURIER TRANSFORMS ###########################

def ft(pix, threads=4):
    """This function performs the 3D FFT of a field in single precision using pyfftw.
    
        Parameters
    ----------
        pix : ndarray
            3D field to transform
        threads : int
            Number of threads to use (default=4)

    Returns
    -------
        pix_ft : ndarray
            Fourier transform of pix
    """
    # align arrays
    grid_3d = pix.shape
    a_in  = pyfftw.empty_aligned(grid_3d,dtype='complex64')
    a_out = pyfftw.empty_aligned(grid_3d,dtype='complex64')

    # plan FFTW
    fftw_plan = pyfftw.FFTW(a_in, a_out, axes=(0,1,2),
                            flags=('FFTW_ESTIMATE',),
                            direction='FFTW_FORWARD', threads=threads)

    # put input array into delta_r and perform FFTW
    a_in [:] = pix;  fftw_plan(a_in,a_out);  return a_out

def ift(pix, threads=4):
    """This function performs the 3D inverse FFT of a field in single precision using pyfftw.
    
    Parameters
    ----------
        pix : ndarray
            3D field to transform
        threads : int
            Number of threads to use (default=4)

    Returns
    -------
        pix_ifft : ndarray
            Inverse Fourier transform of pix
    """
    grid_3d = pix.shape
    a_in  = pyfftw.empty_aligned(grid_3d,dtype='complex64')
    a_out = pyfftw.empty_aligned(grid_3d,dtype='complex64')

    # plan FFTW
    fftw_plan = pyfftw.FFTW(a_in, a_out, axes=(0,1,2),
                            flags=('FFTW_ESTIMATE',),
                            direction='FFTW_BACKWARD', threads=threads)

    # put input array into delta_r and perform FFTW
    a_in [:] = pix;  fftw_plan(a_in,a_out);  return a_out

########################### PLOTTING ###########################

def plotter(mat,axis=2,shift=1,v1=None,v2=None):
    """General purpose function for plotting a 3D density field, averaging over one dimension.
    
        Parameters
    ----------
        mat : ndarray
            3D density field to be plotted
        axis : int
            Axis to average over (default=2)
        shift : bool
            Whether to apply an FFTshift operation (default=True)
        v1 : float
            Minimum value to plot (default=None)
        v2 : float
            Maximum value to plot (default=None)

    """
    plt.figure()
    if shift:
        plot_mat = np.fft.fftshift(mat)
    else:
        plot_mat = mat
    if v1 is not None:
        if v2==None:
            v2 = -v1
        plt.imshow(plot_mat.real.mean(axis=axis),vmax=v1,vmin=v2)
    else:
        plt.imshow(plot_mat.real.mean(axis=axis))
    plt.colorbar();
    plt.show();
