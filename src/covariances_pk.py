# covariances_pk.py (Oliver Philcox, 2021)
### Covariance definitions for optimal estimator P_ell(k) and bispectrum estimation

import sys, os, copy, time
import numpy as np
# custom definitions
sys.path.append('../src')
from opt_utilities import ft, ift

def applyC(input_map,nbar,MAS_mat,pk_map,Yk_lm,Yr_lm,v_cell,shot_fac,use_MAS=True):
    """Apply the fiducial covariance to a pixel map x, i.e. C[x] = S[x]+N[x].

    We decompose P(k;x) = \sum_l P_l(k) L_l(k.x) where x is the position of the second galaxy and use spherical harmonic decompositions.
    P_l(k) are the even fiducial power spectrum multipoles, taken as an input (including the MAS window if relevant).
    We also input the Fourier- and configuration-space real spherical harmonics, as well as the cell volume."""
    return applyS(input_map,nbar,MAS_mat,pk_map,Yk_lm,Yr_lm,v_cell,use_MAS=use_MAS)+applyN(input_map,nbar,MAS_mat,v_cell,shot_fac,use_MAS=use_MAS)

def applyS(input_map,nbar,MAS_mat,pk_map,Yk_lm,Yr_lm,v_cell,use_MAS=True):
    """Apply S optionally including MAS effects. This includes all even multipoles up to len(pk_map).
    We assume that the nbar map is unpixellized."""
    if use_MAS:
        tmp_map = ift(ft(input_map)/MAS_mat)*nbar
    else:
        tmp_map = input_map*nbar
    f_nxP = 0.
    n_l = len(pk_map)
    for i in range(n_l):
        # Compute Sum_m Y_lm(k)FT[Y_lm(r)n(r)x(r)]
        f_nx_l = 0.
        for m_i in range(len(Yk_lm[i])):
            f_nx_l += ft(tmp_map*Yr_lm[i][m_i])*Yk_lm[i][m_i]

        # Add contribution to # Sum_L P_L(k) (4pi)/(2L+1) SUM_M Y_LM*(k) F[n x Y_LM](k)
        f_nxP += 4.*np.pi/(4.*i+1.)*f_nx_l*pk_map[i]

    # Compute Sum_L IFT[P_ell(k) (4pi)/(2L+1) Sum_M Y_LM*(k) F[n x Y_LM](k)](r)
    if use_MAS:
        return ift(ft(nbar*ift(f_nxP))/MAS_mat)/v_cell
    else:
        return nbar*ift(f_nxP)/v_cell

def applyN(input_map,nbar,MAS_mat,v_cell,shot_fac,use_MAS=True):
    """Apply N, optionally including MAS effects. We assume the nbar map does not contain MAS effects.
    shot_fac is equal to [<w_data^2> + alpha^2 < w_randoms^2>]/<w_data>"""
    if use_MAS:
        return shot_fac*ift(1./MAS_mat*ft(nbar*ift(ft(input_map)/MAS_mat)))/v_cell
    else:
        return shot_fac*nbar*input_map/v_cell

def applyCinv_fkp(input_map,nbar,MAS_mat,v_cell,shot_fac,P_fkp=1e4,use_MAS=True):
    """Apply C^-1 in the approximate FKP prescription. This assumes C(r,r') = delta_D(r-r')n(r)[a_shot+n(r)P_fkp].

    We optionally include the full MAS effects also, and use only pixels with (unwindowed) nbar>0."""
    fkp_weight = nbar*(nbar*P_fkp+shot_fac)
    ratio_map = np.zeros(input_map.shape,dtype=np.complex64)
    if use_MAS:
        tmp_map = ift(ft(input_map)*MAS_mat)
    else:
        tmp_map = input_map
    f = nbar>0 # don't include any empty cells!
    ratio_map[f] = tmp_map[f]/fkp_weight[f]
    if use_MAS:
        return ift(MAS_mat*ft(ratio_map))*v_cell
    else:
        return ratio_map*v_cell

def applyCinv_approx(input_map,nbar,MAS_mat,v_cell,shot_fac,P_fkp=1e4,use_MAS=True):
    """Approximate C^-1 based on N^-1, including an FKP rescaling. This is independent of the input cosmology.

     Note we filter out any cells with n_bar = 0
    """
    return applyCinv_fkp(input_map,nbar,MAS_mat,v_cell,shot_fac,P_fkp=P_fkp,use_MAS=use_MAS)

def apply_inv_preconditoner(pix,nbar,MAS_mat,v_cell,shot_fac,P_fkp=1e4,use_MAS=True):
    """Apply the inverse preconditioner matrix, here using the small-scale FKP form"""
    return applyCinv_approx(pix,nbar,MAS_mat,v_cell,shot_fac,P_fkp=P_fkp,use_MAS=use_MAS)

def applyCinv(pix,nbar,MAS_mat,P3D,Yk_lm,Yr_lm,v_cell,shot_fac,applyC=applyC,applyCinv_approx=applyCinv_approx,
              max_it = 100, abs_tol = 1e-8, rel_tol = None, verb=1,
             apply_inv_preconditoner=apply_inv_preconditoner,P_fkp=1e4,use_MAS=True):
    """Solve the system C.x = pix, i.e. x = C^-1 . pix via preconditioned conjugate-gradient descent.
    This uses an input set of P_L(k) multipoles and spherical harmonics to construct the covariance."""
    start = time.time()

    # define initial guess for inverse and first term in sequence
    x = applyCinv_approx(pix,nbar,MAS_mat,v_cell,shot_fac,P_fkp=P_fkp,use_MAS=use_MAS)
    r = pix - applyC(x,nbar,MAS_mat,P3D,Yk_lm,Yr_lm,v_cell,shot_fac,use_MAS=use_MAS)
    p = apply_inv_preconditoner(r,nbar,MAS_mat,v_cell,shot_fac,P_fkp=P_fkp,use_MAS=use_MAS)
    C_p = applyC(p,nbar,MAS_mat,P3D,Yk_lm,Yr_lm,v_cell,shot_fac,use_MAS=use_MAS)

    pre_r = apply_inv_preconditoner(r,nbar,MAS_mat,v_cell,shot_fac,P_fkp=P_fkp,use_MAS=use_MAS)
    old_sum = np.sum(r*pre_r)
    init_sum = old_sum.copy()
    alpha = old_sum/np.sum(p*C_p)
    save_sum = old_sum.copy()

    assert max_it < len(pix.ravel())

    for i in range(max_it):
        if i%10==0 and i>0:
            # Check for stalling and stop if stalled
            if np.abs((save_sum-old_sum)/old_sum)<0.05:
                if verb: print("Inversion stalled after step %d; exiting (ratio %.2e/%.2e)"%(i+1,new_sum,init_sum))
                break
            save_sum = old_sum

        # update x
        x = x+alpha*p
        # update r
        r = r-alpha*C_p
        # update tilde-C^-1.r
        pre_r = apply_inv_preconditoner(r,nbar,MAS_mat,v_cell,shot_fac,P_fkp=P_fkp,use_MAS=use_MAS)
        # update sum(r * tilde-C^-1.r)
        new_sum = np.sum(r*pre_r)
        # update p
        p = pre_r + (new_sum/old_sum)*p
        # Check for convergence
        if new_sum/init_sum<0:
            print("Bad sum: %.2e, %.2e"%(new_sum,init_sum))
        if rel_tol!=None:
            if np.sqrt(new_sum/init_sum)<rel_tol:
                if verb: print("Inversion stopped early after %d iterations (ratio %.2e/%.2e)"%(i+1,new_sum,init_sum))
                break
        else:
            if np.sqrt(new_sum)<abs_tol:
                if verb: print("Inversion stopped early after %d iterations (ratio %.2e/%.2e)"%(i+1,new_sum,init_sum))
                break
        # update sum
        old_sum = new_sum
        # update C.p
        C_p = applyC(p,nbar,MAS_mat,P3D,Yk_lm,Yr_lm,v_cell,shot_fac,use_MAS=use_MAS)
        # compute alpha
        alpha = old_sum/np.sum(p*C_p)
    if i==max_it-1:
        print("CGD did not stop early: is this converged? (ratio %.2e/%.2e)"%(new_sum,init_sum))
    if verb:
        print("\nInversion took %d seconds"%(time.time()-start))
    return x

def applyC_alpha(input_map,nbar,MAS_mat,Yk_lm,Yr_lm,v_cell,k_filters,lmax,use_MAS=True,data=False):
    """Compute derivatives C_{,alpha}(r,r') for the data covariance C. We assume C_D = Sum_alpha C_{,alpha} p_alpha.
    We compute the derivatives with respect to all power spectrum bins and even multipoles, up to ell=lmax.

    This includes pixellation in full if use_MAS=True, else it just multiplies by the MAS window if data=True.
    """
    out_derivs = []

    if use_MAS:
        tmp_map = ift(ft(input_map)/MAS_mat)*nbar
    else:
        tmp_map = input_map*nbar

    n_l = lmax//2+1
    n_k = len(k_filters)

    for i in range(n_l):

        # Compute Sum_m Y_lm(k)FT[Y_lm(r)n(r)x(r)]
        f_nx_l = 0.

        for m_i in range(len(Yk_lm[i])):
            f_nx_l += ft(tmp_map*Yr_lm[i][m_i])*Yk_lm[i][m_i]

        if not use_MAS:
            if data:
                # add in factor of M^2 to remove pixellation effects (for data only)
                f_nx_l *= MAS_mat**2.

        for a in range(n_k):
            # Compute Theta^a(k) (4pi)/(2L+1) SUM_M Y_LM*(k) F[n x Y_LM](k)
            tmp2 = 4.*np.pi/(4.*i+1.)*k_filters[a]*f_nx_l

            # Add to output array
            if use_MAS:
                out_derivs.append(ift(ft(nbar*ift(tmp2))/MAS_mat)/v_cell)
            else:
                out_derivs.append(nbar*ift(tmp2)/v_cell)

    return out_derivs
