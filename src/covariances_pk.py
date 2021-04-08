# covariances_pk.py (Oliver Philcox, 2021)
### Covariance definitions for optimal estimator P_ell(k) estimation

import sys, os, copy, time
import numpy as np
# custom definitions
sys.path.append('/home/ophilcox/bk_opt/src')
from opt_utilities import ft, ift

def applyC(input_map,nbar,MAS_mat,pk_map,Yk_lm,Yr_lm,v_cell,shot_fac):
    """Apply the fiducial covariance to a pixel map x, i.e. C[x] = S[x]+N[x].

    We decompose P(k;x) = \sum_l P_l(k) L_l(k.x) where x is the position of the second galaxy and use spherical harmonic decompositions.
    P_l(k) are the even fiducial power spectrum multipoles, taken as an input (including the MAS window if relevant).
    We also input the Fourier- and configuration-space real spherical harmonics, as well as the cell volume."""
    return applyS(input_map,nbar,MAS_mat,pk_map,Yk_lm,Yr_lm,v_cell)+applyN(input_map,nbar,MAS_mat,v_cell,shot_fac)

def applyS(input_map,nbar,MAS_mat,pk_map,Yk_lm,Yr_lm,v_cell):
    """Apply S properly including MAS effects. This includes all even multipoles up to len(pk_map).
    We assume that the nbar map """
    tmp_map = ift(ft(input_map)/MAS_mat)*nbar
    f_nxP = 0.
    n_l = len(pk_map)
    for i in range(n_l):
        # Compute Sum_m Y_lm(k)FT[Y_lm(r)n(r)x(r)]
        f_nx_l = 0.
        for m_i in range(len(Yk_lm[i])):
            f_nx_l += ft(tmp_map*Yr_lm[i][m_i])*Yk_lm[i][m_i]

        # Add contribution to # Sum_L P_L(k) (4pi)/(2L+1) SUM_M Y_LM*(k) F[n x Y_LM](k)
        f_nxP += 4.*np.pi/(4.*i+1.)*f_nx_l*pk_map[i]

    # Compute N[x] + Sum_L IFT[P_ell(k) (4pi)/(2L+1) Sum_M Y_LM*(k) F[n x Y_LM](k)](r)
    out = ift(ft(nbar*ift(f_nxP))/MAS_mat)/v_cell
    return out

def applyN(input_map,nbar,MAS_mat,v_cell,shot_fac):
    """Apply N including MAS effects. We assume the nbar map does not contain MAS effects.
    shot_fac is equal to [<w_data^2> + alpha^2 < w_randoms^2>]/<w_data>"""
    return shot_fac*ift(1./MAS_mat*ft(nbar*ift(ft(input_map)/MAS_mat)))/v_cell

def applyCinv_fkp(input_map,nbar,MAS_mat,v_cell,shot_fac,P_fkp=1e4):
    """Apply C^-1 in the approximate FKP prescription. This assumes C(r,r') = delta_D(r-r')n(r)[a_shot+n(r)P_fkp].

    We include the full MAS effects also, and use only pixels with (unwindowed) nbar>0."""
    fkp_weight = nbar*(nbar*P_fkp+shot_fac)
    ratio_map = np.zeros_like(input_map)
    tmp_map = ift(ft(input_map)*MAS_mat)
    f = nbar>0 # don't include any empty cells!
    ratio_map[f] = tmp_map[f]/fkp_weight[f]
    return ift(MAS_mat*ft(ratio_map))*v_cell

def applyCinv_approx(input_map,nbar,MAS_mat,v_cell,shot_fac,P_fkp=1e4):
    """Approximate C^-1 based on N^-1, including an FKP rescaling. This is independent of the input cosmology.

     Note we filter out any cells with n_bar = 0
    """
    tmp_map = ift(ft(input_map)*MAS_mat)
    ratio_map = np.zeros_like(input_map)
    denom = nbar*(shot_fac+P_fkp*nbar)
    filt = nbar>0
    ratio_map[filt] = tmp_map[filt]/denom[filt]
    return ift(ft(ratio_map)*MAS_mat)*v_cell

def apply_inv_preconditoner(pix,nbar,MAS_mat,v_cell,shot_fac,P_fkp=1e4):
    """Apply the inverse preconditioner matrix, here using the small-scale FKP form"""
    return applyCinv_approx(pix,nbar,MAS_mat,v_cell,shot_fac,P_fkp=P_fkp)

def applyCinv(pix,nbar,MAS_mat,P3D,Yk_lm,Yr_lm,v_cell,shot_fac,applyC=applyC,applyCinv_approx=applyCinv_approx,
              max_it = 100, abs_tol = 1e-8, rel_tol = None, verb=1,
             apply_inv_preconditoner=apply_inv_preconditoner,P_fkp=1e4):
    """Solve the system C.x = pix, i.e. x = C^-1 . pix via preconditioned conjugate-gradient descent.
    This uses an input set of P_L(k) multipoles and spherical harmonics to construct the covariance."""
    start = time.time()

    # define initial guess for inverse and first term in sequence
    x = applyCinv_approx(pix,nbar,MAS_mat,v_cell,shot_fac,P_fkp=P_fkp)
    r = pix - applyC(x,nbar,MAS_mat,P3D,Yk_lm,Yr_lm,v_cell,shot_fac)
    p = apply_inv_preconditoner(r,nbar,MAS_mat,v_cell,shot_fac,P_fkp=P_fkp)
    C_p = applyC(p,nbar,MAS_mat,P3D,Yk_lm,Yr_lm,v_cell,shot_fac)

    pre_r = apply_inv_preconditoner(r,nbar,MAS_mat,v_cell,shot_fac,P_fkp=P_fkp)
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
        pre_r = apply_inv_preconditoner(r,nbar,MAS_mat,v_cell,shot_fac,P_fkp=P_fkp)
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
        C_p = applyC(p,nbar,MAS_mat,P3D,Yk_lm,Yr_lm,v_cell,shot_fac)
        # compute alpha
        alpha = old_sum/np.sum(p*C_p)
    if i==max_it-1:
        print("CGD did not stop early: is this converged? (ratio %.2e/%.2e)"%(new_sum,init_sum))
    if verb:
        print("\nInversion took %d seconds"%(time.time()-start))
    return x

def applyC_alpha(input_map,nbar,MAS_mat,Yk_lm,Yr_lm,v_cell,k_filters,lmax):
    """Compute derivatives C_{,alpha}(r,r') for the data covariance C. We assume C_D = Sum_alpha C_{,alpha} p_alpha.
    We compute the derivatives with respect to all power spectrum bins and even multipoles, up to ell=lmax.

    This includes MAS corrections.
    """
    out_derivs = []

    tmp_map = ift(ft(input_map)/MAS_mat)*nbar
    n_l = lmax//2+1
    n_k = len(k_filters)

    for i in range(n_l):

        # Compute Sum_m Y_lm(k)FT[Y_lm(r)n(r)x(r)]
        f_nx_l = 0.

        for m_i in range(len(Yk_lm[i])):
            f_nx_l += ft(tmp_map*Yr_lm[i][m_i])*Yk_lm[i][m_i]

        for a in range(n_k):
            # Compute Theta^a(k) (4pi)/(2L+1) SUM_M Y_LM*(k) F[n x Y_LM](k)
            tmp2 = 4.*np.pi/(4.*i+1.)*k_filters[a]*f_nx_l

            # Add to output array
            out_derivs.append(ift(ft(nbar*ift(tmp2))/MAS_mat)/v_cell)

    return out_derivs
