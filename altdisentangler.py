import numpy as np
import time

from disentangler_utils import *

def trunc_svd(M, k):
    """
    Truncated rank-k SVD of matrix M.
    """
    U, s, Vh = scp.linalg.svd(M, full_matrices=False, lapack_driver='gesvd')

    Mk = U[:, :k] @ np.diag(s[:k]) @ Vh[:k, :]

    return Mr, U, s, Vh
    
def alt_disentangler(X, l, r, b, c, k, Q0=None, max_iter=500, ftol=1e-8, tol=1e-8):
    """
    Run the alternating disentangler on matrix X given target rank k.
    
    Parameters
    ----------
    X : real-valued array of dimension lr x bc
    l, r, b, c : int
        size l x r x b x c of the 4-dimensional tensor (where X is a 
        particular unfolding)
    k : int
        target rank
    Q0 : lr x lr orthogonal matrix, optional
        initial disentangler, defaults to identity 
    max_iter : int, optional
        maximum number of disentangler iterations
    ftol : float, optional
        absolute convergence based on change in function value
    tol : float, optional
        absolute convergence based on change in disentangler norm
    
    Returns
    -------
    results : dict of intermediate and final outputs
    """
    Qs = np.zeros([max_iter, l*r, l*r]) # array of intermediate disentanglers

    if Q0 is None:
        Qs[0] = np.eye(l*r)
    else:
        Qs[0] = Q0
        
    dQs = np.zeros([max_iter-1]) # change in disentangler norm
    
    ts = np.zeros(max_iter) # wall time
    fvals = np.zeros(max_iter) # function values
    
    AQX = A(Qs[0] @ X, l, r, b, c)
    Mk, _, s, Vh = trunc_svd(AQX, k)

    fvals[0] = np.sum(s[k:]**2)
    
    svs = np.zeros([max_iter, len(s)]) # singular values
    svs[0, :] = s

    tic = time.time()
    for i in range(1, max_iter):
    
        # procrustes
        M = A_inv(Mk, l, r, b, c) @ np.conj(X).T
        U, _, Vh = scp.linalg.svd(M, full_matrices=False, lapack_driver='gesvd')
        
        Qs[i] = U @ Vh
        dQs[i-1] = np.linalg.norm(Qs[i] - Qs[i-1])

        AQX = A(Qs[i] @ X, l, r, b, c)
        Mk, _, s, Vh = trunc_svd(AQX, k)
    
        fvals[i] = np.sum(s[k:]**2)
        svs[i, :] = s
        
        toc = time.time()
        ts[i] = toc - tic

        # display progress
        print('iteration %d: dQ = %e, fval = %e'%(i, dQs[i-1], fvals[i]))

        if dQs[i-1] < tol or np.abs(fvals[i]-fvals[i-1]) < ftol:
            print('convergence reached')
            break
            
    result = {}
    result['point'] = Qs[i] # final disentangler
    result['ts'] = ts[:i+1]
    result['Qs'] = Qs[:i+1]
    result['dQs'] = dQs[:i]
    result['fvals'] = fvals[:i+1]
    result['svs'] = svs[:i+1]
    result['sv_after'] = s # final singular value distribution
    result['i'] = i

    return result
