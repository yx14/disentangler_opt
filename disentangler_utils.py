import numpy as np
import scipy as scp

def get_s(Q, X, l, r, b, c):
    """
    Get singular values after applying disentangler Q to unfolding matrix X.

    Parameters
    ----------
    Q : lr x lr orthogonal matrix (the disentangler)
    X : real-valued array of dimension lr x bc
    l, r, b, c : int
        size l x r x b x c of the 4-dimensional tensor (where X is a 
        particular unfolding)

    Returns
    -------
    s : singular value distribution of A(QX)
    """
    QX = Q @ X
    AQX = A(QX, l, r, b, c)

    U, s, Vh = scp.linalg.svd(AQX, full_matrices=False, lapack_driver='gesvd')

    return s
            
def save_result(filename, result, X, l, r, b, c, save_svs=False):
    """
    Save results of Riemannian optimization to 'filename.'
    """
    time = np.array(result.log['iterations']['time']) - result.log['iterations']['time'][0]
    cost = np.array(result.log['iterations']['cost'])
    gnorm = np.array(result.log['iterations']['gradient_norm'])

    sv_before = get_s(result.log['iterations']['point'][0], X, l, r, b, c)

    sv_after = get_s(result.point, X, l, r, b, c)

    if save_svs:
        svs = np.zeros([len(cost), len(sv_before)])
        Qs = result.log['iterations']['point']
        for i, Q in enumerate(Qs):
            svs[i, :] = get_s(Q, X, l, r, b, c)
        np.savez(filename, X=X, Q_after=result.point, svs=svs, sv_before=sv_before, sv_after=sv_after, time=time, cost=cost, gnorm=gnorm)
    else:
        np.savez(filename, X=X, Q_after=result.point, sv_before=sv_before, sv_after=sv_after, time=time, cost=cost, gnorm=gnorm)


def proj(X, Q):
    """
    Project X onto the tangent space of the orthogonal matrix manifold at Q.
    """
    return 1/2*(X - Q.dot(X.T.dot(Q)))

def A(X, l, r, b, c):
    """
    Reshape and permute a matrix X, with dimension lr x bc,
    to a matrix with dimension lc x rb.
    """
    t = X.reshape([l, r, b, c])
    t_p = t.transpose([0, 3, 1, 2]) # l, c, r, b
    
    return t_p.reshape([l*c, r*b])

def A_inv(mat, l, r, b, c):
    """
    Reshape and permute a matrix of dimension lc x rb to lr x bc.
    """
    t = mat.reshape([l, c, r, b])
    t_p = t.transpose([0, 2, 3, 1])
    
    return t_p.reshape([l*r, b*c])
 
def get_grad_euc(Q, X, l, r, b, c, k):
    # Find the Euclidean gradient of the truncation error objective function c_k:
    # Let m = min(lc, rb).
    # c_k \equiv \sum_{i=k}^{m-1} (s_i)^2
    
    # compute the singular values
    QX = Q.dot(X)
    AQX = A(QX, l, r, b, c)
    U, s, Vh = scp.linalg.svd(AQX, full_matrices=False, lapack_driver='gesvd')
    
    # set the singular values before k to zero 
    s[:k] = 0.
    
    grad_euc = A_inv(U @ np.diag(2*s) @ Vh, l, r, b, c).dot(X.T)
    
    # projection
    return grad_euc

def get_grad(Q, X, l, r, b, c, k):
    # Find the Riemannian gradient of the truncation error objective function c_k:
    # Let m = min(lc, rb).s
    # c_k = \sum_{i=k}^{m-1} (s_i)^2
    
    grad_euc = get_grad_euc(Q, X, l, r, b, c, k)
    
    # projection onto tangent space of orthogonal matrix manifold at Q
    return proj(grad_euc, Q)

def build_F(s):
    """
    Helper function for computing Hessian matvec for c_k.
    """
    F = np.zeros([len(s), len(s)])
    
    for i in range(len(s)):
        for j in range(len(s)):
            if i == j:
                continue
            F[i, j] = 1/(s[j]**2 - s[i]**2)
            
    return F

def get_hess_matvec(E, Q, X, l, r, b, c, k):
    """
    Riemannian Hessian matvec for c_k, applied to E.
    """
    # compute the singular values
    QX = Q.dot(X)
    AQX = A(QX, l, r, b, c)
    U, s, Vh = scp.linalg.svd(AQX, full_matrices=False, lapack_driver='gesvd')
    
    dfds = np.zeros(s.shape)
    dfds[k:] = 2*s[k:]

    m = U.shape[0]
    k = U.shape[1]
    n = Vh.shape[1]
    
    AEX = A(E.dot(X), l, r, b, c)
    
    F = build_F(s)
    
    ######## left term #########
    
    DUE = U @ (F*(U.T @ AEX @ Vh.T @ np.diag(s) + np.diag(s) @ Vh @ AEX.T @ U)) + \
          (np.eye(m) - U @ U.T) @ AEX @ Vh.T @ np.diag(1/s)
    
    d2fds2 = np.zeros(k) # a vector
    d2fds2[k:] = 2
    
    # a matrix 
    Ds = np.diag(U.T @ AEX @ Vh.T) # also a vector
    
    # element-wise multiply, turn into diagonal matrix as needed
    DdfE = d2fds2*Ds
    
    DVE = Vh.T @ (F*(np.diag(s) @ U.T @ AEX @ Vh.T + Vh @ AEX.T @ U @ np.diag(s))) + \
          (np.eye(n) - Vh.T @ Vh) @ AEX.T @ U @ np.diag(1/s)
    
    Dgrad_euc = A_inv(DUE @ np.diag(dfds) @ Vh, l, r, b, c) + \
                 A_inv(U @ np.diag(DdfE) @ Vh, l, r, b, c) + \
                 A_inv(U @ np.diag(dfds) @ DVE.T, l, r, b, c)
    
    left = Dgrad_euc @ X.T
    
    ######## right term ########
        
    grad_euc = get_grad_euc(Q, X, l, r, b, c, k)
    
    right = E @ grad_euc.T @ Q + Q @ left.T @ Q + Q @ grad_euc.T @ E
    
    return proj(0.5*(left - right), Q)

########## Other objective functions ##########

def get_grad_euc_renyi(Q, X, l, r, b, c, alpha):
    # Find the Euclidean gradient of the Renyi-alpha entropy, S^{\alpha}:
    # Let m = min(lc, rb).
    # S^{\alpha} \equiv 1/(1-\alpha) ln(\sum_{i=0}^{m-1} s_i^{2\alpha})

    # compute the singular values
    QX = Q.dot(X)
    AQX = A(QX, l, r, b, c)
    U, s, Vh = scp.linalg.svd(AQX, full_matrices=False, lapack_driver='gesvd')
        
    fac = 2*alpha/(1-alpha)/np.sum(s**(2*alpha))
    s = fac*s**(2*alpha - 1)

    grad_euc = A_inv(U @ np.diag(s) @ Vh, l, r, b, c).dot(X.T)
    
    return grad_euc

def get_grad_euc_vn(Q, X, l, r, b, c):
    # Find the Euclidean gradient of the von Neumann entropy, S^{vN}:
    # Let m = min(lc, rb).
    # S^{vN} \equiv -\sum_{i=0}^{m-1} s_i^2 ln(s_i^2)
    
    # compute the singular values
    QX = Q.dot(X)
    AQX = A(QX, l, r, b, c)
    U, s, Vh = scp.linalg.svd(AQX, full_matrices=False, lapack_driver='gesvd')

    s = -2*s*(np.log(s**2) + 1)

    grad_euc = A_inv(U @ np.diag(s) @ Vh, l, r, b, c).dot(X.T)
    
    return grad_euc

