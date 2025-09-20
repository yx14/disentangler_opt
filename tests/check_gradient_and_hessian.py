from disentangler_utils import *

import pymanopt
from pymanopt.tools.diagnostics import check_gradient, check_hessian


def main():
    """
    This test uses pymanopt tools to check the Riemannian gradient of the
    Renyi-1/2 and von Neumann objective functions, as well as the
    Riemannian gradient and Hessian of the truncation error objective function, c_k.

    For example, to check the gradient for c_k given a random tangent vector V, 
    step size t, random orthogonal matrix Q, and retraction R(tV), the difference
    between c_k and its first order approximation is:

        |c_k(QR(tV)) - c_k(Q) - t<grad(Q), V>| ~ O(t^2)

    For a range of t, we would expect to see an error scaling as O(t^2). 

    To run the test, add the parent directory for testing
    export PYTHONPATH=(absolute path to parent):$PYTHONPATH 
    """
    np.random.seed(40)

    l = 4
    r = 4
    b = 6
    c = 6

    X = np.random.randn(l*r, b*c)
    X = X/np.linalg.norm(X)

    k = 4

    manifold = pymanopt.manifolds.SpecialOrthogonalGroup(l*r, retraction='polar')

    print('gradient check for Renyi-1/2 objective function')
    alpha = 0.5
        
    @pymanopt.function.numpy(manifold)
    def cost(Q):

        QX = Q @ X
        AQX = A(QX, l, r, b, c)

        U, s, Vh = np.linalg.svd(AQX, full_matrices=False)

        # objective function
        return 1/(1-alpha)*np.log(np.sum(s**(2*alpha)))

    @pymanopt.function.numpy(manifold)
    def euclidean_gradient(Q):
        return get_grad_euc_renyi(Q, X, l, r, b, c, alpha)

    problem = pymanopt.Problem(manifold, cost, euclidean_gradient=euclidean_gradient)
    check_gradient(problem)

    print('gradient check for von Neumann objective function')
    @pymanopt.function.autograd(manifold)
    def cost_vn(Q):

        QX = Q @ X
        AQX = A(QX, l, r, b, c)

        U, s, Vh = scp.linalg.svd(AQX, full_matrices=False, lapack_driver='gesvd')

        return -2*np.sum(s**2*np.log(s)) 

    @pymanopt.function.numpy(manifold)
    def euclidean_gradient(Q):
        return get_grad_euc_vn(Q, X, l, r, b, c)

    problem = pymanopt.Problem(manifold, cost_vn, euclidean_gradient=euclidean_gradient)
    check_gradient(problem)
    
    print('gradient check for truncation error objective function')
    @pymanopt.function.numpy(manifold)
    def cost(Q):

        QX = Q @ X
        AQX = A(QX, l, r, b, c)

        U, s, Vh = np.linalg.svd(AQX, full_matrices=False)

        return np.sum(s[k:]**2)

    @pymanopt.function.numpy(manifold)
    def euclidean_gradient(Q):
        return get_grad_euc(Q, X, l, r, b, c, k)
    
    @pymanopt.function.numpy(manifold)
    def riemannian_hessian(Q, E):
        fgradx = problem.riemannian_gradient(Q)
        grad_norm = manifold.norm(Q, fgradx)

        return Q.T @ (get_hess_matvec(Q@E, Q, X, l, r, b, c, k))

    problem = pymanopt.Problem(manifold, cost, euclidean_gradient=euclidean_gradient, 
                               riemannian_hessian=riemannian_hessian)
    
    check_gradient(problem)
    
    print('Hessian check for truncation error objective function (not regularized)')
    check_hessian(problem)
    
if __name__ == '__main__':
    main()


