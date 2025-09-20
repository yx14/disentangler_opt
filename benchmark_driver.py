import argparse

from disentangler_utils import *
from altdisentangler import *

import pymanopt
  
def main():

    parser = argparse.ArgumentParser(description="Benchmarking for disentangling optimization methods.")
    parser.add_argument('-dim', dest='dim', nargs='+', type=int, help='dimension of tensor, e.g., 2 2 2 2')
    parser.add_argument('-type', dest='type', choices=['randn', 'tfi'], type=str, help='type of tensor')
    # optional arguments, depending on type, disentangler, and obj
    parser.add_argument('-disentangler', dest='disentangler', default='cg', type=str, choices=['alt', 'cg', 'newton',
    'altcg', 'cgnewton', 'altnewton'], help='optimization methods, including alternating (alt), \
    riemannian conjugate gradient (cg), riemannian trust-region newton (newton), and hybrid combinations')
    parser.add_argument('-obj', dest='obj', default='rank', choices=['rank', 'renyihalf', 'vn'], 
    help='objective function, including target rank, renyi-1/2 entropy, and von neumann entropy')
    parser.add_argument('-seed', dest='seed', default=0, type=int, help='seed index for randn tensors')
    parser.add_argument('-rank', dest='rank', default=2, type=int, help='target rank parameter for rank obj function')
    parser.add_argument('-reg', dest='reg', default=1e-12, type=float,
                        help='regularization constant for newton')
    parser.add_argument('-alt_max_iter', dest='alt_max_iter', default=10000, type=int, help='maximum number of\
    alt iterations')
    parser.add_argument('-cg_max_iter', dest='cg_max_iter', default=4000, type=int, help='maximum number of\
    cg iterations')
    parser.add_argument('-newton_max_iter', dest='newton_max_iter', default=1000, type=int, help='maximum number of\
    newton iterations')
    parser.add_argument('-newton_max_inner', dest='newton_max_inner', default=100, type=int, help='maximum number of\
    inner conjugate gradient iterations within a single (outer) iteration of newton')
    parser.add_argument('--save_svs', dest='save_svs', action='store_true', help='whether to save singular values/iteration')
    
    args = parser.parse_args()
 
    filename = "./results/"
    
    ####### dimensions [l, r, b, c] #######
    if len(args.dim) == 1:
       n = args.dim[0]
       dim = [n, n, n, n]
    elif len(args.dim) == 2:
       m, n = args.dim
       dim = [m, m, n, n]
    elif len(args.dim) == 4:
       dim = args.dim
    else:
       raise ValueError("dim must be [n], [m, n], or [l, r, b, c]")

    l, r, b, c = dim
    
    ####### tensor type #######
    if args.type == 'tfi':
        filename += 'tfi_'
        # [c, l, r, b], where c goes with l and r goes with b after split
        test = np.load('isotns_data/TFI_g3.5_8x8_psi_real_l_%d_r_%d_c_%d_b_%d_chi_12_eta_20.npz'%(l, r, c, b))
        # theta is already normalized
        # transpose to [l, r, b, c] then reshape
        X = test['theta'].transpose(1, 2, 3, 0).reshape([l*r, b*c])
        
    else:
        
        # from running random.randint(0, 2**32 - 1)
        seeds = [1577899298, 2890526676, 3481281077, 2482928114, 168814527]

        rng = np.random.default_rng(seed=seeds[args.seed])

        X = rng.normal(size=(l*r, b*c))

        if args.type == 'randn':
            filename += 'randn_'

        filename += 'seed%d_'%(args.seed)
  
        X = X/np.linalg.norm(X)

    filename += 'l%d_r%d_b%d_c%d_'%(l, r, b, c)
    
    # whether to run von neumann or renyi-half (cg method only)
    if 'renyihalf' in args.obj:
        filename += 'renyihalf_'
       
        if 'cg' not in args.disentangler:
            raise ValueError('only cg is compatible with renyihalf')
            
        print('running cg', flush=True)

        filename += 'cg_max_iter%d_'%(args.cg_max_iter)

        manifold = pymanopt.manifolds.SpecialOrthogonalGroup(l*r, retraction='polar')

        alpha = 0.5
        
        @pymanopt.function.numpy(manifold)
        def cost(Q):

            QX = Q @ X
            AQX = A(QX, l, r, b, c)

            U, s, Vh = scp.linalg.svd(AQX, full_matrices=False, lapack_driver='gesvd')

            # objective function
            return 1/(1-alpha)*np.log(np.sum(s**(2*alpha)))

        @pymanopt.function.numpy(manifold)
        def euclidean_gradient(Q):
            return get_grad_euc_renyi(Q, X, l, r, b, c, alpha)

        problem = pymanopt.Problem(manifold, cost, euclidean_gradient=euclidean_gradient)

        optimizer = pymanopt.optimizers.ConjugateGradient(log_verbosity=1, max_time=3600, 
                                                          max_iterations=args.cg_max_iter, min_gradient_norm=1e-8)

        # initial point is identity
        result = optimizer.run(problem, initial_point=np.eye(l*r))

        save_result(filename, result, X, l, r, b, c, args.save_svs)

        X = result.point @ X # if additional optimizers are run
        
    if 'vn' in args.obj:
        filename += 'vn_'
       
        if 'cg' not in args.disentangler:
            raise ValueError('only cg is compatible with vn')
            
        print('running cg', flush=True)

        filename += 'cg_max_iter%d_'%(args.cg_max_iter)

        manifold = pymanopt.manifolds.SpecialOrthogonalGroup(l*r, retraction='polar')
        
        @pymanopt.function.numpy(manifold)
        def cost(Q):

            QX = Q @ X
            AQX = A(QX, l, r, b, c)

            U, s, Vh = scp.linalg.svd(AQX, full_matrices=False, lapack_driver='gesvd')

            # objective function
            return -2*np.sum(s**2*np.log(s)) 

        @pymanopt.function.numpy(manifold)
        def euclidean_gradient(Q):
            return get_grad_euc_vn(Q, X, l, r, b, c)

        problem = pymanopt.Problem(manifold, cost, euclidean_gradient=euclidean_gradient)

        optimizer = pymanopt.optimizers.ConjugateGradient(log_verbosity=1, max_time=3600, 
                                                          max_iterations=args.cg_max_iter, min_gradient_norm=1e-8)

        # initial point is identity
        result = optimizer.run(problem, initial_point=np.eye(l*r))

        save_result(filename, result, X, l, r, b, c, args.save_svs)

        X = result.point @ X # if additional optimizers are run
        
    ####### target rank objective function #######
    if 'rank' in args.obj:
        filename += 'rank%d_'%(args.rank)
  
        rank = args.rank

        ####### optimizers #######
        if 'alt' in args.disentangler:

            print('running alt', flush=True)

            filename += 'alt_max_iter%d_'%(args.alt_max_iter)

            result = alt_disentangler(X, l, r, b, c, rank, Q0=np.eye(l*r), max_iter=args.alt_max_iter, ftol=1e-12, tol=1e-12)

            AX = A(X, l, r, b, c)

            U, sv_before, Vh = scp.linalg.svd(AX, full_matrices=False, lapack_driver='gesvd')

            if args.save_svs:
                np.savez(filename, X=X, Q_after=result['point'], svs=result['svs'], sv_before=sv_before, sv_after=result['sv_after'], 
                     time=result['ts'], cost=result['fvals'], dQs=result['dQs'])
            else:
                np.savez(filename, X=X, Q_after=result['point'], sv_before=sv_before, sv_after=result['sv_after'], 
                     time=result['ts'], cost=result['fvals'], dQs=result['dQs'])

            X = result['point'] @ X # if additional optimizers are run

        if 'cg' in args.disentangler:

            print('running cg', flush=True)

            filename += 'cg_max_iter%d_'%(args.cg_max_iter)

            manifold = pymanopt.manifolds.SpecialOrthogonalGroup(l*r, retraction='polar')

            @pymanopt.function.numpy(manifold)
            def cost(Q):

                QX = Q @ X
                AQX = A(QX, l, r, b, c)

                U, s, Vh = scp.linalg.svd(AQX, full_matrices=False, lapack_driver='gesvd')

                # objective function
                return np.sum(s[rank:]**2)

            @pymanopt.function.numpy(manifold)
            def euclidean_gradient(Q):
                return get_grad_euc(Q, X, l, r, b, c, rank)

            problem = pymanopt.Problem(manifold, cost, euclidean_gradient=euclidean_gradient)


            optimizer = pymanopt.optimizers.ConjugateGradient(log_verbosity=1, max_time=3600, 
                                                                  max_iterations=args.cg_max_iter, min_gradient_norm=1e-8)

            # initial point is identity
            result = optimizer.run(problem, initial_point=np.eye(l*r))
               
            save_result(filename, result, X, l, r, b, c, args.save_svs)
            
            X = result.point @ X # if additional optimizers are run

        if 'newton' in args.disentangler:

            print('running newton', flush=True)

            filename += 'newton_reg%.2e_maxinner_%d'%(args.reg, args.newton_max_inner)

            manifold = pymanopt.manifolds.SpecialOrthogonalGroup(l*r, retraction='polar')

            @pymanopt.function.numpy(manifold)
            def cost(Q):

                QX = Q @ X
                AQX = A(QX, l, r, b, c)

                U, s, Vh = scp.linalg.svd(AQX, full_matrices=False, lapack_driver='gesvd')

                # objective function
                return np.sum(s[rank:]**2)

            @pymanopt.function.numpy(manifold)
            def euclidean_gradient(Q):
                return get_grad_euc(Q, X, l, r, b, c, rank)

            # regularized hessian
            @pymanopt.function.numpy(manifold)
            def riemannian_hessian(Q, E):
                fgradx = problem.riemannian_gradient(Q)
                grad_norm = manifold.norm(Q, fgradx)

                return Q.T @ (get_hess_matvec(Q@E, Q, X, l, r, b, c, rank) + args.reg)

            problem = pymanopt.Problem(manifold, cost, euclidean_gradient=euclidean_gradient,
                                                       riemannian_hessian=riemannian_hessian)

            optimizer = pymanopt.optimizers.TrustRegions(log_verbosity=1, max_time=3600, 
                                                             max_iterations=args.newton_max_iter, min_gradient_norm=1e-8)

            # initial point is identity
            result = optimizer.run(problem, initial_point=np.eye(l*r), maxinner=args.newton_max_inner)

            save_result(filename, result, X, l, r, b, c, args.save_svs)
            
    return
    
if __name__ == '__main__':
    main()
