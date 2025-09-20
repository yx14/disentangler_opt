import argparse

from disentangler_utils import *
from altdisentangler import *

import pymanopt

# return list of trunc errors
def get_trunc_err(s):
    trunc_err = np.zeros(len(s))
    for i in range(len(s)):
        trunc_err[i] = np.sum(s[i:]**2)
        
    return trunc_err
def main():

    parser = argparse.ArgumentParser(description="Xinary search for disentangling given a tolerance, \
    using the target rank objective function.")
    # binary search parameters
    parser.add_argument('-eps', dest='eps', default=1e-6, type=float, help='requested tolerance for binary search')
    parser.add_argument('-max_iter', dest='max_iter', default=30, help='maximum number of binary search iterations', type=int)
    # tensor parameters
    parser.add_argument('-dim', dest='dim', nargs='+', type=int, help='dimension of tensor, e.g., 2 2 2 2')
    parser.add_argument('-type', dest='type', choices=['randn', 'tfi'], type=str, help='type of tensor')
    # optional arguments
    parser.add_argument('-disentangler', dest='disentangler', choices=['alt', 'cg'], default='cg',
    type=str, help='optimization methods, including alternating (alt) and riemannian conjugate gradient (cg)')
    parser.add_argument('-seed', dest='seed', default=0, type=int, help='seed index for randn tensors')
    parser.add_argument('-alt_max_iter', dest='alt_max_iter', default=10000, type=int, help='maximum number of\
    alt iterations')
    parser.add_argument('-cg_max_iter', dest='cg_max_iter', default=4000, type=int, help='maximum number of\
    cg iterations')
    parser.add_argument('-cg_max_iter_init', dest='cg_max_iter_init', default=4000, type=int, help='maximum number of\
    cg iterations used for the Renyi-1/2 initialization')

    args = parser.parse_args()
    
    filename = './results/binsearch_' 
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
        
        # cg result with renyi-1/2 objective
        renyi_filename = './results/tfi_l%d_r%d_b%d_c%d_renyihalf_cg_max_iter%d_.npz'%(l, r, b, c, 
        args.cg_max_iter_init)
        
    else:
        filename += 'randn_'
        # from running random.randint(0, 2**32 - 1)
        seeds = [1577899298, 2890526676, 3481281077, 2482928114, 168814527]

        rng = np.random.default_rng(seed=seeds[args.seed])

        X = rng.normal(size=(l*r, b*c))

        filename += 'seed%d_'%(args.seed)
  
        X = X/np.linalg.norm(X)
        
        # cg result with renyi-1/2 objective
        renyi_filename = './results/randn_seed%d_l%d_r%d_b%d_c%d_renyihalf_cg_max_iter%d_.npz'%(args.seed, l, r, b, c,
        args.cg_max_iter_init)
    
    filename += 'l%d_r%d_b%d_c%d_'%(l, r, b, c)
    if 'alt' in args.disentangler:
        filename += 'alt_max_iter%d_eps%.2e_bin_iter'%(args.alt_max_iter, args.eps)
    elif 'cg' in args.disentangler:
        filename += 'cg_max_iter%d_eps%.2e_bin_iter'%(args.cg_max_iter, args.eps)

    kl = 0
    
    # find kr 
    test = np.load(renyi_filename)
    kr = np.where(get_trunc_err(test['sv_after']) < args.eps)[0][0]
    kopt = kr
    
    it = 1
    
    while kl <= kr and it < args.max_iter:
        k = int(np.ceil((kl + kr)/2))
        print('on iter %d, k = %d, kl = %d, kr = %d'%(it, k, kl, kr), flush=True)

        ####### optimizers #######
        if 'alt' in args.disentangler:

            print('running alt', flush=True)

            result = alt_disentangler(X, l, r, b, c, k, Q0=np.eye(l*r), max_iter=args.alt_max_iter, ftol=1e-12, tol=1e-12)

            AX = A(X, l, r, b, c)

            U, sv_before, Vh = scp.linalg.svd(AX, full_matrices=False, lapack_driver='gesvd')

            np.savez(filename + str(it), X=X, Q_after=result['point'], Qs=result['Qs'], 
                     sv_before=sv_before, sv_after=result['sv_after'], 
                     time=result['ts'], cost=result['fvals'], dQs=result['dQs'], k=k, kr=kr, kl=kl, kopt=kopt)
            
            # cost of rank
            c_k = result['fvals'][-1]

        elif 'cg' in args.disentangler:

            print('running cg', flush=True)

            manifold = pymanopt.manifolds.SpecialOrthogonalGroup(l*r, retraction='polar')

            @pymanopt.function.numpy(manifold)
            def cost(Q):

                QX = Q @ X
                AQX = A(QX, l, r, b, c)

                U, s, Vh = scp.linalg.svd(AQX, full_matrices=False, lapack_driver='gesvd')

                # objective function
                return np.sum(s[k:]**2)

            @pymanopt.function.numpy(manifold)
            def euclidean_gradient(Q):
                return get_grad_euc(Q, X, l, r, b, c, k)

            problem = pymanopt.Problem(manifold, cost, euclidean_gradient=euclidean_gradient)

            optimizer = pymanopt.optimizers.ConjugateGradient(log_verbosity=1, max_time=3600, 
                                                                  max_iterations=args.cg_max_iter, min_gradient_norm=1e-8)

            # initial point is identity
            result = optimizer.run(problem, initial_point=np.eye(l*r))
               
            time = np.array(result.log['iterations']['time']) - result.log['iterations']['time'][0]
            cost = np.array(result.log['iterations']['cost'])
            gnorm = np.array(result.log['iterations']['gradient_norm'])

            sv_before = get_s(result.log['iterations']['point'][0], X, l, r, b, c)

            sv_after = get_s(result.point, X, l, r, b, c)

            np.savez(filename + str(it), X=X, U=result.point, sv_before=sv_before, sv_after=sv_after, time=time, 
                     cost=cost, gnorm=gnorm, k=k, kr=kr, kl=kl, kopt=kopt)                        
            
            # cost of rank
            c_k = np.array(result.log['iterations']['cost'])[-1]
            
        if c_k <= args.eps:
            kopt = k
            kr = k - 1
        else:
            kl = k + 1
        
        it = it + 1
    
    print('kopt ', kopt)
    return
    
if __name__ == '__main__':
    main()
