from .plotting_utils import *

# different objective functions for rank estimation: random tensor
def fig11a(data_folder, save_folder):
    n = 12
    seed = 0
    
    disentanglers = ['Renyi-1/2', 'Von Neumann']
    disentanglers_iter = ['renyihalf_cg_max_iter4000_', 'vn_cg_max_iter4000_',]

    # color and line cycle indices
    c_idxs = [0, 2]
    l_idxs = [1, 2]
    
    fig = plt.figure(figsize=(6,6))

    for i, disentangler in enumerate(disentanglers):
            
        test = np.load(data_folder + 'randn_seed%d_l%d_r%d_b%d_c%d_%s.npz'%(seed, 
                                                    n, n, n, n, disentanglers_iter[i]))

        if i == 0:
            plt.plot(range(n*n-1), get_trunc_err(test['sv_before'])[:-1], c=color_cycle[3], 
                     label='Initial', lw=2, linestyle=l_cycle[5])
        
        plt.plot(range(n*n-1), get_trunc_err(test['sv_after'])[:-1], 
                     c=color_cycle[c_idxs[i]], label=disentangler, lw=2, 
                     linestyle=l_cycle[l_idxs[i]])
    
    # plot rank minimization for different target ranks
    ranks = [40, 44, 48]
    
    for rank_idx, rank in enumerate(ranks):
        test = np.load(data_folder + 'randn_seed%d_l%d_r%d_b%d_c%d_rank%d_%s.npz'%(seed, 
                                                    n, n, n, n, rank, 'cg_max_iter4000_'))

        plt.scatter([rank], [test['cost'][-1]], 
                        color=cmap_custom_purple(0.15 + 0.65*rank_idx/2), label='Target Rank $k = %d$'%rank, lw=2)
    
    plt.xlabel('$k$')
    plt.ylabel('$c_k$')
  
    # rank estimation from counting the degrees of freedom
    plt.axvline([44], c=color_cycle[7], linestyle='--', lw=2, zorder=-2)
    plt.yscale('log')
    plt.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tick_params(axis='x', which='major', pad=12)
    plt.tick_params(axis='y', which='major', pad=8)

    plt.savefig(save_folder + 'fig11a.svg', bbox_inches='tight', dpi=1200, transparent=True)