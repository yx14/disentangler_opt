from .plotting_utils import *

# Wall time: random tensor
def fig8a(data_folder, save_folder):
    n = 12
    seed = 0
    
    rank = 44

    disentanglers = ['Alt', 'RCG', 'RNTR', 'RNTR-reg']
    disentanglers_iter = ['alt_max_iter10000_', 'cg_max_iter4000_',
                          'newton_reg0.00e+00_maxinner_100', 'newton_reg1.00e-12_maxinner_100']
    
    # color and line cycle indices
    c_idxs = [0, 1, 2, 4]
    l_idxs = [0, 1, 2, 3]
    
    fig = plt.figure(figsize=(6,6))

    for i, disentangler in enumerate(disentanglers):
            
        test = np.load(data_folder + 'randn_seed%d_l%d_r%d_b%d_c%d_rank%d_%s.npz'%(seed, 
                                                    n, n, n, n, rank, disentanglers_iter[i]))

        plt.plot(test['time'], test['cost'], 
                 c=color_cycle[c_idxs[i]], label=disentangler, 
                 lw=2, linestyle=l_cycle[l_idxs[i]])
        
    plt.xlabel('Time (s)')
    plt.ylabel('$c_{%d}$'%(rank))
  
    plt.xlim([-4, 120])
    plt.ylim([2e-10, 1])
    plt.yscale('log')
    plt.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.xticks([0, 50, 100])
    plt.tick_params(axis='x', which='major', pad=12)
    plt.tick_params(axis='y', which='major', pad=8)

    plt.savefig(save_folder + 'fig8a.svg', bbox_inches='tight', dpi=1200, transparent=True)