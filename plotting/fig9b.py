from .plotting_utils import *

# singular value spectrum: isoTNS tensor
# for different algorithms, the spectra correspond to approximately the same wall time
def fig9b(data_folder, save_folder):
    l = 8
    r = 8
    b = 60
    c = 12

    rank = 8

    disentanglers = ['Alt', 'RCG', 'RNTR', 'RNTR-reg']
    disentanglers_iter = ['alt_max_iter10000_', 'cg_max_iter4000_', 
                          'newton_reg0.00e+00_maxinner_100', 'newton_reg1.00e-12_maxinner_100']

    # color and line cycle indices
    c_idxs = [0, 1, 2, 4]
    l_idxs = [0, 1, 2, 3]

    fig = plt.figure(figsize=(6,6))
     
    # the final time in Fig 8b is set by Alt's wall time. 
    Alt_wall_time = np.load(data_folder + 'tfi_l%d_r%d_b%d_c%d_rank%d_%s.npz'%( 
                                                    l, r, b, c, rank, disentanglers_iter[0]))['time'][-1]
    
    for i, disentangler in enumerate(disentanglers):
        
        test = np.load(data_folder + 'tfi_l%d_r%d_b%d_c%d_rank%d_%s.npz'%( 
                                                    l, r, b, c, rank, disentanglers_iter[i]))

        if i == 0:
            plt.plot(range(l*c), test['sv_before'], c=color_cycle[3], 
                     label='Initial', lw=2, linestyle=l_cycle[5])
        
        # find wall time closest to Alt 
        min_time_index = np.argmin(np.abs(test['time'] - Alt_wall_time))
        
        # plot singular value spectrum corresponding to that time
        plt.plot(range(l*c), test['svs'][min_time_index, :], 
                 c=color_cycle[c_idxs[i]], label=disentangler, 
                 lw=2, linestyle=l_cycle[l_idxs[i]])

    plt.xlabel('$i$')
    plt.ylabel(r'$\sigma_i$')
    
    plt.axvline([rank], c=color_cycle[7], linestyle='--', lw=2, zorder=-2)
    plt.yscale('log')
    plt.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tick_params(axis='x', which='major', pad=12)
    plt.tick_params(axis='y', which='major', pad=8)

    plt.savefig(save_folder + 'fig9b.svg', bbox_inches='tight', dpi=1200, transparent=True)