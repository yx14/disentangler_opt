from .plotting_utils import *

# wall time for Alt + RCG hybrid methods: isoTNS tensor
def fig10b(data_folder, save_folder):
    l = 8
    r = 8
    b = 60
    c = 12
    
    rank = 2*l
    
    # maximum iteration count for Alt (starting method)
    maxiters = [250, 500, 1000]

    disentanglers = ['Alt', 'RCG', 'Alt-iter250-RCG', 'Alt-iter500-RCG', 'Alt-iter1000-RCG']
    disentanglers_iter = ['alt_max_iter10000_', 'cg_max_iter4000_'] 
    disentanglers_iter += ['alt_max_iter%d_cg_max_iter4000_'%(maxiter) for maxiter in maxiters]

    # color and line cycle indices
    c_idxs = [0, 1, 5, 7, 6]
    l_idxs = [0, 1, 3, 4, 5]
    
    fig = plt.figure(figsize=(6,6))
    
    for i, disentangler in enumerate(disentanglers[:2]):

        test = np.load(data_folder + 'tfi_l%d_r%d_b%d_c%d_rank%d_%s.npz'%( 
                                                    l, r, b, c, rank, disentanglers_iter[i]))
        
        plt.plot(test['time'], test['cost'], c=color_cycle[c_idxs[i]], label=disentangler, 
                 lw=2, linestyle=l_cycle[l_idxs[i]])

    for i, disentangler in enumerate(disentanglers[2:]):
        test = np.load(data_folder + 'tfi_l%d_r%d_b%d_c%d_rank%d_%s.npz'%( 
                                                    l, r, b, c, rank, 'alt_max_iter%d_'%(maxiters[i])))
        
        time = list(test['time'][:maxiters[i]])
        cost = list(test['cost'][:maxiters[i]])
        
        test = np.load(data_folder + 'tfi_l%d_r%d_b%d_c%d_rank%d_%s.npz'%( 
                                                    l, r, b, c, rank, disentanglers_iter[i+2]))

        time += list(test['time'] + time[-1])
        cost += list(test['cost'])
        
        plt.plot(time, cost, c=color_cycle[c_idxs[i+2]], label=disentangler, 
                             lw=2, linestyle=l_cycle[l_idxs[i+2]])

        
    plt.yscale('log')
    plt.legend(frameon=False, bbox_to_anchor=[2, 1.1, 0, 0])

    plt.xlabel('Time (s)')
    plt.ylabel('$c_{%d}$'%(rank))

    plt.xlim((-2.5, 60))
    plt.ylim((3.206436630887529e-06, 2.2e-4))

    plt.tick_params(axis='x', which='major', pad=12)
    plt.tick_params(axis='y', which='major', pad=8)
    
    plt.savefig(save_folder + 'fig10b.svg', bbox_inches='tight', dpi=1200, transparent=True)
    
    # Save inset: representative gradient for one seed
    fig = plt.figure(figsize=(6,6))
    
    for i, disentangler in enumerate(disentanglers[1:2]):

        test = np.load(data_folder + 'tfi_l%d_r%d_b%d_c%d_rank%d_%s.npz'%( 
                                                    l, r, b, c, rank, disentanglers_iter[i+1]))

        plt.plot(test['time'], test['gnorm'], c=color_cycle[c_idxs[i+1]], 
                 label=disentangler, lw=2, linestyle=l_cycle[l_idxs[i+1]])

    for i, disentangler in enumerate(disentanglers[2:]):
        
        test = np.load(data_folder + 'tfi_l%d_r%d_b%d_c%d_rank%d_%s.npz'%( 
                                                    l, r, b, c, rank, 'alt_max_iter%d_'%(maxiters[i])))

        time = list(test['time'][:maxiters[i]])

        test = np.load(data_folder + 'tfi_l%d_r%d_b%d_c%d_rank%d_%s.npz'%( 
                                                    l, r, b, c, rank, disentanglers_iter[i+2]))

        time = list(test['time'] + time[-1])

        plt.plot(time, test['gnorm'], c=color_cycle[c_idxs[i+2]], 
                 label=disentangler, lw=2, linestyle=l_cycle[l_idxs[i+2]])


    plt.yscale('log')
    plt.legend(frameon=False, bbox_to_anchor=[2, 1.1, 0, 0])

    plt.xlabel('Time (s)')
    plt.ylabel(r'$|\mathrm{grad} \ c_{%d}|$'%(rank))

    plt.xlim([-2.5, 60])

    plt.xticks([0, 60])
    plt.yticks([1e-3, 1e-6])
    plt.tick_params(axis='x', which='major', pad=12)
    plt.tick_params(axis='y', which='major', pad=8)

    plt.savefig(save_folder + 'fig10b_inset.svg', dpi=1200, bbox_inches='tight', transparent=True)
