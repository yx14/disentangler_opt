from plotting_utils import *

# wall time for Alt + RCG hybrid methods: random tensor
def fig10a(data_folder, save_folder):
    n = 12
    
    rank = 44
    
    # maximum iteration count for Alt (starting method)
    maxiters = [5, 10, 20]

    # average over five random seeds 
    seeds = [0, 1, 2, 3, 4]

    disentanglers = ['Alt', 'RCG', 'Alt-iter5-RCG', 'Alt-iter10-RCG', 'Alt-iter20-RCG']
    disentanglers_iter = ['alt_max_iter10000_', 'cg_max_iter4000_'] 
    disentanglers_iter += ['alt_max_iter%d_cg_max_iter4000_'%(maxiter) for maxiter in maxiters]

    # color and line cycle indices
    c_idxs = [0, 1, 5, 7, 6]
    l_idxs = [0, 1, 3, 4, 5]
    
    fig = plt.figure(figsize=(6,6))
    
    for i, disentangler in enumerate(disentanglers):

        time_all = []
        cost_all = []

        max_times = []
        
        for seed in seeds:

            if i >= 2:
                test = np.load(data_folder + 'randn_seed%d_l%d_r%d_b%d_c%d_rank%d_%s.npz'%( 
                                                            seed, n, n, n, n, rank, 'alt_max_iter%d_'%(maxiters[i-2])))

                # time, cost for Alt
                time = list(test['time'][:maxiters[i-2]])
                cost = list(test['cost'][:maxiters[i-2]])
                
                # append time, cost for RCG after running Alt
                test = np.load(data_folder + 'randn_seed%d_l%d_r%d_b%d_c%d_rank%d_%s.npz'%( 
                                                            seed, n, n, n, n, rank, disentanglers_iter[i]))

                time += list(test['time'][1:] + time[-1])
                cost += list(test['cost'][1:])

            else:
                test = np.load(data_folder + 'randn_seed%d_l%d_r%d_b%d_c%d_rank%d_%s.npz'%( 
                                                        seed, n, n, n, n, rank, disentanglers_iter[i]))
                time = test['time']
                cost = test['cost']

            f = interp1d(time, cost, kind='cubic')
            time_sample = np.linspace(0, int(max(time)), int(max(time))*2+1)

            # track sampled times and costs
            time_all.append(time_sample)
            cost_all.append(f(time_sample))

            max_times.append(int(max(time)))
           
        max_times.sort()

        # choose time interval containing at least 3 interpolated sample points
        num_times = max_times[2]*2+1
        time_sample = np.linspace(0, max_times[2], num_times)
        
        # take average and stdev 
        cost_mean = []
        for time_idx in range(num_times):
            cost_subset = [cost_all[j][time_idx] for j in range(len(seeds)) if len(time_all[j]) >= num_times]
            cost_mean.append(np.mean(cost_subset))

        cost_mean = np.array(cost_mean)
        
        plt.plot(time_sample, cost_mean, c=color_cycle[c_idxs[i]], label=disentanglers[i], 
                 lw=2, linestyle=l_cycle[l_idxs[i]])

    plt.yscale('log')
    plt.legend(frameon=False, bbox_to_anchor=[2, 1.1, 0, 0])

    plt.xlabel('Time (s)')
    plt.ylabel('$c_{%d}$'%(rank))

    plt.xlim([-7.800000000000001, 163.8])
    plt.ylim([1.6066799963193843e-10, 1.0])

    plt.yticks([1e0, 1e-2, 1e-4, 1e-6, 1e-8])
    plt.tick_params(axis='x', which='major', pad=12)
    plt.tick_params(axis='y', which='major', pad=8)
    
    plt.savefig(save_folder + 'fig10a.svg', bbox_inches='tight', dpi=1200, transparent=True)
    
    # Save inset: representative gradient for one seed
    fig = plt.figure(figsize=(6,6))
    
    seed = 0
    for i, disentangler in enumerate(disentanglers[1:2]):

        test = np.load(data_folder + 'randn_seed%d_l%d_r%d_b%d_c%d_rank%d_%s.npz'%( 
                                                    seed, n, n, n, n, rank, disentanglers_iter[i+1]))

        plt.plot(test['time'], test['gnorm'], c=color_cycle[c_idxs[i+1]], 
                 label=disentangler, lw=2, linestyle=l_cycle[l_idxs[i+1]])

    for i, disentangler in enumerate(disentanglers[2:]):
        
        test = np.load(data_folder + 'randn_seed%d_l%d_r%d_b%d_c%d_rank%d_%s.npz'%( 
                                                    seed, n, n, n, n, rank, 'alt_max_iter%d_'%(maxiters[i])))

        time = list(test['time'][:maxiters[i]])

        test = np.load(data_folder + 'randn_seed%d_l%d_r%d_b%d_c%d_rank%d_%s.npz'%( 
                                                    seed, n, n, n, n, rank, disentanglers_iter[i+2]))

        time = list(test['time'] + time[-1])

        plt.plot(time, test['gnorm'], c=color_cycle[c_idxs[i+2]], 
                 label=disentangler, lw=2, linestyle=l_cycle[l_idxs[i+2]])


    plt.yscale('log')
    plt.legend(frameon=False, bbox_to_anchor=[2, 1.1, 0, 0])

    plt.xlabel('Time (s)')
    plt.ylabel(r'$|\mathrm{grad} \ c_{%d}|$'%(rank))

    plt.xlim([-7.800000000000001, 163.8])
    plt.ylim([1e-8, 1e-1])

    plt.xticks([0, 150])
    plt.yticks([1e-1, 1e-8])
    plt.tick_params(axis='x', which='major', pad=12)
    plt.tick_params(axis='y', which='major', pad=8)

    plt.savefig(save_folder + 'fig10a_inset.svg', dpi=1200, bbox_inches='tight', transparent=True)
