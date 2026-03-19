import argparse
import numpy as np

def tab(data_folder, save_folder, num=1):
    
    eps = 1e-6
    
    if num == 1: # Table 1
        n = 12
        seed = 0
        iters = range(1, 7)
        filename = data_folder + 'binsearch_randn_seed%d_l%d_r%d_b%d_c%d_cg_max_iter4000_eps%.2e_'%(
                    seed, n, n, n, n, eps)
    elif num == 2: # Table 2 
        l = 8
        r = 8
        b = 60
        c = 12
        iters = range(1, 6)
        filename = data_folder + 'binsearch_tfi_l%d_r%d_b%d_c%d_alt_max_iter10000_eps%.2e_'%(
                    l, r, b, c, eps)
    else:
        raise ValueError("Table name is either 1 or 2")
        
    ks = []
    kls = []
    krs = []
    kopts = []
    c_ks = []

    for it in iters:

        test = np.load(filename + 'bin_iter%d.npz'%it)

        ks.append(test['k'])
        kls.append(test['kl'])
        krs.append(test['kr'])
        kopts.append(test['ku'])
        c_ks.append(test['cost'][-1])
            
    data = np.column_stack((iters, ks, kls, krs, kopts, c_ks))
    np.savetxt("tables/table%d.csv"%num, 
               data, 
               delimiter=",", 
               header = "iter, k, k_l, k_r, k_opt, c_k",
               fmt=["%d", "%d", "%d", "%d", "%d", "%.2e"]
              )
        
     
def main():

    parser = argparse.ArgumentParser(prog='Save csv files of manuscript tables')
    parser.add_argument('action', help='Table name, either 1 or 2')
    parser.add_argument('-data_folder', default='manuscript_data/', help='data folder', type=str)
    parser.add_argument('-save_folder', default='tables/', help='saved tables folder', type=str)

    args = parser.parse_args()
    
    num = int(args.action)            
  
    tab(args.data_folder, args.save_folder, num)     

    return

if __name__ == '__main__':
    main()
    