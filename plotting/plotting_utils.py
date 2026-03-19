import colorsys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import interp1d

# style file
plt.style.use('norm.mplstyle')

# From https://arxiv.org/pdf/2107.02270
color_cycle = [(63, 144, 218), (255, 169, 14),
                  (189, 31, 1), (148, 164, 162),
                  (131, 45, 182), (169, 107, 89),
                  (231, 99, 0), (185, 172, 112),
                  (113, 117, 129), (146, 218, 221)]

color_cycle = [np.array(i)/255 for i in color_cycle]

# linestyle cycle
l_cycle = ['-', (0, (5, 1)),'dashdot', (5, (10, 3)), (0, (1, 1)), (0, (5, 1, 1, 1, 1, 1)), ]

# font sizes
TICK_SIZE = 24
LABEL_SIZE = 30

plt.rc('axes', titlesize=LABEL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=LABEL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=TICK_SIZE)    # legend fontsize

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# create a purple cmap for Figs 11 and 12
def scale_lightness(rgb, scale_l):
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s = s)

# dec - float between 0 and 1
# from https://arxiv.org/pdf/2107.02270, Table I, 6 colors
# scaling: https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
def cmap_custom_purple(dec):
    purple_custom = np.array((131, 45, 182))/255
    scale_fac = 2 + (0.6 - 2)*dec
    return scale_lightness(purple_custom[:3], scale_fac)

# helper function for Fig 11
# input: s - array of singular values
# return: trunc_err, where trunc_err[k] is the truncation error for rank k  
def get_trunc_err(s):
    trunc_err = np.zeros(len(s))
    for i in range(len(s)):
        trunc_err[i] = np.sum(s[i:]**2)
        
    return trunc_err