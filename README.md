# General

This python repository accompanies the arXiv pre-print:2508.19409. The scripts in this repository are used to benchmark different optimization methods for the tensor disentangling problem: Given a four-dimensional tensor ${\cal X} \in \mathbb{R}^{l \times r \times b \times c}$, we would like to disentangle it along a particular bipartition of indices: $(lc, rb)$. To do this, we seek an orthogonal matrix (the "disentangler"), $Q \in O(lr)$, and contract it with ${\cal X}$, resulting in ${\cal X'}$. By optimizing $Q$, the goal is to find a low-rank approximation of the  unfolding of ${\cal X'}$ into an $lc\times rb$-dimensional matrix. 

# Installation 

The scripts are in python, and have been tested on macOS and Linux systems. 

The conda environment manager is used to install dependencies, through environment.yml or conda-lock.yml. A fork of the pymanopt package is installed, which enables logging for the trust-regions newton optimizer, and modifies numpy SVD calls to use the 'GESVD' LAPACK driver for numerical stability.

  The environment.yml file generates an environment called "disentangle": 
```
  conda env create -f environment.yml
```
  The conda-lock.yml is a lockfile that uses the conda-lock package (https://github.com/conda/conda-lock). To generate an environment called "disentangle" using conda-lock:
```    
  conda create -n disentangle conda-lock
  conda-lock install -n disentangle
```

# List of main python files

- altdisentangler.py : implementation of the alternating disentangler
- benchmark_driver.py : disentangling benchmarking with different objectives, methods, and tensors
- binsearch_driver.py : binary search implementation given a truncation error tolerance and tensor, which returns the optimal rank after disentangling
- disentangler_utils.py : implementation of different objective functions and their Riemannian gradients. For the truncation error objective, the Riemannian Hessian is also implemented
- make_figure.py : reproduce plots in manuscript
- make_table.py : reproduce tables in manuscript 

## Tests 

- check_basic_utils.py : check correctness of tensor composition map and its inverse 
- check_gradient_and_hessian.py : check correctness of Riemannian gradients and hessians using pymanopt utility functions

To run tests: 
```
python check_basic_utils.py
python check_gradient_and_hessian.py
```

# Small Examples

Results are saved in a "results" subdirectory (create before running the following examples). 

## Benchmarking

To check the full list of options in the benchmark driver:
```
python benchmark_driver.py -h
```

To run the CG disentangler for 100 iterations on a $6 \times 6 \times 6 \times 6$ random tensor, using the Renyi-1/2 entropy objective function (estimated time < 1 sec):
```
python benchmark_driver.py -type randn -dim 6 -obj renyihalf -cg_max_iter 100
```

To run the alternating disentangler for 100 iterations on an isoTNS tensor, using the target rank objective function 
and a target rank of 16 (estimated time ~ 7 sec): 
```
python benchmark_driver.py -type randn -dim 8 8 60 12 -obj rank -rank 16 -cg_max_iter 100
```

## Binary search

To check the full list of options in the binary search driver:
```
python binsearch_driver.py -h
```

The binary search driver relies on a Renyi-1/2 result to set the upper bound on the target rank, which is produced by the benchmark driver. Assuming that the Renyi-1/2 benchmarking example above has already been run on a random tensor, we can run binary search (estimated time ~ 2 sec):
```
python binsearch_driver.py -type randn -dim 6 -cg_max_iter 100 -cg_max_iter_init 100
```

# Reproducing manuscript figures and tables

## Generating manuscript data

We list the python commands used to generate results for Figures 7-12 and Tables 1-2. The default execution reproduces the main qualitative results without additional parameter tuning. A full reproduction run exceeds 10 minutes on a standard laptop: for default execution, a given run of benchmark_driver.py takes up to 4 minutes for the "alt" and "cg" disentangler methods, and up to 1 hours for the "newton" method. A rough wall time estimate of the full reproduction run is ~1.5 days.

We use fixed seeds for random tensor benchmarks. Outside of generating the random tensors, the code does not have stochastic components.    

Optimization results are saved in a "results" subdirectory of disentangler_opt:
```
mkdir results
``` 

Figs 7a, 8a, and 9a:
```
python benchmark_driver.py -type randn -dim 12 -obj rank -rank 44 -disentangler alt --save_svs -data_folder results/
python benchmark_driver.py -type randn -dim 12 -obj rank -rank 44 -disentangler cg --save_svs -data_folder results/
python benchmark_driver.py -type randn -dim 12 -obj rank -rank 44 -disentangler newton -reg 0.0 --save_svs -data_folder results/
python benchmark_driver.py -type randn -dim 12 -obj rank -rank 44 -disentangler newton --save_svs -data_folder results/
```

Figs 7b, 8b, and 9b:
```
python benchmark_driver.py -type tfi -dim 8 8 60 12 -obj rank -rank 8 -disentangler alt --save_svs -data_folder results/
python benchmark_driver.py -type tfi -dim 8 8 60 12 -obj rank -rank 8 -disentangler cg --save_svs -data_folder results/
python benchmark_driver.py -type tfi -dim 8 8 60 12 -obj rank -rank 8 -disentangler newton -reg 0.0 --save_svs -data_folder results/
python benchmark_driver.py -type tfi -dim 8 8 60 12 -obj rank -rank 8 -disentangler newton --save_svs -data_folder results/
```

Fig 10a:
```
for i in {0..4}; do
    python benchmark_driver.py -type randn -dim 12 -seed $i -obj rank -rank 44 -disentangler alt -data_folder results/
    python benchmark_driver.py -type randn -dim 12 -seed $i -obj rank -rank 44 -disentangler cg -data_folder results/
    python benchmark_driver.py -type randn -dim 12 -seed $i -obj rank -rank 44 -disentangler altcg -alt_max_iter 5 -data_folder results/
    python benchmark_driver.py -type randn -dim 12 -seed $i -obj rank -rank 44 -disentangler altcg -alt_max_iter 10 -data_folder results/
    python benchmark_driver.py -type randn -dim 12 -seed $i -obj rank -rank 44 -disentangler altcg -alt_max_iter 20 -data_folder results/
done
```

Fig 10b:
```
python benchmark_driver.py -type tfi -dim 8 8 60 12 -obj rank -rank 16 -disentangler alt -data_folder results/
python benchmark_driver.py -type tfi -dim 8 8 60 12 -obj rank -rank 16 -disentangler cg -data_folder results/
python benchmark_driver.py -type tfi -dim 8 8 60 12 -obj rank -rank 16 -disentangler altcg -alt_max_iter 250 -data_folder results/
python benchmark_driver.py -type tfi -dim 8 8 60 12 -obj rank -rank 16 -disentangler altcg -alt_max_iter 500 -data_folder results/
python benchmark_driver.py -type tfi -dim 8 8 60 12 -obj rank -rank 16 -disentangler altcg -alt_max_iter 1000 -data_folder results/
```

Fig 10c:
```
for i in {0..4}; do
    python benchmark_driver.py -type randn -dim 12 -seed $i -obj rank -rank 44 -disentangler alt -data_folder results/
    python benchmark_driver.py -type randn -dim 12 -seed $i -obj rank -rank 44 -disentangler newton -reg 0.0 -data_folder results/
    python benchmark_driver.py -type randn -dim 12 -seed $i -obj rank -rank 44 -disentangler altnewton -alt_max_iter 5 -reg 0.0 -data_folder results/
    python benchmark_driver.py -type randn -dim 12 -seed $i -obj rank -rank 44 -disentangler altnewton -alt_max_iter 10 -reg 0.0 -data_folder results/
    python benchmark_driver.py -type randn -dim 12 -seed $i -obj rank -rank 44 -disentangler altnewton -alt_max_iter 20 -reg 0.0 -data_folder results/
done
```

Fig 10d:
```
python benchmark_driver.py -type tfi -dim 8 8 60 12 -obj rank -rank 16 -disentangler alt -data_folder results/
python benchmark_driver.py -type tfi -dim 8 8 60 12 -obj rank -rank 16 -disentangler newton -reg 0.0 -data_folder results/
python benchmark_driver.py -type tfi -dim 8 8 60 12 -obj rank -rank 16 -disentangler altnewton -alt_max_iter 250 -reg 0.0 -data_folder results/
python benchmark_driver.py -type tfi -dim 8 8 60 12 -obj rank -rank 16 -disentangler altnewton -alt_max_iter 500 -reg 0.0 -data_folder results/
python benchmark_driver.py -type tfi -dim 8 8 60 12 -obj rank -rank 16 -disentangler altnewton -alt_max_iter 1000 -reg 0.0 -data_folder results/
```

Figs 11a and 12a:
```
python benchmark_driver.py -type randn -dim 12 -obj renyihalf --save_svs -data_folder results/
python benchmark_driver.py -type randn -dim 12 -obj vn --save_svs -data_folder results/
python benchmark_driver.py -type randn -dim 12 -obj rank -rank 40 -disentangler cg --save_svs -data_folder results/
python benchmark_driver.py -type randn -dim 12 -obj rank -rank 44 -disentangler cg --save_svs -data_folder results/
python benchmark_driver.py -type randn -dim 12 -obj rank -rank 48 -disentangler cg --save_svs -data_folder results/
```

Figs 11b and 12b:
```
python benchmark_driver.py -type tfi -dim 8 8 60 12 -obj renyihalf --save_svs -data_folder results/
python benchmark_driver.py -type tfi -dim 8 8 60 12 -obj vn --save_svs -data_folder results/
python benchmark_driver.py -type tfi -dim 8 8 60 12 -obj rank -rank 8 -disentangler alt -data_folder results/
python benchmark_driver.py -type tfi -dim 8 8 60 12 -obj rank -rank 16 -disentangler alt -data_folder results/
python benchmark_driver.py -type tfi -dim 8 8 60 12 -obj rank -rank 24 -disentangler alt -data_folder results/
```

Table 1:
```
python benchmark_driver.py -type randn -dim 12 -obj renyihalf -data_folder results/
python binsearch_driver.py -type randn -dim 12 -disentangler cg -data_folder results/
```

Table 2:
```
python benchmark_driver.py -type tfi -dim 8 8 60 12 -obj renyihalf -data_folder results/
python binsearch_driver.py -type tfi -dim 8 8 60 12 -disentangler alt -data_folder results/
```

## Figure plotting

By default, we plot the data in manuscript_data, and save the figures as an SVG file in figures. The "make_figure.py" file generates all sub-figures in Figures 7-12.

To generate and save a figure, the figure number (e.g. 7a) is specified as:
```
python make_figure.py 7a -data_folder manuscript_data/ -save_folder figures/
```

For Figure 5 (which examines the cost landscape for a random tensor using a fixed seed, and does not rely on manuscript_data) and save an SVG file in figures:
```
python make_figure.py 5 -save_folder figures/
```

## Table generation

By default, we create tables using data in manuscript_data, and save the table as a CSV file in tables. The "make_table.py" file generates Tables 1 and 2.

To generate and save a figure, the table number (e.g. 1 or 2) is specified as:
```
python make_table.py 1 -data_folder manuscript_data/ -save_folder tables/
```

# Data 

Several example tensors are stored in the isotns_data directory. These tensors are intermediate outputs from an isometric tensor network state (isoTNS) ansatz.

For random tensor benchmarking, we use seeds listed in (TODO).

## Tensor ordering convention

In the scripts, the convention we use in the 4-tensor leg ordering is $(l, r, b, c)$, where the disentangler is applied to the (grouped) $lr$ indices, and the SVD after permuting and reshaping is applied to an unfolding of dimensions $(lc, rb)$.

The isoTNS tensors are initially ordered as $(c, l, r, b)$, and their indices permuted after loading (see ```benchmark_driver.py``` for an example).

## Data background 

The isoTNS ansatz approximates the ground state of the two-dimensional Transverse-Field Ising model (2D TFIM) on an 8x8 square lattice with open boundaries, which has Hamiltonian

$
H = -J \sum_{\langle i, j \rangle} Z_i Z_j - g \sum_i X_i.
$

The approximation is computed via a 2D generalization of time-evolving block decimation, using the following parameters: imaginary-time step size $d\tau = 0.025$, bond dimension $\chi=12$, and orthogonality hypersurface bond dimension $\eta=20$. For more information on this tensor network ansatz, please refer to:

- Zaletel, M.P., Pollmann, F.: Isometric Tensor Network States in Two Dimensions. Physical Review Letters 124(3), 037201 (2020) https://doi.org/10.1103/PhysRevLett.124.037201

- Lin, S.-H., Zaletel, M.P., Pollmann, F.: Efficient simulation of dynamics in
two-dimensional quantum spin systems with isometric tensor networks. Physical Review B 106(24), 245102 (2022) https://doi.org/10.1103/PhysRevB.106.245102





