# General

This python repository accompanies the arXiv pre-print:2508.19409. The scripts in this repository are used to benchmark different optimization methods for the tensor disentangling problem: Given a four-dimensional tensor ${\cal X} \in \mathbb{R}^{l \times r \times b \times c}$, we would like to disentangle it along a particular bipartition of indices: $(lc, rb)$. To do this, we seek an orthogonal matrix (the "disentangler"), $Q \in O(lr)$, and contract it with ${\cal X}$, resulting in ${\cal X'}$. By optimizing $Q$, the goal is to find a low-rank approximation of the  unfolding of ${\cal X'}$ into an $lc\times rb$-dimensional matrix. 

# Installation 

The scripts are in python, and have been tested on [TODO: mac op system] and [TODO: perlmutter op system]. A environment manager such as conda may be used to install dependencies. The typical install time is [TODO: estimate minutes.]  

## Dependencies

The scripts were tested using the following versions:
- python 3.13.0
- numpy 2.1.3
- scipy 1.14.1
- pymanopt 2.2.1
- matplotlib X (for running scripts in the test subdirectory)

We make two changes to the pymanopt package. First, in order to enable logging for the trust-regions newton optimizer, we add the following to the while loop (line 174) in pymanopt/optimizers/trust_regions.py:

self._add_log_entry(
                iteration=iteration,
                point=x,
                cost=fx,
                gradient_norm=norm_grad,
            )

Second, the default svd is lapack's gesdd, which can fail for various objectives, resulting in an 'svd failed to converge error.' In all svd calls, both in pymanopt and our objective functions, we use 

scp.linalg.svd(Y, lapack_driver='gesvd')

This includes modifying line 76 of the polar retraction in pymanopt/manifolds/group.py. See: https://stackoverflow.com/questions/63761366/numpy-linalg-linalgerror-svd-did-not-converge-in-linear-least-squares-on-first

# List of python files

- altdisentangler.py : implementation of the alternating disentangler
- benchmark_driver.py : disentangling benchmarking with different objectives, methods, and tensors
- binsearch_driver.py : binary search implementation given a truncation error tolerance and tensor, which returns the optimal rank after disentangling
- disentangler_utils.py : implementation of different objective functions and their Riemannian gradients. For the truncation error objective, the Riemannian Hessian is also implemented

## Tests 
- check_basic_utils.py : check correctness of tensor composition map and its inverse 
- check_gradient_and_hessian.py : check correctness of Riemannian gradients and hessians using pymanopt utility functions

To run tests: 
```
python check_basic_utils.py
python check_gradient_and_hessian.py
```

# Examples

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

# Data 

Several example tensors are stored in the isotns_data directory. These tensors are intermediate outputs from an isometric tensor network state (isoTNS) ansatz.

## Tensor ordering convention

In the scripts, the convention we use in the 4-tensor leg ordering is $(l, r, b, c)$, where the disentangler is applied to the (grouped) $lr$ indices, and the SVD after permuting and reshaping is applied to an unfolding of dimensions $(lc, rb)$.

The isoTNS tensors are initially ordered as $(c, l, r, b)$, and their indices permuted after loading (see ```benchmark_driver.py``` for an example).

## Data background 
The isoTNS ansatz approximates the ground state of the two-dimensional Transverse-Field Ising model (2D TFIM) on an 8x8 square lattice with open boundaries, which has Hamiltonian

$
H = -J \sum_{\langle i, j \rangle} Z_i Z_j - g \sum_i X_i.
$

The approximation is computed via a 2D generalization of time-evolving block decimation, using the following parameters: imaginary-time step size $d\tau = 0.025$, bond dimension $\chi=12$, and orthogonality hypersurface bond dimension $\eta=20$. For more information on this tensor network ansatz, please refer to:

- Zaletel, M.P., Pollmann, F.: Isometric Tensor Network States in Two Dimen-
sions. Physical Review Letters 124(3), 037201 (2020) https://doi.org/10.1103/PhysRevLett.124.037201

- Lin, S.-H., Zaletel, M.P., Pollmann, F.: Efficient simulation of dynamics in
two-dimensional quantum spin systems with isometric tensor networks. Phys-
ical Review B 106(24), 245102 (2022) https://doi.org/10.1103/PhysRevB.106.245102

# Tensor conventions




