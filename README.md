# TrustRegionReflective.jl

A Julia implementation of the Trust Region Reflective algorithm for solving bound-constrained nonlinear least-squares problems. 

This code used to be part of a larger codebase. It is now a separate package but has not been tested outside of the scope of the larger codebase. Use at your own risk.

## Features
- Implements the Trust Region Reflective algorithm.
- Supports bound constraints on variables.
- All the functions are - I think - type stable - which allows the solver to be used with, for example, `CuArrays` from the `CUDA.jl`.
- Ideally this solver becomes part of, say, [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl). I don't have time myself to make this happen though.

## Acknowledgments
This package is inspired by the following sources:
- [Nikolay Mayorov's blog post](https://nmayorov.wordpress.com/2015/06/19/trust-region-reflective-algorithm/) on the Trust Region Reflective algorithm.
- The [SciPy library](https://github.com/scipy/scipy), specifically its implementation of the Trust Region Reflective algorithm.

## License
This package is licensed under the BSD 3-Clause License. Parts of the code are derived from the SciPy library, which is also licensed under the BSD 3-Clause License.
