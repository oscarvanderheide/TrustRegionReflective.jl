module TrustRegionReflective

# Registered modules
using LinearAlgebra
using TimerOutputs
using CUDA # Added for CUDA.jl direct usage

"""
    struct TRFOptions{T<:Real}

Configuration options for the Trust Region Reflective (TRF) solver.

# Type Parameters
- `T<:Real`: The floating-point type for tolerances and ratios.

# Fields
- `min_ratio::T`: The minimum acceptable ratio of actual to predicted reduction. A step is a
ccepted if the ratio is above this value. Typically between 0 and 0.25.
- `max_iter_trf::Int`: The maximum number of iterations for the main Trust Region Reflective a
lgorithm.
- `max_iter_steihaug::Int`: The maximum number of iterations for the Steihaug-Toint conjugate 
gradient subproblem solver.
- `tol_steihaug::T`: The tolerance for the Steihaug-Toint subproblem. Iterations stop when t
he norm of the residual is below this tolerance.
- `init_scale_radius::T`: The initial scaling factor for the trust region radius, typically 
applied to the norm of the initial `x0`.
- `save_every_iter::Bool`: A boolean flag indicating whether to save intermediate results (e.g.
, `x`, `f`, `r`) at each iteration. This can be memory-intensive.
- `modfified_reduction_for_ratio::Bool`: A boolean flag indicating whether to use a modified
reduction for the ratio calculation that takes into account the coleman li scaling factors.
"""
@kwdef struct TRFOptions{T<:Real}
    min_ratio::T = T(0.1)
    max_iter_trf::Int = 20
    max_iter_steihaug::Int = 20
    tol_steihaug::T = T(1E-6)
    init_scale_radius::T = T(0.1)
    save_every_iter::Bool = false
    modfified_reduction_for_ratio::Bool = false
end

"""
    struct SolverState

Represents the state of the optimization algorithm at a given iteration.

# Fields
- `x`: History of parameter vectors (stored as columns in a matrix).
- `f`: History of objective function values (stored as a vector).
- `r`: History of residual vectors (stored as columns in a matrix).
- `t`: History of elapsed times (stored as a vector).
"""
mutable struct SolverState
    x
    f
    r
    t
end

include("utils.jl")
include("solver.jl")
include("steihaug.jl")

export trust_region_reflective, TRFOptions

end # module
