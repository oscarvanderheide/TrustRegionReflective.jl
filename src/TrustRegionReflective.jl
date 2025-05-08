module TrustRegionReflective

# Registered modules
using LinearAlgebra
using TimerOutputs
using CUDA # Added for CUDA.jl direct usage

"""
    struct TRFOptions{T<:Real, I<:Int, B<:Bool}

Configuration options for the Trust Region Reflective (TRF) solver.

# Type Parameters
- `T<:Real`: The floating-point type for tolerances and ratios.
- `I<:Int`: The integer type for iteration counts.
- `B<:Bool`: The boolean type for flags.

# Fields
- `min_ratio::T`: The minimum acceptable ratio of actual to predicted reduction. A step is a
ccepted if the ratio is above this value. Typically between 0 and 0.25.
- `max_iter_trf::I`: The maximum number of iterations for the main Trust Region Reflective a
lgorithm.
- `max_iter_steihaug::I`: The maximum number of iterations for the Steihaug-Toint conjugate 
gradient subproblem solver.
- `tol_steihaug::T`: The tolerance for the Steihaug-Toint subproblem. Iterations stop when t
he norm of the residual is below this tolerance.
- `init_scale_radius::T`: The initial scaling factor for the trust region radius, typically 
applied to the norm of the initial `x0`.
- `save_every_iter::B`: A boolean flag indicating whether to save intermediate results (e.g.
, `x`, `f`, `r`) at each iteration. This can be memory-intensive.
"""
struct TRFOptions{T<:Real,I<:Int,B<:Bool}
    min_ratio::T
    max_iter_trf::I
    max_iter_steihaug::I
    tol_steihaug::T
    init_scale_radius::T
    save_every_iter::B
end

"""
    struct SolverState

Represents the state of the optimization algorithm at a given iteration.

# Fields
- `x`: History of parameter vectors (stored as columns in a matrix).
- `f`: History of objective function values (stored as a 1xN matrix).
- `r`: History of residual vectors (stored as columns in a matrix).
- `t`: History of elapsed times (stored as a 1xN matrix).
"""
struct SolverState
    x::AbstractMatrix
    f::AbstractMatrix
    r::AbstractMatrix
    t::AbstractMatrix{Float64} # Assuming t is always Float64 based
end

include("utils.jl")
include("solver.jl")
include("steihaug.jl")

export trf, TRFOptions

end # module
