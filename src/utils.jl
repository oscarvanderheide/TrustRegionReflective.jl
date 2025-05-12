"""
    all_within_bounds(x, LB, UB)

Check if all elements of `x` are within the corresponding lower bounds `LB` and upper bounds `UB`.

# Arguments
- `x::AbstractVector{T}`: The vector to be checked.
- `LB::AbstractVector{T}`: The lower bounds.
- `UB::AbstractVector{T}`: The upper bounds.

# Returns
- `true` if all elements of `x` are within the bounds, `false` otherwise.
"""
function all_within_bounds(x::T, LB::T, UB::T) where {T<:AbstractVector{<:Real}}

    if length(x) != length(LB) || length(x) != length(UB)
        error("    all_within_bounds: Lengths of x, LB, and UB do not match")
    end

    return all((x .>= LB) .& (x .<= UB))
end

"""
    stepsize_to_bound_feasible_region(x, s, LB, UB)

The function computes a the smallest positive scalar stepsize, such that `x + stepsize * s` is on the bound.
- `x::AbstractVector`: The current point.
- `s::AbstractVector`: The search direction.
- `LB::AbstractVector`: The lower bounds for each element of `x`.
- `UB::AbstractVector`: The upper bounds for each element of `x`.

Returns a tuple `(stepsize, boundary_hit)`:
- `stepsize`: The minimum stepsize to reach the feasible region.
- `boundary_hit`: An array indicating which boundaries are hit (0: bound not hit, -1: lower bound hit, 1: upper bound hit).
"""
function stepsize_to_bound_feasible_region(x::T, s::T, LB::T, UB::T) where {T<:AbstractVector{<:Real}}

    # Check that initially x is within the bounds
    if !all_within_bounds(x, LB, UB)
        error("    Somehow x is not within the bounds")
    end

    # Identify non-zero elements in s
    non_zero = s .!= 0
    non_zero = parent(non_zero)

    # Initialize steps with Inf
    steps = similar(x)
    steps .= Inf

    # Compute steps for non-zero elements
    steps[non_zero] = max.((LB[non_zero] - x[non_zero]) ./ s[non_zero], (UB[non_zero] - x[non_zero]) ./ s[non_zero])

    # Compute the minimum stepsize
    stepsize = minimum(steps)

    # Check that stepsize is positive
    if stepsize < 0
        error("    stepsize_to_bound_feasible_region: Negative stepsize encountered")
        return
    end

    # Identify which boundaries are hit
    boundary_hit = (steps .== stepsize)
    boundary_hit = parent(boundary_hit)

    return stepsize, boundary_hit
end

"""
    positive_stepsize_to_bound_trust_region(x::V{T}, p::V{T}, trust_radius::T)

Compute the positive stepsize `τ` such that the norm of `x + τ * p` equals the `trust_radius`. This is done by solving the quadratic equation `norm(x + τ * p)^2 = trust_radius^2` for `τ`.

# Arguments
- `x::AbstractVector{T}`: The current point.
- `p::AbstractVector{T}`: The search direction.
- `trust_radius::T`: The trust radius.

# Returns
- `τ::T`: The positive stepsize to reach the trust region boundary.
"""
function positive_stepsize_to_bound_trust_region(x::V, p::V, trust_radius::T) where {V<:AbstractVector{T}} where {T}
    if trust_radius <= 0
        error("    positive_stepsize_to_bound_trust_region: Trust radius must be positive")
    end

    # Coefficients for the quadratic equation
    a = norm(p)^2
    b = 2 * dot(x, p)
    c = norm(x)^2 - trust_radius^2

    # Solve the quadratic equation for τ using the quadratic formula
    τ = (-b + sqrt(b^2 - 4 * a * c)) / (2 * a)

    if τ < 0
        error("    positive_stepsize_to_bound_trust_region: Negative stepsize encountered")
    end

    return τ
end

"""
    function adjust_trust_radius(
        current_ratio::T,
        step::AbstractVector{T},
        Δ::T,
        min_ratio::T
    ) where {T<:Real}

Adjust the trust region radius based on the current ratio of actual reduction to predicted reduction in the cost.

The function reduces the radius if the current ratio is "poor" in the sense that it is  less than `min_ratio`. It increases the radius if the current ratio is "good" in the sense that it is greater than 0.5 and the step is very close to the trust region boundary. In other situations, it keeps the radius as is.

# Arguments
- `current_ratio`: The current ratio of actual to predicted reduction.
- `step`: The current step vector.
- `Δ`: The current trust region radius.
- `min_ratio`: The minimum acceptable ratio.

# Returns
- The adjusted trust region radius.
"""
function adjust_trust_radius(
    current_ratio::T,
    step::AbstractVector{T},
    Δ::T,
    min_ratio
) where {T<:Real}

    # @info "Norm of step: $(norm(step)), Trust Radius: $(Δ)"

    norm_step = round(norm(step), digits=2)

    if norm_step > T(1.01) * round(Δ, digits=2)
        error("    adjust_trust_radius: Norm of step ($norm_step) is greater than trust radius ($(round(Δ, digits=2)))")
    end

    if current_ratio < min_ratio # The trust region is too large. Reduce the radius.
        # @info "Trust Radius too large"
        Δ = Δ / 4

    elseif (current_ratio > 1 / 2) && (norm(step) > (T(0.95) * Δ))

        # The trust region seems to be too small. Increase the radius.
        # @info "Trust Radius too small"
        Δ = 2 * Δ

    else # The trust region seems to be fine. Keep the radius the same.
        # @info "Trust Radius just fine"
    end

    return Δ
end

"""
    function coleman_li_scaling_factors(x, g, LB, UB)

First-Order optimality condition in the case of box constraints is:

D^2 .* g(x) = v(x) .* g(x) = 0,

with g the gradient of f and v containing scaling factors based on distances to the boundaries to which -g points. The scaling factors in v make it more difficult to get onto the boundary during the reconstruction.

# Arguments
- `x::AbstractVector{<:Real}`: The current parameter estimates.
- `g::AbstractVector{<:Real}`: The gradient of the objective function w.r.t. x.
- `LB::AbstractVector{<:Real}`: Lower bounds for the parameters.
- `UB::AbstractVector{<:Real}`: Upper bounds for the parameters.

# Returns
- `v::AbstractVector{<:Real}`: The scaling factors calculated based on the distance to the boundaries.
- `dv::AbstractVector{<:Real}`: The partial derivatives of `v`

# Example
"""
function coleman_li_scaling_factors(x::T, g::T, LB::T, UB::T) where {T<:AbstractVector{<:Real}}

    # Check that initially x is within the bounds
    if !all_within_bounds(x, LB, UB)
        error("    Somehow x is not within the bounds")
    end

    # Initialize the scaling factors to 1
    v = similar(x)
    v .= 1

    # Mask that selects parameters for which -g is positive and which have upper bounds
    mask⁺ = (-g .>= 0) .& (UB .< Inf)
    # g and UB are ComponentVectors, having mask as ComponentVector gives issues for some reason so we convert it to a regular vector with `parent`
    mask⁺ = parent(mask⁺)
    # Set scaling factors to the distance to the upper bound for these parameters
    v[mask⁺] = UB[mask⁺] - x[mask⁺]

    # Mask that selects parameters for which -g is negative and which have lower bounds
    mask⁻ = (-g .<= 0) .& (LB .> -Inf)
    # g and LB are ComponentVectors, having mask as ComponentVector gives issues for some reason so we convert it to a regular vector with `parent`
    mask⁻ = parent(mask⁻)
    # Set scaling factors to the distance to the lower bound for these parameters
    v[mask⁻] = x[mask⁻] - LB[mask⁻]

    # Check if any of the distances are negative
    if any(v .< 0)
        error("Somehow v has negative values")
    end

    # Compute the partial derivatives of the distances
    dv = similar(v)
    dv .= 0
    dv[mask⁺] .= -1
    dv[mask⁻] .= 1

    return v, dv
end

"""
    evaluate_quadratic(H, g, s) where {T<:Real}

Evaluate the quadratic approximation to the objective `f`. The rational (1//2) is used to avoid unwanted conversion of the result to `Float64`.

# Arguments
- `H::Function`: The Hessian matrix.
- `g::AbstractVector{T}`: The gradient vector.
- `s::AbstractVector{T}`: The input vector.

# Returns
- `value::Real`: The value of the quadratic approximation to `f(s)`.
"""
function evaluate_quadratic(H, g::T, s::T) where {T<:AbstractVector{<:Real}}
    value = g' * s + (1 // 2) * (s' * H(s))
    return value
end

"""
    build_quadratic_1d(
        H,
        g::AbstractVector{T},
        s::AbstractVector{T},
        s0::AbstractVector{T}
    ) where {T<:Real}

Parameterize a multivariate quadratic function along a line. The rational (1//2) is used to avoid unwanted conversion of the result to `Float64`.

# Arguments
- `H::Function`: The Hessian matrix.
- `g::AbstractVector{<:Real}`: The gradient vector.
- `s::AbstractVector{<:Real}`: The direction vector.
- `s0::AbstractVector{<:Real}`: The starting point vector.

# Returns
- `a::Real`: The coefficient of the quadratic term.
- `b::Real`: The coefficient of the linear term.
- `c::Real`: The constant term.
"""
function build_quadratic_1d(H, g::T, s::T, s0::T) where {T<:AbstractVector{<:Real}}
    # Parameterize a multivariate quadratic function along a line.
    # f(t) = (1/2) * (s0 + s*t)' * H * (s0 + s*t) + g' * (s0 + s*t)

    a = (1 // 2) * s' * H(s)
    b = g' * s + (1 // 2) * s' * H(s0) + (1 // 2) * s' * H(s0)
    c = g' * s0 + (1 // 2) * s0' * H(s0)

    return a, b, c

end

"""
    minimize_quadratic_1d(a::T, b::T, lb::T, ub::T, c::T) where T <: Real

Minimize a 1-dimensional quadratic function subject to bounds.

The function minimizes the quadratic function `t -> at² + bt + c` subject to the bounds `lb <= t <= ub`. The minimum is either found on the boundary, or at the point where the gradient vanishes (`-0.5b/a`).

# Arguments
- `a::Real`: The coefficient of `t²`.
- `b::Real`: The coefficient of `t`.
- `lb::Real`: The lower bound for `t`.
- `ub::Real`: The upper bound for `t`.
- `c::Real`: The constant term.

# Returns
- `argument::real`: The value of `t` that minimizes the function.
- `minval::Real`: The minimum value of the function.
"""
function minimize_quadratic_1d(a::T, b::T, lb::T, ub::T, c::T) where {T<:Real}

    # Initialize the possible values of t
    t = [lb, ub]

    # If a is not zero, check if the extremum of the function is within the bounds
    if a != 0
        extremum = -(1 // 2) * b / a
        if lb < extremum < ub
            push!(t, extremum)
        end
    end

    # Compute the function values at the possible values of t
    y = @. a * t^2 + b * t + c

    # Find the minimum function value and the corresponding value of t
    minval = minimum(y)
    argument = t[findfirst(y .== minval)]

    return argument, minval
end

"""
    stepsizes_to_bound_trust_region(x::AbstractVector{T}, s::AbstractVector{T}, trust_radius::T) where {T<:Real}

Find the intersection of the line t-> x + t*s with the boundary of the trust region.

This function solves the quadratic equation with respect to t:
||(x + s*t)||^2 = trust_radius^2.
Returns [t-, t+], the negative and positive roots.

# Arguments
- `x::AbstractVector{<:Real}`: The current point.
- `s::AbstractVector{<:Real}`: The direction vector.
- `trust_radius::Real`: The trust region radius.

# Returns
- `t⁻::Real`: The negative root of the quadratic equation.
- `t⁺::Real`: The positive root of the quadratic equation.

# Raises
- `ValueError` if `s` is zero or `x` is not within the trust region.
"""
function stepsizes_to_bound_trust_region(x::V, s::V, trust_radius::T) where {V<:AbstractVector{T}} where {T<:Real}


    if iszero(s)
        error("    stepsizes_to_bound_trust_region: step s can't be zero")
    end

    a = s' * s
    b = x' * s
    c = x' * x - trust_radius^2

    if c > 0
        error("     stepsizes_to_bound_trust_region: x itself is not within the trust region")
    end

    # Root from one fourth of the discriminant.
    d = sqrt(b * b - a * c)

    # Computations below avoid loss of significance, see "Numerical Recipes".
    q = -(b + abs(d) * sign(b))
    t¹ = q / a
    t² = c / q

    if t¹ < t²
        t⁻ = t¹
        t⁺ = t²
    else
        t⁻ = t²
        t⁺ = t¹
    end

    return t⁻, t⁺
end

"""
    compute_newton_step(x, gn, g_hat, H_hat, LB, UB)

Compute and evaluate the full Inexact Newton step if feasible.

# Arguments
- `x`: The current point.
- `gn`: The Inexact Newton direction.
- `g_hat`: The scaled gradient vector.
- `H_hat`: The scaled Hessian function.
- `LB`: Lower bounds for variables.
- `UB`: Upper bounds for variables.

# Returns
- `is_feasible`: Boolean indicating if step is feasible.
- `step`: The Newton step if feasible, otherwise nothing.
- `step_hat`: The scaled Newton step if feasible, otherwise nothing.
- `step_value`: The quadratic value if feasible, otherwise nothing.
"""
function compute_newton_step(
    x::AbstractVector{T},
    gn::AbstractVector{T},
    gn_hat::AbstractVector{T},
    g_hat::AbstractVector{T},
    H_hat,
    LB::AbstractVector{T},
    UB::AbstractVector{T}
) where {T<:Real}
    if all_within_bounds(x + gn, LB, UB)
        step_value = evaluate_quadratic(H_hat, g_hat, gn_hat)
        @debug "Full Inexact Newton step is feasible"
        return true, gn, gn_hat, step_value
    end
    return false, nothing, nothing, nothing
end

"""
    compute_reflected_step(x, gn, gn_hat, g_hat, H_hat, D, trust_radius, theta, LB, UB)

Compute the Reflected Inexact Newton step.

# Arguments
- `x`: The current point.
- `gn`: The Inexact Newton direction.
- `gn_hat`: The scaled Inexact Newton direction.
- `g_hat`: The scaled gradient vector.
- `H_hat`: The scaled Hessian function.
- `D`: The scaling vector.
- `trust_radius`: The trust region radius.
- `theta`: Safety factor for feasibility (< 1).
- `LB`: Lower bounds for variables.
- `UB`: Upper bounds for variables.

# Returns
- `step`: The reflected step vector.
- `step_hat`: The scaled reflected step vector.
- `step_value`: The quadratic value at the reflected step.
"""
function compute_reflected_step(
    x::AbstractVector{T},
    gn::AbstractVector{T},
    gn_hat::AbstractVector{T},
    g_hat::AbstractVector{T},
    H_hat,
    D::AbstractVector{T},
    trust_radius::T,
    theta::T,
    LB::AbstractVector{T},
    UB::AbstractVector{T}
) where {T<:Real}
    # Compute step to boundary
    p_steplength, boundary_hit = stepsize_to_bound_feasible_region(x, gn, LB, UB)
    
    # Get the reflection direction
    rf_hat = copy(gn_hat)
    rf_hat[boundary_hit] = -rf_hat[boundary_hit]
    rf = D .* rf_hat
    
    # Boundary point and scaled step
    boundary_step = p_steplength * gn
    boundary_step_hat = p_steplength * gn_hat
    x_boundary = x + (one(T) - T(0.01)) * boundary_step  # Slightly inside boundary
    
    # Compute limits for the reflection step
    _, to_trust = stepsizes_to_bound_trust_region(boundary_step_hat, rf_hat, trust_radius)
    to_feasible, _ = stepsize_to_bound_feasible_region(x_boundary, rf, LB, UB)
    
    # Find bounds on reflection step
    rf_steplength = min(to_trust, to_feasible)
    
    # Default to infinite value (will not be chosen)
    rf_value = convert(T, Inf)
    rf_result = copy(gn)  # Default value
    rf_hat_result = copy(gn_hat)  # Default value
    
    # Calculate reflection step if possible
    if rf_steplength > zero(T)
        # Lower bound: slightly back from boundary
        rf_steplength_l = (one(T) - theta) * p_steplength / rf_steplength
        
        # Upper bound: either trust region boundary or feasible region boundary
        if rf_steplength == to_feasible
            rf_steplength_u = theta * to_feasible  # Stay slightly inside feasible region
        else
            rf_steplength_u = to_trust  # Go to trust region boundary
        end
        
        # Check if reflection range is valid
        if rf_steplength_l <= rf_steplength_u
            # Optimize along reflection direction
            a, b, c = build_quadratic_1d(H_hat, g_hat, rf_hat, boundary_step_hat)
            @debug "Computing reflection step by minimizing quadratic"
            optimal_t, rf_value = minimize_quadratic_1d(a, b, rf_steplength_l, rf_steplength_u, c)
            
            # Compute resulting step vectors
            rf_hat_result = boundary_step_hat + optimal_t * rf_hat
            rf_result = D .* rf_hat_result
        end
    else
        @debug "Reflection step invalid: rf_steplength = $rf_steplength"
    end
    
    return rf_result, rf_hat_result, rf_value
end

"""
    compute_interior_newton_step(x, gn, gn_hat, g_hat, H_hat, D, theta, LB, UB)

Compute a scaled Newton step that ensures strict interior feasibility.
First scales the Newton direction to reach the boundary of the feasible region,
then scales back by theta to stay strictly inside.

# Arguments
- `x`: Current point.
- `gn`: Original Inexact Newton direction.
- `gn_hat`: Scaled Inexact Newton direction.
- `g_hat`: Scaled gradient vector.
- `H_hat`: Scaled Hessian function.
- `D`: Scaling vector.
- `theta`: Safety factor for feasibility (< 1).
- `LB`: Lower bounds for variables.
- `UB`: Upper bounds for variables.

# Returns
- `step`: The interior Newton step.
- `step_hat`: The scaled interior Newton step.
- `step_value`: The quadratic value at the interior Newton step.
"""
function compute_interior_newton_step(
    x::AbstractVector{T},
    gn::AbstractVector{T},
    gn_hat::AbstractVector{T},
    g_hat::AbstractVector{T},
    H_hat,
    D::AbstractVector{T},
    theta::T,
    LB::AbstractVector{T},
    UB::AbstractVector{T}
) where {T<:Real}
    # First find the step to the boundary, just as in the original implementation
    p_steplength, _ = stepsize_to_bound_feasible_region(x, gn, LB, UB)
    
    # Scale the Newton direction to reach the boundary
    boundary_step = p_steplength * gn
    boundary_step_hat = p_steplength * gn_hat
    
    # Then apply theta to stay strictly inside the feasible region
    step = theta * boundary_step
    step_hat = theta * boundary_step_hat
    
    @debug "Computing interior Newton step value (from boundary step scaled by theta)"
    step_value = evaluate_quadratic(H_hat, g_hat, step_hat)
    
    return step, step_hat, step_value
end

"""
    compute_steepest_descent_step(x, g_hat, D, trust_radius, theta, LB, UB)

Compute the steepest descent step.

# Arguments
- `x`: The current point.
- `g_hat`: The scaled gradient vector.
- `H_hat`: The scaled Hessian function.
- `D`: The scaling vector.
- `trust_radius`: The trust region radius.
- `theta`: Safety factor for feasibility (< 1).
- `LB`: Lower bounds for variables.
- `UB`: Upper bounds for variables.

# Returns
- `step`: The steepest descent step.
- `step_hat`: The scaled steepest descent step.
- `step_value`: The quadratic value at the steepest descent step.
"""
function compute_steepest_descent_step(
    x::AbstractVector{T},
    g_hat::AbstractVector{T},
    H_hat,
    D::AbstractVector{T},
    trust_radius::T,
    theta::T, 
    LB::AbstractVector{T},
    UB::AbstractVector{T}
) where {T<:Real}
    # Steepest descent direction
    sd_hat = -g_hat
    sd_dir = D .* sd_hat
    
    # Calculate step limits
    sd_hat_norm = norm(sd_hat)
    to_trust = sd_hat_norm > 0 ? trust_radius / sd_hat_norm : convert(T, Inf)
    to_feasible, _ = stepsize_to_bound_feasible_region(x, sd_dir, LB, UB)
    
    # Determine maximum step length
    if to_feasible < to_trust
        sd_steplength_max = theta * to_feasible  # Stay slightly inside feasible region
    else
        sd_steplength_max = to_trust  # Limited by trust region
    end
    
    # Optimize along steepest descent direction
    a, b, c = build_quadratic_1d(H_hat, g_hat, sd_hat, zero(sd_hat))
    @debug "Computing steepest descent step by minimizing quadratic"
    optimal_t, step_value = minimize_quadratic_1d(a, b, zero(T), sd_steplength_max, zero(T))
    
    # Compute resulting step vectors
    step_hat = optimal_t * sd_hat
    step = D .* step_hat
    
    return step, step_hat, step_value
end

"""
    choose_step(x, H_hat, g_hat, gn, gn_hat, D, trust_radius, theta, LB, UB)

Choose the best step from three potential steps: Inexact Newton, Reflected Inexact Newton, or Steepest Descent.
Selects the step that minimizes the quadratic approximation while respecting trust region and feasibility constraints.

# Arguments
- `x`: The current point.
- `H_hat`: The scaled Hessian function.
- `g_hat`: The scaled gradient vector.
- `gn`: The Inexact Newton direction.
- `gn_hat`: The scaled Inexact Newton direction.
- `D`: The scaling vector.
- `trust_radius`: The trust region radius.
- `theta`: Safety factor for feasibility (< 1).
- `LB`: Lower bounds for variables.
- `UB`: Upper bounds for variables.

# Returns
- `step`: The chosen step.
- `step_hat`: The chosen step in the scaled space.
- `step_value`: The value of the quadratic approximation at the chosen step.
"""
function choose_step(
    x::AbstractVector{T},
    H_hat,
    g_hat::AbstractVector{T},
    gn::AbstractVector{T},
    gn_hat::AbstractVector{T},
    D::AbstractVector{T},
    trust_radius::T,
    theta::T,
    LB::AbstractVector{T},
    UB::AbstractVector{T}
) where {T<:Real}
    # First check if the full Newton step is feasible
    newton_feasible, newton_step, newton_step_hat, newton_value = 
        compute_newton_step(x, gn, gn_hat, g_hat, H_hat, LB, UB)
    
    if newton_feasible
        @debug "Choosing full Inexact Newton step"
        return newton_step, newton_step_hat, newton_value
    end
    
    # Compute the three candidate steps
    @debug "Computing candidate steps"
    
    # 1. Reflected Newton step
    rf_step, rf_step_hat, rf_value = compute_reflected_step(
        x, gn, gn_hat, g_hat, H_hat, D, trust_radius, theta, LB, UB)
    
    # 2. Interior Newton step - Fixed: now passing all required arguments
    gn_step, gn_step_hat, gn_value = compute_interior_newton_step(
        x, gn, gn_hat, g_hat, H_hat, D, theta, LB, UB)
    
    # 3. Steepest descent step
    sd_step, sd_step_hat, sd_value = compute_steepest_descent_step(
        x, g_hat, H_hat, D, trust_radius, theta, LB, UB)
    
    # Select the step with the minimum value
    step_values = [gn_value, rf_value, sd_value]
    min_value, idx = findmin(step_values)
    
    @debug "Step values - Newton: $(gn_value), Reflected: $(rf_value), Steepest Descent: $(sd_value)"
    
    if idx == 1
        @debug "Choosing scaled Inexact Newton step"
        return gn_step, gn_step_hat, gn_value
    elseif idx == 2
        @debug "Choosing Reflected Inexact Newton step"
        return rf_step, rf_step_hat, rf_value
    else
        @debug "Choosing Steepest Descent step"
        return sd_step, sd_step_hat, sd_value
    end
end

"""
    function _calculate_ratio(actual_reduction, g, H, s, ŝ, C, modfified_reduction_for_ratio, to)

Calculate the ratio of actual to predicted reduction in the cost function.
The ratio is calculated as the actual reduction divided by the predicted reduction. The predicted reduction is computed using the quadratic approximation of the cost function.
The actual reduction is modified by subtracting the quadratic term of the predicted reduction if `modfified_reduction_for_ratio` is true.

# Arguments
- `actual_reduction::T`: The actual reduction in the cost function.
- `g::AbstractVector{T}`: The gradient vector.
- `H::Function`: The Hessian operator.
- `s::AbstractVector{T}`: The step vector.
- `ŝ::AbstractVector{T}`: The scaled step vector.
- `C::AbstractVector{T}`: The scaling vector.
- `modfified_reduction_for_ratio::Bool`: A flag indicating whether to modify the actual reduction for the ratio calculation.
- `to::TimerOutputs.TimerOutput`: The timer output object for logging.

# Returns
`ratio::T`: The ratio of actual to predicted reduction.
"""
function _calculate_ratio(actual_reduction, g, H, s, ŝ, C, modfified_reduction_for_ratio, to)

    @timeit to "Predict reduction" predicted_reduction = -((g' * s) + (1 // 2) * s' * (H * s))

    if modfified_reduction_for_ratio
        @timeit to "Modify reduction" modified_reduction = actual_reduction - ((1 // 2) * ŝ' * (C .* ŝ))
        ratio = modified_reduction / predicted_reduction
    else
        ratio = actual_reduction / predicted_reduction
    end

    return ratio
end