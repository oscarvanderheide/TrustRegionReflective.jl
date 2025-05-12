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

    @info "Norm of step: $(norm(step)), Trust Radius: $(Δ)"

    norm_step = round(norm(step), digits=2)

    if norm_step > 1.01 * round(Δ, digits=2)
        error("    adjust_trust_radius: Norm of step ($norm_step) is greater than trust radius ($(round(Δ, digits=2)))")
    end

    if current_ratio < min_ratio # The trust region is too large. Reduce the radius.
        @info "Trust Radius too large"
        Δ = Δ / 4

    elseif (current_ratio > 1 / 2) && (norm(step) > (0.95 * Δ))

        # The trust region seems to be too small. Increase the radius.
        @info "Trust Radius too small"
        Δ = 2 * Δ

    else # The trust region seems to be fine. Keep the radius the same.

        @info "Trust Radius just fine"

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
    choose_step(x, H_hat, g_hat, gn, gn_hat, D, trust_radius, theta, LB, UB)

    (x::AbstractVector{T}, H_hat::AbstractMatrix{T}, g_hat::AbstractVector{T}, gn::AbstractVector{T}, gn_hat::AbstractVector{T}, D::AbstractVector{T}, trust_radius::T, theta::T, LB::AbstractVector{T}, UB::AbstractVector{T}) where T <: Real

Choose the best step from three potential steps.

This function considers three potential steps: the Inexact Newton step, the Reflected Inexact Newton step, and the Steepest Descent step. It chooses the step that minimizes the quadratic approximation and lies within the trust region and the feasible region.

# Arguments
- `x`: The current point.
- `H_hat`: The Hessian matrix.
- `g_hat`: The gradient vector.
- `gn`: The Inexact Newton direction.
- `gn_hat`: The Inexact Newton direction in the scaled space.
- `D`: The scaling vector.
- `trust_radius`: The trust region radius.
- `theta`: The scaling factor for the step size.
- `LB`: The lower bounds for the variables.
- `UB`: The upper bounds for the variables.

# Returns
- `step`: The chosen step.
- `step_hat`: The chosen step in the scaled space.
- `step_value`: The value of the quadratic approximation at the chosen step.
"""
function chooseStep(
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

    # Ok so we found a step p that minimizes the quadratic approximation thingy and lies within the trust region.
    # If x + p also lies within the feasible region then this is the step we are going to take

    if all_within_bounds(x + gn, LB, UB)
        step = gn
        step_hat = gn_hat
        step_value = evaluate_quadratic(H_hat, g_hat, gn_hat)
        @info "          The Inexact Newton step was chosen"
        return step, step_hat, step_value
    end

    # If x + p does not lie within the feasible region,
    # we consider THREE different steps and take the best one

    # The steps should lie inside the trust region and inside the
    # feasible region. To check whether the steps lie inside the trust
    # region we use the _hat variables (then we can check their
    # 2-norms, otherwise we would have to mess with norms). To check
    # whether steps lie in the feasible region we work in the original
    # variables

    # POTENTIAL STEP 1: Move from x in direction p until you hit the feasible region boundary.
    #                   From there move in reflected direction rf
    p_steplength, boundary_hit = stepsize_to_bound_feasible_region(x, gn, LB, UB)

    # Reflected direction (it doesn't matter if we do the
    # reflection on rf_hat or on rf since the scaling thing is
    # diagonal)
    rf_hat = gn_hat
    rf_hat[boundary_hit] = -1 * rf_hat[boundary_hit]
    rf = D .* rf_hat

    # Add p_steplength * p to x so that we are sitting on the boundary
    gn = p_steplength * gn
    gn_hat = p_steplength * gn_hat
    x_on_boundary = x + (99 // 100) * gn

    # From (x + p), which lies on the boundary, we - at most - move in the direction r until
    # we hit either the trust region boundary or the feasible region boundary
    _, to_trust = stepsizes_to_bound_trust_region(gn_hat, rf_hat, trust_radius)
    to_feasible, _ = stepsize_to_bound_feasible_region(x_on_boundary, rf, LB, UB)

    # Find lower and upper bounds on a step size  along the reflected
    # direction, considering the strict feasibility requirement. There is no
    # single correct way to do that, the chosen approach seems to work best
    # on test problems.
    rf_steplength = min(to_trust, to_feasible)

    if rf_steplength > 0

        rf_steplength_l = (one(T) - theta) * p_steplength / rf_steplength # We need to be in the interior, hence the (1-theta) to move just a little bit away from the boundary

        if rf_steplength == to_feasible
            rf_steplength_u = theta * to_feasible # Multiply by theta < 1 to stay in the interior
        elseif rf_steplength == to_trust
            rf_steplength_u = to_trust
        end

    else
        @info "        rf_steplength <= 0? What's going on?"
        rf_steplength_l = zero(T)
        rf_steplength_u = -one(T)
    end

    # Check if reflection step is available.
    if rf_steplength_l <= rf_steplength_u
        a, b, c = build_quadratic_1d(H_hat, g_hat, rf_hat, gn_hat)
        @info "        minimize_quadratic_1d"
        rf_steplength, rf_value = minimize_quadratic_1d(a, b, rf_steplength_l, rf_steplength_u, c)
        rf_hat = rf_hat * rf_steplength
        rf_hat = rf_hat + gn_hat
        rf = D .* rf_hat
    else
        rf_value = eltype(x)(Inf) # Reflection step is bad in this case
    end

    # POTENTIAL STEP 2: x + theta * p (strictly interior point)

    # gn was previously scaled such that x + gn lies on the
    # boundary of the feasible region. For differentiability of
    # things we prefer to remain in the interior of the feasible
    # region. To this end, scale by some theta that is almost 1
    gn = theta * gn
    gn_hat = theta * gn_hat
    @info "        evaluate_quadratic"
    gn_value = evaluate_quadratic(H_hat, g_hat, gn_hat)

    # POTENTIAL STEP 3: The steepest descent direction

    sd_hat = -g_hat
    sd = D .* sd_hat

    to_trust = trust_radius / norm(sd_hat)
    to_feasible, _ = stepsize_to_bound_feasible_region(x, sd, LB, UB)

    if to_feasible < to_trust
        sd_steplength_max = theta * to_feasible
    else
        sd_steplength_max = to_trust
    end

    a, b, c = build_quadratic_1d(H_hat, g_hat, sd_hat, zero(T) * g_hat)
    @info "        minimize_quadratic_1d"
    sd_steplength, sd_value = minimize_quadratic_1d(a, b, zero(T), sd_steplength_max, zero(T))
    sd_hat = sd_steplength * sd_hat
    sd = sd_steplength * sd

    # Now choose the one that gives the smallest value

    values = [gn_value rf_value sd_value]
    minVal = minimum(values)
    index = findfirst(y -> y == minVal, values)
    index = index[1]
    @info "        gn: $(gn_value), rf: $(rf_value), sd: $(sd_value)"

    if index == 1
        step = gn
        step_hat = gn_hat
        step_value = gn_value
        @info "        The Inexact Newton step, restricted to feasible region, was chosen"
    end
    if index == 2
        step = rf
        step_hat = rf_hat
        step_value = rf_value
        @info "        The Reflected Inexact Newton step was chosen"
    end
    if index == 3
        step = sd
        step_hat = sd_hat
        step_value = sd_value
        @info "        The Steepest Descent step was chosen"
    end

    return step, step_hat, step_value

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