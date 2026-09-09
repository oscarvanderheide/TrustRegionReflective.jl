# function steihaug(H, g, Δ, P, maxit, tol, z0; store_all_steps=false)

#     # This implementation is really just a copy-paste of Algorithm 7.2
#     # of Nocedal & Wright

#     # H  = Hessian multiplication function
#     # r  = Current residual
#     # d  = Current B-conjugate search direction
#     # z  = Linear combination of previous B-conjugate directions

#     @info "    Steihaug CG:"
#     ϵ = eps(eltype(g))
#     η = min(tol, norm(g) / length(g)) # Should give quadratic convergence near solution
#     # η = min( 0.5, sqrt(norm(g)) ); # Should give superlinear convergence near solution

#     tol = η * norm(g)
#     # @info "        Current error tolerance for Steihaug is $(tol)")
#     # @info "        Current trust radius is $(Δ)", )
#     # Initialize things for CG algorithm
#     z = zeros(length(g))
#     r = g

#     Y = P(r)
#     d = -Y

#     if store_all_steps
#         # initialize empty array to store Steihaug steps
#         steps = eltype(g)[]
#         sizehint!(steps, maxit * length(d))
#     end

#     if norm(r) < tol
#         @info "        Nothing to gain, residual is already small enough from the start"
#         push!(steps, z)
#     end

#     iter = 1

#     while iter <= maxit

#         print(".")

#         # @info "        Iteration $(iter) of inner loop, current CG-residual: $(norm(r))"
#         Hd = H(d)
#         dHd = d' * Hd
#         #  realResidual = 0.5 * p' * B(p) + g' * p    # This thing
#         #  should be monotonically decreasing (and it does)

#         if dHd < ϵ
#             @info "        Direction of negative curvature encountered: should not occurbecause of Gauss-Newton method?"
#             τ = positive_stepsize_to_bound_trust_region(z, d, Δ)
#             step = z + τ * d
#             if store_all_steps
#                 push!(steps, step)
#             end
#             break
#         end

#         α = (r' * Y) / dHd
#         z_new = z + α * d

#         if norm(z_new) > Δ
#             @info "        Fell out of trust radius after iteration $(iter)"
#             τ = positive_stepsize_to_bound_trust_region(z, d, Δ)
#             step = z + τ * d
#             if store_all_steps
#                 push!(steps, step)
#             end
#             break
#         end

#         r_new = r + α * Hd
#         norm_r_new = norm(r_new)

#         if norm_r_new < tol
#             @info "        Steihaug-CG converged with CG-residual = $(norm_r_new) afteriteration $(iter)"
#             step = z_new
#             if store_all_steps
#                 push!(steps, step)
#             end
#             break
#         end

#         Y_new = P(r_new)
#         β = (Y_new' * r_new) / (Y' * r)
#         d_new = -Y_new + β * d

#         # Prepare for next iteration
#         r = r_new
#         d = d_new
#         z = z_new
#         Y = Y_new

#         if iter == maxit
#             @info "        Steihaug-CG failed to converge, CG-residual = $(norm_r_new)"
#             step = z_new
#             if store_all_steps
#                 push!(steps, step)
#             end
#             break
#         else
#             iter = iter + 1
#             if store_all_steps
#                 push!(steps, z_new)
#             end
#         end
#     end

#     if store_all_steps
#         return steps = reshape(steps, length(g), :)
#     else
#         return step
#     end

# end

"""
    steihaug_store_steps(H, g, Δ, P, maxit, tol, z0)

This function implements the Steihaug conjugate gradient (CG) method for solving a trust region subproblem. It is used to find the minimum of a quadratic function subject to a trust region constraint.

# Arguments
- `H`: Hessian multiplication function.
- `g`: Current residual.
- `Δ`: Current trust radius.
- `P`: Preconditioner function.
- `maxit`: Maximum number of iterations.
- `tol`: Tolerance for convergence.
- `z0`: Initial solution.

# Output
- `steps`: Array of intermediate solutions obtained during the Steihaug CG iterations.

The function follows Algorithm 7.2 of Nocedal & Wright.
"""
function steihaug_store_steps(H, g, Δ, P, maxit, tol, z0)

    # This implementation is really just a copy-paste of Algorithm 7.2
    # of Nocedal & Wright

    # H  = Hessian multiplication function
    # r  = Current residual
    # d  = Current B-conjugate search direction
    # z  = Linear combination of previous B-conjugate directions

    @info "    Steihaug CG:"
    ϵ = eps()
    ϵ = eps(eltype(g))
    η = min(tol, norm(g) / length(g)) # Should give quadratic convergence near solution
    # η = min( 0.5, sqrt(norm(g)) ); # Should give superlinear convergence near solution

    tol = η * norm(g)
    # @info "        Current error tolerance for Steihaug is $(tol)")
    # @info "        Current trust radius is $(Δ)", )
    # Initialize things for CG algorithm
    z = zero(g)
    r = g

    Y = P(r)
    d = -Y

    # initialize empty arrays to store Steihaug steps and their precomputed norms
    steps = typeof(g)[]
    step_norms = eltype(g)[]
    sizehint!(steps, maxit)
    sizehint!(step_norms, maxit)

    norm_r = norm(r)
    if norm_r <= tol || iszero(norm_r)
        @info "        Nothing to gain, residual is already small enough from the start"
        push!(steps, z)
        push!(step_norms, zero(eltype(g)))
        return steps, step_norms
    end

    iter = 1

    while iter <= maxit


        # @info "        Iteration $(iter) of inner loop, current CG-residual: $(norm(r))"
        Hd = H(d)
        dHd = d' * Hd
        #  realResidual = 0.5 * p' * B(p) + g' * p    # This thing
        #  should be monotonically decreasing (and it does)

        # A zero preconditioned direction cannot reach the boundary; return the current
        # iterate instead of forming Inf * 0. Only reachable when the preconditioner maps
        # the residual to exactly zero, which is a bug in the preconditioner: warn, because
        # the solver reads the resulting zero step as convergence.
        if iszero(norm(d))
            @warn "        Steihaug-CG: preconditioned direction is exactly zero; returning a zero step. Check the preconditioner."
            # z is already the last entry of `steps` on every iteration but the first.
            if isempty(steps)
                push!(steps, z)
                push!(step_norms, norm(z))
            end
            break
        end

        # Only nonpositive curvature justifies a boundary step. An absolute threshold,
        # even after dividing by ‖d‖², misclassifies small positive eigenvalues when the
        # objective is rescaled. Comparing against ‖d‖‖Hd‖ -- the natural size of dHd --
        # is invariant under both rescalings, and unlike `dHd <= 0` it also catches dHd
        # underflowing to zero, which would otherwise send a full boundary step along a
        # direction made of nothing but round-off.
        if dHd <= ϵ * norm(d) * norm(Hd)
            @info "        Direction of negative curvature encountered: should not occur because of Gauss-Newton method?"
            τ = positive_stepsize_to_bound_trust_region(z, d, Δ)
            step = z + τ * d
            push!(steps, step)
            push!(step_norms, norm(step))
            break
        end

        α = (r' * Y) / dHd
        z_new = z + α * d
        norm_z_new = norm(z_new)

        if norm_z_new > Δ
            @info "        Fell out of trust radius after iteration $(iter)"
            τ = positive_stepsize_to_bound_trust_region(z, d, Δ)
            step = z + τ * d
            push!(steps, step)
            push!(step_norms, norm(step))
            break
        end

        r_new = r + α * Hd
        norm_r_new = norm(r_new)

        if norm_r_new <= tol
            @info "        Steihaug-CG converged with CG-residual = $(norm_r_new) after iteration $(iter)"
            push!(steps, z_new)
            push!(step_norms, norm_z_new)
            break
        end

        Y_new = P(r_new)
        # Guard against division by zero: when Y'*r ≈ 0 the CG β blows up.
        # This can happen when the preconditioner maps the residual to near-zero.
        # Scaled by ‖Y‖‖r‖ for the same reason as the curvature test above: the bare
        # comparison measures how short the vectors are, not how close to orthogonal.
        Yr = Y' * r
        if abs(Yr) <= ϵ * norm(Y) * norm(r)
            @info "        Steihaug-CG: Y'*r ≈ 0, terminating to avoid NaN in β"
            push!(steps, z_new)
            push!(step_norms, norm_z_new)
            break
        end
        β = (Y_new' * r_new) / Yr
        d_new = -Y_new + β * d

        # Prepare for next iteration
        r = r_new
        d = d_new
        z = z_new
        Y = Y_new

        if iter == maxit
            @info "        Steihaug-CG failed to converge, CG-residual = $(norm_r_new)"
            push!(steps, z_new)
            push!(step_norms, norm_z_new)
            break
        else
            iter = iter + 1
            push!(steps, z_new)
            push!(step_norms, norm_z_new)
        end
    end

    return steps, step_norms
end
