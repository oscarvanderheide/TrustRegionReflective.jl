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

    # initialize empty array to store Steihaug steps
    steps = typeof(g)[]
    sizehint!(steps, maxit * length(d))

    if norm(r) < tol
        @info "        Nothing to gain, residual is already small enough from the start"
        push!(steps, z)
    end

    iter = 1

    while iter <= maxit

        
        # @info "        Iteration $(iter) of inner loop, current CG-residual: $(norm(r))"
        Hd = H(d)
        dHd = d' * Hd
        #  realResidual = 0.5 * p' * B(p) + g' * p    # This thing
        #  should be monotonically decreasing (and it does)

        # if dHd < ϵ
        if dHd < ϵ
            @info "        Direction of negative curvature encountered: should not occur because of Gauss-Newton method?"
            τ = positive_stepsize_to_bound_trust_region(z, d, Δ)
            push!(steps, z + τ * d)
            break
        end

        α = (r' * Y) / dHd
        z_new = z + α * d

        if norm(z_new) > Δ
            @info "        Fell out of trust radius after iteration $(iter)"
            τ = positive_stepsize_to_bound_trust_region(z, d, Δ)
            push!(steps, z + τ * d)
            break
        end

        r_new = r + α * Hd
        norm_r_new = norm(r_new)

        if norm_r_new < tol
            @info "        Steihaug-CG converged with CG-residual = $(norm_r_new) after iteration $(iter)"
            push!(steps, z_new)
            break
        end

        Y_new = P(r_new)
        β = (Y_new' * r_new) / (Y' * r)
        d_new = -Y_new + β * d

        # Prepare for next iteration
        r = r_new
        d = d_new
        z = z_new
        Y = Y_new

        if iter == maxit
            @info "        Steihaug-CG failed to converge, CG-residual = $(norm_r_new)"
            push!(steps, z_new)
            break
        else
            iter = iter + 1
            push!(steps, z_new)
        end
    end

    return steps
end