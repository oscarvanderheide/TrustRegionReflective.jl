function trust_region_reflective(
    objective::Function, 
    x0::AbstractVector{T}, 
    LB::AbstractVector{T}, 
    UB::AbstractVector{T}, 
    callback::Function,
    to::TimerOutputs.TimerOutput,
    options::TRFOptions = TRFOptions{T}(),
    ) where T<:Real

    # Load the initial guess
    x = x0

    @info "    Calling f,r,g,H = objective(x,'frgH')"
    @timeit to "frgH" f, r, g, H = objective(x, "frgH")

    # Set initial trust radius Δ
    Δ = norm(x)
    if Δ == zero(T)
        # When x0 is all zeros, use a default value based on problem dimensions
        Δ = one(T) * sqrt(length(x))
    end
    
    # Set initial trust radius with the guaranteed non-zero norm
    Δ = options.init_scale_radius * Δ |> eltype(x)
    Δlimit = Δ * 1E-10 |> eltype(x)

    iter = 1
    converged = false

    t = 0.0 |> eltype(x)

    @info "Initialize solver state"

    @timeit to "Initialize state" state = SolverState(x, f, r, t)    

    while ((iter < (options.max_iter_trf + 1)) && (!converged))

        # We determine two scaling fawctors: one from the diagonal of JᴴJ. This one makes parameters with low curvature move faster.
        # The other is related to the distance of parameters to their respective boundaries.
        # It slows down parameters that are close to their boundaries.

        @info "ITERATION #$(iter)"

        if iter > 1
            @info "    Calling f,r,g,H = objective(x,'frgH')"
            @timeit to "Objective (frgH)" f, r, g, H = objective(x, "frgH")
        end

        @info "    f: $(f)"
        @info "    Δ: $(Δ)"
        
        # Check for convergence based on gradient norm and function value
        g_norm = norm(g)
        tol = options.tol_steihaug
        
        # For least squares problems, check if we're close enough to the minimum
        if (g_norm < tol) || (iter > 1 && abs(state.f[end-1] - f) < tol * max(1.0, abs(f)))
            @info "Convergence achieved: gradient norm $(g_norm) or function change below tolerance"
            converged = true
            break
        end

        # # Ensure x is within bounds (with a small numerical tolerance)
        # @timeit to "Snap to bounds" x = snap_to_bounds(x, LB, UB)

        @timeit to "CL scaling" v, dv = coleman_li_scaling_factors(x, g, LB, UB)

        # Make scaling operator and scale gradient and Hessian
        @timeit to "D" D = sqrt.(v)
        @timeit to "g_hat" ĝ = D .* g
        @timeit to "C" C = dv .* g
        Ĥ = x -> (D .* (H * (D .* x))) + (C .* x)

        step_accepted = false
        perform_steihaug = true
        sh_iter = -1

        steps = nothing

        while !step_accepted

            # Compute potential step using Steihaug
            P = y -> y # Preconditioner, currently not used
            z0 = zero(ĝ)
            if perform_steihaug

                @timeit to "Steihaug" steps = steihaug_store_steps(Ĥ, ĝ, Δ, P, options.max_iter_steihaug, options.tol_steihaug, z0)
                ŝ = steps[end]
            else
                ŝ = steps[sh_iter]
            end

            s = D .* ŝ
            # Select best step taking into account feasible region
            θ = max(0.995, 1 - norm(v .* g, Inf)) |> eltype(x)
            @info "Choose step"

            @timeit to "Choose step" step, step_hat, step_value = choose_step(x, Ĥ, ĝ, s, ŝ, D, Δ, θ, LB, UB)
            @timeit to "x_new" x_new = x + step

            # Compute new objective
            @info "    Calling f,r = objective(x_new,'fr')"
            @timeit to "Objective (fr)" f_new, r_new = objective(x_new, "fr")
            
            # Compute actual reduction and modification based on the scaling
            actual_reduction = f - f_new

            # Calculate ratio of actual to predicted reduction
            ratio = _calculate_ratio(actual_reduction, g, H, s, ŝ, C, options.modfified_reduction_for_ratio, to)

            # Determine whether to accept the step
            if (actual_reduction > 0 && ratio > 0.1) || Δ < Δlimit
                @info "    Step accepted"
                step_accepted = true

                @timeit to "Adjust Δ" Δ = adjust_trust_radius(ratio, ŝ, Δ, options.min_ratio)

                x = x_new
                f = f_new
                r = r_new

                @timeit to "Update state" begin
                    state.x = hcat(state.x, x)
                    state.f = vcat(state.f, f)
                    state.r = hcat(state.r, r)
                    state.t = vcat(state.t, t)
                end

                perform_steihaug = true
            else
                @info "    Find a smaller step (reduction: $(actual_reduction)"

                sh_iter = length(steps)

                while sh_iter >= length(steps)
                    Δ = Δ / 2
                    @info "   Trust radius reduced to: $(Δ)"
                    @timeit to "Find smaller step" sh_iter = findlast(norm.(steps) .<= Δ)

                    if sh_iter === nothing
                        sh_iter = 1
                        break
                    end

                end
                if sh_iter == 1
                    perform_steihaug = true
                else
                    perform_steihaug = false
                end
            end
        end # Step accepted

        callback(iter, state)
        
        iter += 1

    end

    return state.x[:, end]
end
