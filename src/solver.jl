function trust_region_reflective(
    objective::AbstractFunction, 
    x0::AbstractVector{T}, 
    LB::AbstractVector{T}, 
    UB::AbstractVector{T}, 
    options::TRFOptions, 
    callback::AbstractFunction, 
    to::TimerOutputs.TimerOutput,
    modfified_reduction_for_ratio::Bool = false
    ) where T<:Real

    # Load the initial guess
    x = x0

    @info "    Calling f,r,g,H = objective(x,'frgH')"
    @timeit to "frgH" f, r, g, H = objective(x, "frgH")

    # # Distance to boundary
    # v, dv = coleman_li_scaling_factors(x, g, LB, UB);
    # # Initial trust radius
    # Δ = 0.1 * norm( x ./ (sqrt.(v) ) );o
    @info "Setting initial trust radius"
    Δ = options.init_scale_radius * norm(x) |> eltype(x)
    Δlimit = Δ * 1E-10 |> eltype(x)

    iter = 1
    converged = false

    t = 0.0

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

        @timeit to "CL scaling" v, dv = coleman_li_scaling_factors(x, g, LB, UB)

        # Make scaling operator and scale gradient and Hessian
        @timeit to "D" D = sqrt.(v)
        @timeit to "g_hat" ĝ = D .* g
        @timeit to "C" C = dv .* g
        Ĥ = x -> (D .* (H * (D .* x))) + (C .* x)

        # Ĥ = Krylov.LinearOperators.LinearOperator(length(x), length(x), true, false, x -> (D .* (H * ( D .* x) ) ) + (C .* x))

        step_accepted = false
        perform_steihaug = true
        sh_iter = -1

        # @timeit to "Allocate steps" steps = zeros(length(x), options.max_iter_steihaug)
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

            # ŝ = Krylov.cg(Ĥ, -ĝ, atol = options.tol_steihaug, rtol = options.tol_steihaug, itmax = options.max_iter_steihaug, radius = Δ, verbose = true)[1]

            s = D .* ŝ
            # Select best step taking into account feasible region
            θ = max(0.995, 1 - norm(v .* g, Inf)) |> eltype(x)
            @info "Choose step"

            @timeit to "Choose step" step, step_hat, step_value = chooseStep(x, Ĥ, ĝ, s, ŝ, D, Δ, θ, LB, UB)
            @timeit to "x_new" x_new = x + step
            # Compute new objective
            @info "    Calling f,r = objective(x_new,'fr')"
            @timeit to "Objective (fr)" f_new, r_new = objective(x_new, "fr")
            # Compute reduction
            actualReduction = -(f_new - f)
            @timeit to "Predict reduction" predictedReduction = -((g' * s) + (1 // 2) * s' * (H * s))
            # predictedReduction2  = -( (ĝ' * ŝ) + (1//2) * ŝ' * Ĥ(ŝ));
            @timeit to "Modify reduction" modifiedReduction = -(f_new - f + (1 // 2) * ŝ' * (C .* ŝ))
            ratio = modifiedReduction / predictedReduction

            @info "   reduction: $(actualReduction)"
            @info "   ratio: $(ratio)"

            if (actualReduction > 0 && ratio > 0.1) || Δ < Δlimit
                @info "    Step accepted"
                step_accepted = true

                @timeit to "Adjust Δ" Δ = adjust_trust_radius(ratio, ŝ, Δ, options.min_ratio)

                x = x_new
                f = f_new
                r = r_new

                @timeit to "Update state" begin
                    state.x = hcat(state.x, x)
                    state.f = hcat(state.f, f)
                    state.r = hcat(state.r, r)
                    state.t = hcat(state.t, t)
                end

                perform_steihaug = true
            else
                @info "    Find a smaller step (reduction: $(actualReduction)"

                # sometimes this part keeps on iterating forever, need add
                # a counter and have some maximum nr of tries

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

        # @timeit to "TimerOutput" begin
        #     TimerOutputs.complement!(to)
        #     print_timer(to)
        # end

    end

    return state.x[:, end]
    # return state
end



# function solver(objective, x0, LB, UB, options::TRFOptions, logger, plotfun, write_to_disk)

#     # Load the initial guess
#     x = x0

#     @info "    Calling f,r,g,H = objective(x,2)"
#     f, r, g, H = objective(x, 2)

#     # # Distance to boundary
#     # v, dv = coleman_li_scaling_factors(x, g, LB, UB);
#     # # Initial trust radius
#     # Δ = 0.1 * norm( x ./ (sqrt.(v) ) );o
#     @info "Setting initial trust radius")
#     Δ = options.init_scale_radius * norm(x) |> eltype(x)
#     Δlimit = Δ * 1E-10 |> eltype(x)


#     iter = 1
#     converged = false
#     t = 0.0

#     state = SolverState(x, f, r, t)
#     options.save_every_iter && write_to_disk(state)

#     while ((iter < (options.max_iter_trf + 1)) && (!converged))

#         # We determine two scaling fawctors: one from the diagonal of JᴴJ. This one makes parameters with low curvature move faster.
#         # The other is related to the distance of parameters to their respective boundaries.
#         # It slows down parameters that are close to their boundaries.

#         @info "ITERATION #$(iter)"
#

#         if iter > 1
#             @info "    Calling f,r,g,H = objective(x,2)"
#             f, r, g, H = objective(x, 2)
#         end

#         @info "    f: $(f)",
#         @info "    Δ: $(Δ)"

#         v, dv = coleman_li_scaling_factors(x, g, LB, UB, logger)

#         # Make scaling operator and scale gradient and Hessian
#         D = sqrt.(v)
#         ĝ = D .* g
#         C = dv .* g
#         Ĥ = x -> (D .* (H * (D .* x))) + (C .* x)
#         # Ĥ = Krylov.LinearOperators.LinearOperator(length(x), length(x), true, false, x -> (D .* (H * ( D .* x) ) ) + (C .* x))

#         step_accepted = false

#         while !step_accepted

#             # Compute potential step using Steihaug
#             P = y -> y # Preconditioner, currently not used
#             z0 = zeros(length(ĝ))
#             ŝ = steihaug(Ĥ, ĝ, Δ, P, options.max_iter_steihaug, options.tol_steihaug, z0, logger)

#             # ŝ = Krylov.cg(Ĥ, -ĝ, atol = options.tol_steihaug, rtol = options.tol_steihaug, itmax = options.max_iter_steihaug, radius = Δ, verbose = true)[1]

#             s = D .* ŝ
#             x_new = x + s
#             # Select best step taking into account feasible region
#             θ = max(0.995, 1 - norm(v .* g, Inf)) |> eltype(x)
#             step, step_hat, step_value = chooseStep(x, Ĥ, ĝ, s, ŝ, D, Δ, θ, LB, UB, logger)
#             x_new = x + step
#             # Compute new objective
#             @info "    Calling f,r = objective(x,0)"
#             f_new, r_new = objective(x_new, 0)
#             # Compute reduction
#             actualReduction = -(f_new - f)
#             predictedReduction = -((g' * s) + 0.5 * s' * (H * s))
#             # predictedReduction2  = -( (ĝ' * ŝ) + 0.5 * ŝ' * Ĥ(ŝ));
#             modifiedReduction = -(f_new - f + 0.5 * ŝ' * (C .* ŝ))
#             ratio = modifiedReduction / predictedReduction

#             @info "   reduction: $(actualReduction)"
#             @info "   ratio: $(ratio)"

#             if (actualReduction > 0 && ratio > 0.1) || Δ < Δlimit
#                 @info "    Step accepted"
#                 step_accepted = true

#                 Δ = adjust_trust_radius(ratio, ŝ, Δ, logger, options.min_ratio)
#                 x = x_new
#                 f = f_new
#                 r = r_new

#                 t += tok()

#                 state.x = hcat(state.x, x)
#                 state.f = hcat(state.f, f)
#                 state.r = hcat(state.r, r)
#                 state.t = hcat(state.t, t)

#             else
#                 @info "    Find a smaller step (reduction: $(actualReduction)"
#                 Δ = 0.5 * Δ
#                 @info "   Trust radius reduced to: $(Δ)"
#             end
#         end # Step accepted

#         iter += 1

#         if options.save_every_iter
#             write_to_disk(state)
#             plotfun(x)
#         end

#     end

#     # write final results to disk
#     output = write_to_disk(state)

#     return output
# end
