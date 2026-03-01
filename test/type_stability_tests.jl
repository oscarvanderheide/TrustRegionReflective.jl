using TrustRegionReflective
using Test
using CUDA
using LinearAlgebra
using LinearMaps

const TRF = TrustRegionReflective

@testset "Type Stability Tests" begin
    n = 5

    # Test each precision in a loop
    @testset "T = $T" for T in (Float64, Float32)
        x  = rand(T, n)
        s  = rand(T, n)
        LB = zeros(T, n)
        UB = ones(T, n) * 10
        g  = rand(T, n)
        step = rand(T, n) .* T(0.5)  # keep small so norm(step) < Δ
        Δ   = T(2)
        trust_radius = T(2)

        # Hessian operator (identity-like, returns same eltype)
        H = LinearMap(x -> T(2) .* x, n, n)

        # Scalars
        a, b, c = T(1), T(-2), T(1)
        lb, ub  = T(0), T(1)

        # Ensure x is inside bounds and trust region
        x_inside = clamp.(x, LB .+ T(0.01), UB .- T(0.01))
        x_small  = x .* (trust_radius / (T(2) * norm(x)))

        @testset "all_within_bounds" begin
            @test @inferred(TRF.all_within_bounds(x_inside, LB, UB)) isa Bool
        end

        @testset "snap_to_bounds" begin
            @test @inferred(TRF.snap_to_bounds(x_inside, LB, UB)) isa Vector{T}
        end

        @testset "stepsize_to_bound_feasible_region" begin
            result = @inferred TRF.stepsize_to_bound_feasible_region(x_inside, s, LB, UB)
            @test result[1] isa T
        end

        @testset "positive_stepsize_to_bound_trust_region" begin
            @test @inferred(TRF.positive_stepsize_to_bound_trust_region(x_small, s, trust_radius)) isa T
        end

        @testset "stepsizes_to_bound_trust_region" begin
            result = @inferred TRF.stepsizes_to_bound_trust_region(x_small, s, trust_radius)
            @test result[1] isa T
            @test result[2] isa T
        end

        @testset "adjust_trust_radius" begin
            @test @inferred(TRF.adjust_trust_radius(T(0.5), step, Δ)) isa T
        end

        @testset "coleman_li_scaling_factors" begin
            result = @inferred TRF.coleman_li_scaling_factors(x_inside, g, LB, UB)
            @test eltype(result[1]) == T
            @test eltype(result[2]) == T
        end

        @testset "evaluate_quadratic" begin
            val = @inferred TRF.evaluate_quadratic(H, g, s)
            @test val isa T
        end

        @testset "build_quadratic_1d" begin
            result = @inferred TRF.build_quadratic_1d(H, g, s, x)
            @test result[1] isa T
            @test result[2] isa T
            @test result[3] isa T
        end

        @testset "minimize_quadratic_1d" begin
            result = @inferred TRF.minimize_quadratic_1d(a, b, lb, ub, c)
            @test result[1] isa T
            @test result[2] isa T
        end

        @testset "steihaug_store_steps" begin
            P = identity
            z0 = zero(g)
            result = @inferred TRF.steihaug_store_steps(H, g, trust_radius, P, 10, T(1e-6), z0)
            @test eltype(result) == Vector{T}
        end

        @testset "compute_newton_step" begin
            # Use a step that stays within bounds
            gn     = T(0.01) .* ones(T, n)
            gn_hat = T(0.01) .* ones(T, n)
            g_hat  = rand(T, n)
            result = @inferred TRF.compute_newton_step(x_inside, gn, gn_hat, g_hat, H, LB, UB)
            @test result[1] isa Bool
            @test result[4] isa T
        end

        @testset "compute_steepest_descent_step" begin
            g_hat = rand(T, n)
            D     = ones(T, n)
            theta = T(0.995)
            result = @inferred TRF.compute_steepest_descent_step(x_inside, g_hat, H, D, trust_radius, theta, LB, UB)
            @test eltype(result[1]) == T  # step
            @test eltype(result[2]) == T  # step_hat
            @test result[3] isa T         # step_value
        end

        @testset "compute_interior_newton_step" begin
            gn     = rand(T, n)
            gn_hat = rand(T, n)
            g_hat  = rand(T, n)
            D      = ones(T, n)
            theta  = T(0.995)
            result = @inferred TRF.compute_interior_newton_step(x_inside, gn, gn_hat, g_hat, H, D, theta, LB, UB)
            @test eltype(result[1]) == T
            @test eltype(result[2]) == T
            @test result[3] isa T
        end

        @testset "compute_reflected_step" begin
            # Make a step that goes out of bounds to trigger reflection
            # Keep it small enough that boundary_step_hat stays within trust region
            gn     = (UB .- x_inside) .* T(1.1)
            gn_hat = gn
            g_hat  = rand(T, n)
            D      = ones(T, n)
            theta  = T(0.995)
            # Use a large enough trust radius to contain boundary_step_hat
            tr_large = T(2) * norm(gn)
            result = @inferred TRF.compute_reflected_step(x_inside, gn, gn_hat, g_hat, H, D, tr_large, theta, LB, UB)
            @test eltype(result[1]) == T
            @test eltype(result[2]) == T
            @test result[3] isa T
        end

        @testset "choose_step" begin
            gn     = rand(T, n) .* T(0.01)
            gn_hat = gn
            g_hat  = rand(T, n)
            D      = ones(T, n)
            theta  = T(0.995)
            result = @inferred TRF.choose_step(x_inside, H, g_hat, gn, gn_hat, D, trust_radius, theta, LB, UB)
            @test eltype(result[1]) == T
            @test eltype(result[2]) == T
            @test result[3] isa T
        end
    end

    # CuArray tests (if CUDA is available)
    if CUDA.functional()
        @testset "T = CuArray{Float64}" begin
            T = Float64
            n_cu = 5
            x  = CuArray(rand(T, n_cu))
            s  = CuArray(rand(T, n_cu))
            LB = CUDA.zeros(T, n_cu)
            UB = CUDA.ones(T, n_cu) .* T(10)
            g  = CuArray(rand(T, n_cu))
            step = CuArray(rand(T, n_cu) .* T(0.5))
            Δ   = T(2)
            trust_radius = T(2)

            H = LinearMap(x -> T(2) .* x, n_cu, n_cu)

            x_inside = clamp.(x, LB .+ T(0.01), UB .- T(0.01))
            x_small  = x .* (trust_radius / (T(2) * norm(x)))

            @testset "all_within_bounds" begin
                @test @inferred(TRF.all_within_bounds(x_inside, LB, UB)) isa Bool
            end

            @testset "stepsize_to_bound_feasible_region" begin
                result = @inferred TRF.stepsize_to_bound_feasible_region(x_inside, s, LB, UB)
                @test result[1] isa T
            end

            @testset "coleman_li_scaling_factors" begin
                result = @inferred TRF.coleman_li_scaling_factors(x_inside, g, LB, UB)
                @test eltype(result[1]) == T
            end

            @testset "evaluate_quadratic" begin
                @test @inferred(TRF.evaluate_quadratic(H, g, s)) isa T
            end

            @testset "adjust_trust_radius" begin
                @test @inferred(TRF.adjust_trust_radius(T(0.5), step, Δ)) isa T
            end

            @testset "steihaug_store_steps" begin
                z0 = CUDA.zeros(T, n_cu)
                result = @inferred TRF.steihaug_store_steps(H, g, trust_radius, identity, 10, T(1e-6), z0)
                @test first(result) isa CuArray{T}
            end
        end
    else
        @info "CUDA not available, skipping CuArray type stability tests."
    end
end
