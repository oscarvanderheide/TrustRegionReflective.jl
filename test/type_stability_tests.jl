using TrustRegionReflective
using Test
using CUDA
using LinearAlgebra
using InteractiveUtils  # For @code_warntype

# Define a function to check if a type is preserved
function check_type_stability(f, args...; name="Function", expected_type=nothing)
    println("Testing type stability for: $name")
    result = f(args...)

    # If result is a tuple, check each element
    if result isa Tuple
        for (i, r) in enumerate(result)
            # Extract the base element type from the first argument
            if args[1] isa AbstractArray
                input_type = eltype(args[1])
                if expected_type === nothing
                    expected = input_type
                else
                    expected = expected_type
                end

                # Handle different result types
                if r isa AbstractArray
                    actual = eltype(r)
                elseif r isa Number
                    actual = typeof(r)
                elseif r isa Bool
                    actual = Bool
                else
                    actual = typeof(r)
                end
            else
                # Handle scalar arguments
                input_type = typeof(args[1])
                if expected_type === nothing
                    expected = input_type
                else
                    expected = expected_type
                end
                actual = typeof(r)
            end

            is_stable = (expected == actual)
            stability_status = is_stable ? "✅" : "❌"
            println("  Result[$i]: $stability_status Expected: $expected, Got: $actual")
            @test is_stable
        end
    else
        # Extract the base element type from the first argument
        if args[1] isa AbstractArray
            input_type = eltype(args[1])
            if expected_type === nothing
                expected = input_type
            else
                expected = expected_type
            end

            # Handle different result types
            if result isa AbstractArray
                actual = eltype(result)
            elseif result isa Number
                actual = typeof(result)
            elseif result isa Bool
                actual = Bool
            else
                actual = typeof(result)
            end
        else
            # Handle scalar arguments
            input_type = typeof(args[1])
            if expected_type === nothing
                expected = input_type
            else
                expected = expected_type
            end
            actual = typeof(result)
        end

        is_stable = (expected == actual)
        stability_status = is_stable ? "✅" : "❌"
        println("  Result: $stability_status Expected: $expected, Got: $actual")
        @test is_stable
    end

    # For CuArray inputs, check if result is also on GPU
    if args[1] isa CuArray
        if result isa Tuple
            for (i, r) in enumerate(result)
                is_gpu = r isa CuArray
                gpu_status = is_gpu ? "✅" : "❌"
                println("  Result[$i] on GPU: $gpu_status")
                @test is_gpu
            end
        else
            is_gpu = result isa CuArray
            gpu_status = is_gpu ? "✅" : "❌"
            println("  Result on GPU: $gpu_status")
            @test is_gpu
        end
    end

    return result
end

# Function to run @code_warntype on a function with given arguments
function check_code_warntype(f, args...; name="Function")
    println("\n==== @code_warntype for $name ====")
    @code_warntype f(args...)
    println("=======================================\n")
end

@testset "Type Stability Tests" begin
    # Create test data with different precision
    n = 5

    # Float64 test vectors
    x64 = rand(n)
    s64 = rand(n)
    LB64 = zeros(n)
    UB64 = ones(n) * 10
    g64 = rand(n)
    step64 = rand(n)
    current_ratio64 = 0.5
    Δ64 = 2.0
    min_ratio64 = 0.25
    a64, b64, c64 = 1.0, -2.0, 1.0
    lb64, ub64 = 0.0, 1.0
    trust_radius64 = 2.0

    # Float32 test vectors
    x32 = rand(Float32, n)
    s32 = rand(Float32, n)
    LB32 = zeros(Float32, n)
    UB32 = ones(Float32, n) * 10
    g32 = rand(Float32, n)
    step32 = rand(Float32, n)
    current_ratio32 = 0.5f0
    Δ32 = 2.0f0
    min_ratio32 = 0.25f0
    a32, b32, c32 = 1.0f0, -2.0f0, 1.0f0
    lb32, ub32 = 0.0f0, 1.0f0
    trust_radius32 = 2.0f0

    # CuArray test vectors (if CUDA is available)
    if CUDA.functional()
        x_cu = CuArray(x64)
        s_cu = CuArray(s64)
        LB_cu = CuArray(LB64)
        UB_cu = CuArray(UB64)
        g_cu = CuArray(g64)
        step_cu = CuArray(step64)
        current_ratio_cu = 0.5
        Δ_cu = 2.0
        min_ratio_cu = 0.25
        trust_radius_cu = 2.0
    end

    # Helper functions for tests
    H64 = x -> I * x  # Identity Hessian
    H32 = x -> I * x
    H_cu = x -> CUDA.CUBLAS.gemv('N', CuArray(Matrix{Float64}(I, n, n)), x)

    @testset "all_within_bounds" begin
        println("\nTesting all_within_bounds:")
        # Float64 test
        result64 = check_type_stability(TrustRegionReflective.all_within_bounds, x64, LB64, UB64, name="all_within_bounds (Float64)", expected_type=Bool)
        check_code_warntype(TrustRegionReflective.all_within_bounds, x64, LB64, UB64, name="all_within_bounds (Float64)")

        # Float32 test
        result32 = check_type_stability(TrustRegionReflective.all_within_bounds, x32, LB32, UB32, name="all_within_bounds (Float32)", expected_type=Bool)
        check_code_warntype(TrustRegionReflective.all_within_bounds, x32, LB32, UB32, name="all_within_bounds (Float32)")

        # CuArray test
        if CUDA.functional()
            result_cu = check_type_stability(TrustRegionReflective.all_within_bounds, x_cu, LB_cu, UB_cu, name="all_within_bounds (CuArray)", expected_type=Bool)
            check_code_warntype(TrustRegionReflective.all_within_bounds, x_cu, LB_cu, UB_cu, name="all_within_bounds (CuArray)")
        end
    end

    @testset "stepsize_to_bound_feasible_region" begin
        println("\nTesting stepsize_to_bound_feasible_region:")
        # Float64 test
        result64 = check_type_stability(TrustRegionReflective.stepsize_to_bound_feasible_region, x64, s64, LB64, UB64, name="stepsize_to_bound_feasible_region (Float64)")
        check_code_warntype(TrustRegionReflective.stepsize_to_bound_feasible_region, x64, s64, LB64, UB64, name="stepsize_to_bound_feasible_region (Float64)")

        # Float32 test
        result32 = check_type_stability(TrustRegionReflective.stepsize_to_bound_feasible_region, x32, s32, LB32, UB32, name="stepsize_to_bound_feasible_region (Float32)")
        check_code_warntype(TrustRegionReflective.stepsize_to_bound_feasible_region, x32, s32, LB32, UB32, name="stepsize_to_bound_feasible_region (Float32)")

        # CuArray test
        if CUDA.functional()
            result_cu = check_type_stability(TrustRegionReflective.stepsize_to_bound_feasible_region, x_cu, s_cu, LB_cu, UB_cu, name="stepsize_to_bound_feasible_region (CuArray)")
            check_code_warntype(TrustRegionReflective.stepsize_to_bound_feasible_region, x_cu, s_cu, LB_cu, UB_cu, name="stepsize_to_bound_feasible_region (CuArray)")
        end
    end

    @testset "positive_stepsize_to_bound_trust_region" begin
        println("\nTesting positive_stepsize_to_bound_trust_region:")
        # Make sure x is within trust region
        x64_small = x64 .* (trust_radius64 / (2.0 * norm(x64)))
        x32_small = x32 .* (trust_radius32 / (2.0f0 * norm(x32)))

        # Float64 test
        result64 = check_type_stability(TrustRegionReflective.positive_stepsize_to_bound_trust_region, x64_small, s64, trust_radius64, name="positive_stepsize_to_bound_trust_region (Float64)")
        check_code_warntype(TrustRegionReflective.positive_stepsize_to_bound_trust_region, x64_small, s64, trust_radius64, name="positive_stepsize_to_bound_trust_region (Float64)")

        # Float32 test
        result32 = check_type_stability(TrustRegionReflective.positive_stepsize_to_bound_trust_region, x32_small, s32, trust_radius32, name="positive_stepsize_to_bound_trust_region (Float32)")
        check_code_warntype(TrustRegionReflective.positive_stepsize_to_bound_trust_region, x32_small, s32, trust_radius32, name="positive_stepsize_to_bound_trust_region (Float32)")

        # CuArray test
        if CUDA.functional()
            x_cu_small = x_cu .* (trust_radius_cu / (2.0 * norm(x_cu)))
            result_cu = check_type_stability(TrustRegionReflective.positive_stepsize_to_bound_trust_region, x_cu_small, s_cu, trust_radius_cu, name="positive_stepsize_to_bound_trust_region (CuArray)")
            check_code_warntype(TrustRegionReflective.positive_stepsize_to_bound_trust_region, x_cu_small, s_cu, trust_radius_cu, name="positive_stepsize_to_bound_trust_region (CuArray)")
        end
    end

    @testset "stepsizes_to_bound_trust_region" begin
        println("\nTesting stepsizes_to_bound_trust_region:")
        # Make sure x is within trust region
        x64_small = x64 .* (trust_radius64 / (2.0 * norm(x64)))
        x32_small = x32 .* (trust_radius32 / (2.0f0 * norm(x32)))

        # Float64 test
        result64 = check_type_stability(TrustRegionReflective.stepsizes_to_bound_trust_region, x64_small, s64, trust_radius64, name="stepsizes_to_bound_trust_region (Float64)")
        check_code_warntype(TrustRegionReflective.stepsizes_to_bound_trust_region, x64_small, s64, trust_radius64, name="stepsizes_to_bound_trust_region (Float64)")

        # Float32 test
        result32 = check_type_stability(TrustRegionReflective.stepsizes_to_bound_trust_region, x32_small, s32, trust_radius32, name="stepsizes_to_bound_trust_region (Float32)")
        check_code_warntype(TrustRegionReflective.stepsizes_to_bound_trust_region, x32_small, s32, trust_radius32, name="stepsizes_to_bound_trust_region (Float32)")

        # CuArray test
        if CUDA.functional()
            x_cu_small = x_cu .* (trust_radius_cu / (2.0 * norm(x_cu)))
            result_cu = check_type_stability(TrustRegionReflective.stepsizes_to_bound_trust_region, x_cu_small, s_cu, trust_radius_cu, name="stepsizes_to_bound_trust_region (CuArray)")
            check_code_warntype(TrustRegionReflective.stepsizes_to_bound_trust_region, x_cu_small, s_cu, trust_radius_cu, name="stepsizes_to_bound_trust_region (CuArray)")
        end
    end

    @testset "adjust_trust_radius" begin
        println("\nTesting adjust_trust_radius:")
        # Float64 test
        result64 = check_type_stability(TrustRegionReflective.adjust_trust_radius, current_ratio64, step64, Δ64, name="adjust_trust_radius (Float64)")
        check_code_warntype(TrustRegionReflective.adjust_trust_radius, current_ratio64, step64, Δ64, name="adjust_trust_radius (Float64)")

        # Float32 test
        result32 = check_type_stability(TrustRegionReflective.adjust_trust_radius, current_ratio32, step32, Δ32, name="adjust_trust_radius (Float32)")
        check_code_warntype(TrustRegionReflective.adjust_trust_radius, current_ratio32, step32, Δ32, name="adjust_trust_radius (Float32)")

        # CuArray test
        if CUDA.functional()
            # Note: current_ratio and Δ are scalars, only step is a CuArray
            result_cu = check_type_stability(TrustRegionReflective.adjust_trust_radius, current_ratio_cu, step_cu, Δ_cu, name="adjust_trust_radius (CuArray)")
            check_code_warntype(TrustRegionReflective.adjust_trust_radius, current_ratio_cu, step_cu, Δ_cu, name="adjust_trust_radius (CuArray)")
        end
    end

    @testset "coleman_li_scaling_factors" begin
        println("\nTesting coleman_li_scaling_factors:")
        # Float64 test
        result64 = check_type_stability(TrustRegionReflective.coleman_li_scaling_factors, x64, g64, LB64, UB64, name="coleman_li_scaling_factors (Float64)")
        check_code_warntype(TrustRegionReflective.coleman_li_scaling_factors, x64, g64, LB64, UB64, name="coleman_li_scaling_factors (Float64)")

        # Float32 test
        result32 = check_type_stability(TrustRegionReflective.coleman_li_scaling_factors, x32, g32, LB32, UB32, name="coleman_li_scaling_factors (Float32)")
        check_code_warntype(TrustRegionReflective.coleman_li_scaling_factors, x32, g32, LB32, UB32, name="coleman_li_scaling_factors (Float32)")

        # CuArray test
        if CUDA.functional()
            result_cu = check_type_stability(TrustRegionReflective.coleman_li_scaling_factors, x_cu, g_cu, LB_cu, UB_cu, name="coleman_li_scaling_factors (CuArray)")
            check_code_warntype(TrustRegionReflective.coleman_li_scaling_factors, x_cu, g_cu, LB_cu, UB_cu, name="coleman_li_scaling_factors (CuArray)")
        end
    end

    @testset "evaluate_quadratic" begin
        println("\nTesting evaluate_quadratic:")
        # Float64 test
        result64 = check_type_stability(TrustRegionReflective.evaluate_quadratic, H64, g64, s64, name="evaluate_quadratic (Float64)")
        check_code_warntype(TrustRegionReflective.evaluate_quadratic, H64, g64, s64, name="evaluate_quadratic (Float64)")

        # Float32 test
        result32 = check_type_stability(TrustRegionReflective.evaluate_quadratic, H32, g32, s32, name="evaluate_quadratic (Float32)")
        check_code_warntype(TrustRegionReflective.evaluate_quadratic, H32, g32, s32, name="evaluate_quadratic (Float32)")

        # CuArray test
        if CUDA.functional()
            result_cu = check_type_stability(TrustRegionReflective.evaluate_quadratic, H_cu, g_cu, s_cu, name="evaluate_quadratic (CuArray)")
            check_code_warntype(TrustRegionReflective.evaluate_quadratic, H_cu, g_cu, s_cu, name="evaluate_quadratic (CuArray)")
        end
    end

    @testset "build_quadratic_1d" begin
        println("\nTesting build_quadratic_1d:")
        # Float64 test
        result64 = check_type_stability(TrustRegionReflective.build_quadratic_1d, H64, g64, s64, x64, name="build_quadratic_1d (Float64)")
        check_code_warntype(TrustRegionReflective.build_quadratic_1d, H64, g64, s64, x64, name="build_quadratic_1d (Float64)")

        # Float32 test
        result32 = check_type_stability(TrustRegionReflective.build_quadratic_1d, H32, g32, s32, x32, name="build_quadratic_1d (Float32)")
        check_code_warntype(TrustRegionReflective.build_quadratic_1d, H32, g32, s32, x32, name="build_quadratic_1d (Float32)")

        # CuArray test
        if CUDA.functional()
            result_cu = check_type_stability(TrustRegionReflective.build_quadratic_1d, H_cu, g_cu, s_cu, x_cu, name="build_quadratic_1d (CuArray)")
            check_code_warntype(TrustRegionReflective.build_quadratic_1d, H_cu, g_cu, s_cu, x_cu, name="build_quadratic_1d (CuArray)")
        end
    end

    @testset "minimize_quadratic_1d" begin
        println("\nTesting minimize_quadratic_1d:")
        # Float64 test
        result64 = check_type_stability(TrustRegionReflective.minimize_quadratic_1d, a64, b64, lb64, ub64, c64, name="minimize_quadratic_1d (Float64)")
        check_code_warntype(TrustRegionReflective.minimize_quadratic_1d, a64, b64, lb64, ub64, c64, name="minimize_quadratic_1d (Float64)")

        # Float32 test
        result32 = check_type_stability(TrustRegionReflective.minimize_quadratic_1d, a32, b32, lb32, ub32, c32, name="minimize_quadratic_1d (Float32)")
        check_code_warntype(TrustRegionReflective.minimize_quadratic_1d, a32, b32, lb32, ub32, c32, name="minimize_quadratic_1d (Float32)")
    end

    # Add tests for steihaug.jl functions
    @testset "steihaug_store_steps" begin
        println("\nTesting steihaug_store_steps:")

        # Float64 test
        tol64 = 1e-8
        maxit64 = 10
        P64 = identity
        z0_64 = zero(g64)

        # Float64 test
        result64 = check_type_stability(TrustRegionReflective.steihaug_store_steps, H64, g64, trust_radius64, P64, maxit64, tol64, z0_64, name="steihaug_store_steps (Float64)")
        check_code_warntype(TrustRegionReflective.steihaug_store_steps, H64, g64, trust_radius64, P64, maxit64, tol64, z0_64, name="steihaug_store_steps (Float64)")

        # Float32 test
        tol32 = 1e-8f0
        maxit32 = 10
        P32 = identity
        z0_32 = zero(g32)
        result32 = check_type_stability(TrustRegionReflective.steihaug_store_steps, H32, g32, trust_radius32, P32, maxit32, tol32, z0_32, name="steihaug_store_steps (Float32)")
        check_code_warntype(TrustRegionReflective.steihaug_store_steps, H32, g32, trust_radius32, P32, maxit32, tol32, z0_32, name="steihaug_store_steps (Float32)")

        # CuArray test
        if CUDA.functional()
            P_cu = identity
            z0_cu = zero(g_cu)
            result_cu = check_type_stability(TrustRegionReflective.steihaug_store_steps, H_cu, g_cu, trust_radius_cu, P_cu, maxit64, tol64, z0_cu, name="steihaug_store_steps (CuArray)")
            check_code_warntype(TrustRegionReflective.steihaug_store_steps, H_cu, g_cu, trust_radius_cu, P_cu, maxit64, tol64, z0_cu, name="steihaug_store_steps (CuArray)")
        end
    end

    # Test main solver function
    @testset "trf" begin
        println("\nTesting trf solver:")

        # Define a simple objective function for testing
        function test_objective_f64(x, mode)
            f = sum((x .- 1.0).^2)
            r = x .- 1.0
            g = 2.0 .* r
            H = x -> 2.0 .* x

            if mode == :f
                return f
            elseif mode == :fr
                return f, r
            elseif mode == :frg
                return f, r, g
            else # mode == :frgH
                return f, r, g, H
            end
        end

        function test_objective_f32(x, mode)
            f = sum((x .- 1.0f0).^2)
            r = x .- 1.0f0
            g = 2.0f0 .* r
            H = x -> 2.0f0 .* x

            if mode == :f
                return f
            elseif mode == :fr
                return f, r
            elseif mode == :frg
                return f, r, g
            else # mode == :frgH
                return f, r, g, H
            end
        end

        function test_objective_cu(x, mode)
            f = sum((x .- 1.0).^2)
            r = x .- 1.0
            g = 2.0 .* r
            H = x -> 2.0 .* x

            if mode == :f
                return f
            elseif mode == :fr
                return f, r
            elseif mode == :frg
                return f, r, g
            else # mode == :frgH
                return f, r, g, H
            end
        end

        # Define TRF options
        options64 = TrustRegionReflective.TRFOptions(0.1, 20, 10, 1e-8, 0.1, false)
        options32 = TrustRegionReflective.TRFOptions(0.1f0, 20, 10, 1e-8f0, 0.1f0, false)

        # Define a dummy callback
        callback = (iter, state) -> nothing

        # Float64 test - just check @code_warntype as the actual solver might be slow
        x0_64 = zeros(2)
        check_code_warntype(TrustRegionReflective.trf, test_objective_f64, x0_64, LB64[1:2], UB64[1:2], options64, callback, name="trf (Float64)")

        # Float32 test
        x0_32 = zeros(Float32, 2)
        check_code_warntype(TrustRegionReflective.trf, test_objective_f32, x0_32, LB32[1:2], UB32[1:2], options32, callback, name="trf (Float32)")

        # CuArray test
        if CUDA.functional()
            x0_cu = CuArray(zeros(2))
            check_code_warntype(TrustRegionReflective.trf, test_objective_cu, x0_cu, CuArray(LB64[1:2]), CuArray(UB64[1:2]), options64, callback, name="trf (CuArray)")
        end
    end

    # Add tests for any dogleg.jl functions if they exist
    if hasmethod(TrustRegionReflective.dogleg_step, Tuple{Any, Any, Any})
        @testset "dogleg_step" begin
            println("\nTesting dogleg_step:")

            # Float64 test
            result64 = check_type_stability(TrustRegionReflective.dogleg_step, g64, H64, trust_radius64, name="dogleg_step (Float64)")
            check_code_warntype(TrustRegionReflective.dogleg_step, g64, H64, trust_radius64, name="dogleg_step (Float64)")

            # Float32 test
            result32 = check_type_stability(TrustRegionReflective.dogleg_step, g32, H32, trust_radius32, name="dogleg_step (Float32)")
            check_code_warntype(TrustRegionReflective.dogleg_step, g32, H32, trust_radius32, name="dogleg_step (Float32)")

            # CuArray test
            if CUDA.functional()
                result_cu = check_type_stability(TrustRegionReflective.dogleg_step, g_cu, H_cu, trust_radius_cu, name="dogleg_step (CuArray)")
                check_code_warntype(TrustRegionReflective.dogleg_step, g_cu, H_cu, trust_radius_cu, name="dogleg_step (CuArray)")
            end
        end
    end
end
