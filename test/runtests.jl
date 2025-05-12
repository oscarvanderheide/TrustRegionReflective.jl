using TrustRegionReflective
using Test
using LinearAlgebra
using CUDA

@testset "stepsize_to_bound_feasible_region tests" begin

    using TrustRegionReflective: stepsize_to_bound_feasible_region

    # Test case 1: x + stepsize * s is on the lower bound
    x = [1.0, 1.0, 1.0]
    s = [-1.0, -2.0, -3.0]
    LB = [0.0, 0.0, 0.0]
    UB = [5.0, 5.0, 5.0]
    expected_stepsize = 1 / 3
    expected_boundary_hit = Bool[0, 0, 1]
    @test stepsize_to_bound_feasible_region(x, s, LB, UB) == (expected_stepsize, expected_boundary_hit)

    # Test case 2: x + stepsize * s is on the upper bound
    x = [1.0, 1.0, 1.0]
    s = [1.0, 0.0, 2.0]
    LB = [0.0, 0.0, 0.0]
    UB = [2.0, 2.0, 2.0]
    expected_stepsize = 1 / 2
    expected_boundary_hit = [0, 0, 1]
    @test stepsize_to_bound_feasible_region(x, s, LB, UB) == (expected_stepsize, expected_boundary_hit)

    # Test case 3: x + stepsize * s is on both lower and upper bounds
    x = [1.0, 1.0, 1.0]
    s = [1.0, -1.0, 0.5]
    LB = [0.0, 0.0, 0.0]
    UB = [2.0, 2.0, 2.0]
    expected_stepsize = 1.0
    expected_boundary_hit = [1, 1, 0]
    @test stepsize_to_bound_feasible_region(x, s, LB, UB) == (expected_stepsize, expected_boundary_hit)

    # Test case 4: Float32 inputs and outputs
    x = Float32[1.0, 1.0, 1.0]
    s = Float32[1.0, 1.0, 1.0]
    LB = Float32[0.0, 0.0, 0.0]
    UB = Float32[5.0, 5.0, 5.0]
    expected_stepsize = 4.0f0
    expected_boundary_hit = Bool[1, 1, 1]
    @test stepsize_to_bound_feasible_region(x, s, LB, UB) == (expected_stepsize, expected_boundary_hit)

    # Test case 5: CuArray inputs
    if CUDA.has_cuda_gpu()
        x = CuArray([1.0, 1.0, 1.0])
        s = CuArray([1.0, 1.0, 1.0])
        LB = CuArray([0.0, 0.0, 0.0])
        UB = CuArray([5.0, 5.0, 5.0])
        expected_stepsize = 4.0f0
        expected_boundary_hit = CuArray(Bool[1, 1, 1])
        @test stepsize_to_bound_feasible_region(x, s, LB, UB) == (expected_stepsize, expected_boundary_hit)
    else
        @info "CUDA not available, skipping CuArray tests."
    end

end

@testset "positive_stepsize_to_bound_trust_region tests" begin

    using TrustRegionReflective: positive_stepsize_to_bound_trust_region
    using LinearAlgebra

    # Test 1: basic check that the norm of x + τ * p is equal to the trust radius
    x = [1.0, 2.0, 3.0]
    p = [4.0, 5.0, 6.0]
    trust_radius = 10.0
    τ = positive_stepsize_to_bound_trust_region(x, p, trust_radius)
    @test norm(x + τ * p) ≈ trust_radius

    # Test 2: Check that τ is as expected 
    x = [0.0, 0.0, 0.0]
    p = [1.0, 0.0, 0.0]
    trust_radius = 1.0
    τ = positive_stepsize_to_bound_trust_region(x, p, trust_radius)
    @test τ == 1.0

    # Error when trust_radius <= 0
    x = [1.0, 0.0, 0.0]
    p = [2.0, 0.0, 0.0]
    trust_radius = 0.0
    @test_throws ErrorException τ == positive_stepsize_to_bound_trust_region(x, p, trust_radius)
    trust_radius = -10.0
    @test_throws ErrorException positive_stepsize_to_bound_trust_region(x, p, trust_radius)

end

@testset "adjust_trust_radius tests" begin

    using TrustRegionReflective: adjust_trust_radius
    using LinearAlgebra

    # Test case 1: current_ratio < min_ratio
    current_ratio = 0.2
    step = [1.0, 2.0, 3.0]
    Δ = norm(step) * 2
    min_ratio = 0.5

    expected_Δ = (1 / 4) * Δ
    actual_Δ = adjust_trust_radius(current_ratio, step, Δ, min_ratio)

    @test expected_Δ ≈ actual_Δ

    # Test case 2: current_ratio > 1/2 and norm(step) > 0.95 * Δ
    current_ratio = 0.8
    step = [1.0, 2.0, 3.0]
    Δ = norm(step) * 1.01
    min_ratio = 0.5

    expected_Δ = 2 * Δ
    actual_Δ = adjust_trust_radius(current_ratio, step, Δ, min_ratio)

    @test expected_Δ ≈ actual_Δ

    # Test case 3: current_ratio > 1/2 but norm(step) <= 0.95 * Δ
    current_ratio = 0.8
    step = [1.0, 1.0, 1.0]
    Δ = 2 * norm(step)
    min_ratio = 0.5

    expected_Δ = Δ
    actual_Δ = adjust_trust_radius(current_ratio, step, Δ, min_ratio)

    @test expected_Δ ≈ actual_Δ

    # Test case 4: min_ratio < current_ratio < 1/2
    current_ratio = 0.4
    step = [1.0, 2.0, 3.0]
    Δ = norm(step) * 1.5
    min_ratio = 0.1

    expected_Δ = Δ
    actual_Δ = adjust_trust_radius(current_ratio, step, Δ, min_ratio)

    @test expected_Δ ≈ actual_Δ

    # Test case 5: norm(step) > Δ errors
    current_ratio = 0.6
    step = [1.0, 2.0, 3.0]
    Δ = 1.0
    min_ratio = 0.5

    expected_Δ = Δ
    @test_throws ErrorException adjust_trust_radius(current_ratio, step, Δ, min_ratio)

    # Test case 6: Float32 - no change
    current_ratio = 0.6f0
    step = Float32[1.0, 2.0, 3.0]
    Δ = 2 * norm(step)
    min_ratio = 0.1f0

    expected_Δ = Δ
    actual_Δ = adjust_trust_radius(current_ratio, step, Δ, min_ratio)

    @test typeof(actual_Δ) == typeof(expected_Δ)
    @test expected_Δ ≈ actual_Δ

    # Test case 7: Float32 - reduce Δ
    current_ratio = 0.05f0
    step = Float32[1.0, 2.0, 3.0]
    Δ = 2 * norm(step)
    min_ratio = 0.1f0

    expected_Δ = Δ / 4
    actual_Δ = adjust_trust_radius(current_ratio, step, Δ, min_ratio)

    @test typeof(actual_Δ) == typeof(expected_Δ)
    @test expected_Δ ≈ actual_Δ

    # Test case 7: Float32 - reduce Δ
    current_ratio = 0.8f0
    step = Float32[1.0, 2.0, 3.0]
    Δ = 1.01f0 * norm(step)
    min_ratio = 0.1f0

    expected_Δ = 2 * Δ
    actual_Δ = adjust_trust_radius(current_ratio, step, Δ, min_ratio)

    @test typeof(actual_Δ) == typeof(expected_Δ)
    @test expected_Δ ≈ actual_Δ

end

@testset "coleman_li_scaling_factors tests" begin

    using TrustRegionReflective: coleman_li_scaling_factors

    # Test case 1: all(g .< 0)
    x1 = [1.0, 2.0, 3.0]
    g1 = [-1.0, -2.0, -3.0]
    LB1 = [0.0, 0.0, 0.0]
    UB1 = [10.0, 10.0, 10.0]
    v1_expected = [9.0, 8.0, 7.0]
    dv1_expected = [-1.0, -1.0, -1.0]
    v1, dv1 = coleman_li_scaling_factors(x1, g1, LB1, UB1)
    @test v1 ≈ v1_expected
    @test dv1 ≈ dv1_expected

    # Test case 2: all(g .< 0), different values
    x2 = [1.0, 2.0, 3.0]
    g2 = [-1.0, -2.0, -3.0]
    LB2 = [0.0, 0.0, 0.0]
    UB2 = [2.0, 10.0, 10.0]
    v2_expected = [1.0, 8.0, 7.0]
    dv2_expected = [-1.0, -1.0, -1.0]
    v2, dv2 = coleman_li_scaling_factors(x2, g2, LB2, UB2)
    @test v2 ≈ v2_expected
    @test dv2 ≈ dv2_expected

    # Test case 3:  all(g .> 0)
    x3 = [1.0, 2.0, 3.0]
    g3 = [1.0, 2.0, 3.0]
    LB3 = [0.0, 0.0, 0.0]
    UB3 = [10.0, 10.0, 10.0]
    v3_expected = [1.0, 2.0, 3.0]
    dv3_expected = [1.0, 1.0, 1.0]
    v3, dv3 = coleman_li_scaling_factors(x3, g3, LB3, UB3)
    @test v3 ≈ v3_expected
    @test dv3 ≈ dv3_expected

    # Test case 4: mixed g
    x4 = [1.0, 2.0, 3.0]
    g4 = [1.0, -2.0, 3.0]
    LB4 = [0.0, 0.0, 0.0]
    UB4 = [2.0, 10.0, 10.0]
    v4_expected = [1.0, 8.0, 3.0]
    dv4_expected = [1.0, -1.0, 1.0]
    v4, dv4 = coleman_li_scaling_factors(x4, g4, LB4, UB4)
    @test v4 ≈ v4_expected
    @test dv4 ≈ dv4_expected

    # Test case 5: All parameters at lower bounds
    x5 = [0.0, 0.0, 0.0]
    g5 = [1.0, 2.0, 3.0]
    LB5 = [0.0, 0.0, 0.0]
    UB5 = [10.0, 10.0, 10.0]
    v5_expected = [0.0, 0.0, 0.0]
    dv5_expected = [1.0, 1.0, 1.0]
    v5, dv5 = coleman_li_scaling_factors(x5, g5, LB5, UB5)
    @test v5 ≈ v5_expected
    @test dv5 ≈ dv5_expected

    # Test case 5: All parameters at upper bounds
    x5 = [10.0, 10.0, 10.0]
    g5 = -[1.0, 2.0, 3.0]
    LB5 = [0.0, 0.0, 0.0]
    UB5 = [10.0, 10.0, 10.0]
    v5_expected = [0.0, 0.0, 0.0]
    dv5_expected = -[1.0, 1.0, 1.0]
    v5, dv5 = coleman_li_scaling_factors(x5, g5, LB5, UB5)
    @test v5 ≈ v5_expected
    @test dv5 ≈ dv5_expected

    # Test case 6: Float32 inputs
    x6 = Float32[1.0, 2.0, 3.0]
    g6 = Float32[1.0, 2.0, 3.0]
    LB6 = Float32[0.0, 0.0, 0.0]
    UB6 = Float32[10.0, 10.0, 10.0]
    v6_expected = Float32[1.0, 2.0, 3.0]
    dv6_expected = Float32[1.0, 1.0, 1.0]

    v6, dv6 = coleman_li_scaling_factors(x6, g6, LB6, UB6)
    @test v6 ≈ v6_expected

    # Test case 7: CuArray inputs
    if CUDA.has_cuda_gpu()
        x7 = CuArray([1.0, 2.0, 3.0])
        g7 = CuArray([1.0, 2.0, 3.0])
        LB7 = CuArray([0.0, 0.0, 0.0])
        UB7 = CuArray([10.0, 10.0, 10.0])
        v7_expected = CuArray([1.0, 2.0, 3.0])
        dv7_expected = CuArray([1.0, 1.0, 1.0])

        v7, dv7 = coleman_li_scaling_factors(x7, g7, LB7, UB7)
        @test v7 ≈ v7_expected
    else
        @info "CUDA not available, skipping CuArray tests."
    end

end

@testset "evaluate_quadratic tests" begin

    using TrustRegionReflective: evaluate_quadratic

    # Test case 1: Hessian matrix is a 2x2 identity matrix, gradient vector is [1, 2], input vector is [3, 4]
    H1 = x -> [1 0; 0 1] * x
    g1 = [1, 2]
    s1 = [3, 4]
    expected_value1 = (1 / 2) * (3^2 + 4^2) + [1, 2]' * [3, 4]
    @test evaluate_quadratic(H1, g1, s1) ≈ expected_value1

    # Test case 2: Hessian matrix is a 3x3 diagonal matrix with diagonal elements [2, 3, 4], gradient vector is [1, 1, 1], input vector is [1, 1, 1]
    H2 = x -> Diagonal([2, 3, 4]) * x
    g2 = [1, 1, 1]
    s2 = [1, 1, 1]
    expected_value2 = (1 / 2) * (2 * 1^2 + 3 * 1^2 + 4 * 1^2) + [1, 1, 1]' * [1, 1, 1]
    @test evaluate_quadratic(H2, g2, s2) ≈ expected_value2

    # Test case 3: Hessian matrix is a 2x2 matrix with all elements equal to 1, gradient vector is [0, 0], input vector is [0, 0]
    H3 = x -> ones(2, 2) * x
    g3 = [0, 0]
    s3 = [0, 0]
    expected_value3 = (1 / 2) * (0^2 + 0^2) + [0, 0]' * [0, 0]
    @test evaluate_quadratic(H3, g3, s3) ≈ expected_value3

    # Test case 4: Hessian matrix is a 2x2 matrix with all elements equal to 0, gradient vector is [1, 2], input vector is [3, 4]
    H4 = x -> zeros(2, 2) * x
    g4 = [1, 2]
    s4 = [3, 4]
    expected_value4 = (1 / 2) * (0^2 + 0^2) + [1, 2]' * [3, 4]
    @test evaluate_quadratic(H4, g4, s4) ≈ expected_value4

    # Test case 5: Hessian matrix is a 3x3 matrix with random elements, gradient vector is [1, 2, 3], input vector is [4, 5, 6]
    H = rand(3, 3)
    H5 = x -> H * x
    g5 = rand(3)
    s5 = rand(3)
    expected_value5 = (1 / 2) * (s5' * H5(s5)) + g5' * s5
    @test evaluate_quadratic(H5, g5, s5) ≈ expected_value5

    # Test case 6: Float32 inputs
    H6 = x -> ones(Float32, 2, 2) * x
    g6 = Float32.([1, 2])
    s6 = Float32.([3, 4])
    expected_value6 = Float32((1 / 2) * (3 * 7 + 4 * 7) + [1, 2]' * [3, 4])
    @test typeof(evaluate_quadratic(H6, g6, s6)) == typeof(expected_value6)
    @test evaluate_quadratic(H6, g6, s6) ≈ expected_value6

    # Test case 7: CuArray inputs
    if CUDA.has_cuda_gpu()
        H7 = x -> CUDA.ones(2, 2) * x
        g7 = CuArray([1, 2])
        s7 = CuArray([3, 4])
        expected_value7 = (1 / 2) * (3 * 7 + 4 * 7) + [1, 2]' * [3, 4]
        @test evaluate_quadratic(H7, g7, s7) ≈ expected_value7
    else
        @info "CUDA not available, skipping CuArray tests."
    end
end

@testset "all_within_bounds tests" begin

    using TrustRegionReflective: all_within_bounds

    # Test case 1: All elements within bounds
    x1 = [1, 2, 3]
    LB1 = [0, 0, 0]
    UB1 = [10, 10, 10]
    @test all_within_bounds(x1, LB1, UB1) == true

    # Test case 2: Some elements outside bounds
    x2 = [1, 2, 3]
    LB2 = [0, 0, 0]
    UB2 = [2, 2, 2]
    @test all_within_bounds(x2, LB2, UB2) == false

    # Test case 3: All elements outside bounds
    x3 = [1, 2, 3]
    LB3 = [10, 10, 10]
    UB3 = [20, 20, 20]
    @test all_within_bounds(x3, LB3, UB3) == false

    # Test case 4: Float32 inputs
    x4 = Float32[1, 2, 3]
    LB4 = Float32[0, 0, 0]
    UB4 = Float32[10, 10, 10]
    @test all_within_bounds(x4, LB4, UB4) == true

    # Test case 5: CuArray inputs
    if CUDA.has_cuda_gpu()
        x5 = CuArray([1, 2, 3])
        LB5 = CuArray([0, 0, 0])
        UB5 = CuArray([10, 10, 10])
        @test all_within_bounds(x5, LB5, UB5) == true
    else
        @info "CUDA not available, skipping CuArray tests."
    end

    # Test case 6: x wrong size
    x6 = [1, 2, 3, 4]
    LB6 = [0, 0, 0]
    UB6 = [10, 10, 10]
    @test_throws ErrorException all_within_bounds(x6, LB6, UB6)

end

@testset "minimize_quadratic_1d tests" begin

    using TrustRegionReflective: minimize_quadratic_1d

    # Test case 1: Minimum at the boundary
    a = 1.0
    b = -2.0
    lb = 0.0
    ub = 1.0
    c = 0.0
    argument, minval = minimize_quadratic_1d(a, b, lb, ub, c)
    @test argument == 1.0
    @test minval == -1.0

    # Test case 2: Minimum within the bounds
    a = 2.0
    b = 0.0
    lb = -2.0
    ub = 2.0
    c = 0.0
    argument, minval = minimize_quadratic_1d(a, b, lb, ub, c)
    @test argument == 0.0
    @test minval == 0.0

    # Test case 3: Minimum outside the bounds
    a = 0.0
    b = -2.0
    lb = 0.0
    ub = 1.0
    c = 1.0
    argument, minval = minimize_quadratic_1d(a, b, lb, ub, c)
    @test argument == 1.0
    @test minval == -1.0

    # Test case 4: Float32 inputs
    a = 1.0f0
    b = -2.0f0
    lb = 0.0f0
    ub = 1.0f0
    c = 0.0f0
    argument, minval = minimize_quadratic_1d(a, b, lb, ub, c)
    @test argument == 1.0f0
    @test minval == -1.0f0

end

@testset "stepsizes_to_bound_trust_region tests" begin

    using TrustRegionReflective: stepsizes_to_bound_trust_region
    using TrustRegionReflective: positive_stepsize_to_bound_trust_region

    using CUDA

    # Test case 1: s is zero
    x = [1.0, 2.0, 3.0]
    s = [0.0, 0.0, 0.0]
    trust_radius = 1.0

    @test_throws ErrorException stepsizes_to_bound_trust_region(x, s, trust_radius)

    # Test case 2: x not within trust region
    x = [1.0, 2.0, 3.0]
    s = [1.0, 1.0, 1.0]
    trust_radius = 1.0

    @test_throws ErrorException stepsizes_to_bound_trust_region(x, s, trust_radius)

    # Test case 3: t⁻ < t⁺
    x = [1.0, 2.0, 3.0]
    s = [1.0, 1.0, 1.0]
    trust_radius = 10.0

    t⁻, t⁺ = stepsizes_to_bound_trust_region(x, s, trust_radius)

    @test t⁻ < t⁺
    @test t⁺ ≈ positive_stepsize_to_bound_trust_region(x, s, trust_radius)

    # Test case 4: t⁻ < t⁺
    x = [1.0, 2.0, 3.0]
    s = [-1.0, -1.0, -1.0]
    trust_radius = 10.0

    t⁻, t⁺ = stepsizes_to_bound_trust_region(x, s, trust_radius)
    @test t⁺ ≈ positive_stepsize_to_bound_trust_region(x, s, trust_radius)

    # Test case 5: Float32 inputs
    x = Float32[1.0, 2.0, 3.0]
    s = Float32[1.0, 1.0, 1.0]
    trust_radius = 10.0f0

    t⁻, t⁺ = stepsizes_to_bound_trust_region(x, s, trust_radius)

    @test typeof(t⁻) == typeof(t⁺) == Float32
    @test t⁺ ≈ positive_stepsize_to_bound_trust_region(x, s, trust_radius)

    # Test case 6: CuArray inputs
    if CUDA.has_cuda_gpu()
        x = CuArray([1.0, 2.0, 3.0])
        s = CuArray([1.0, 1.0, 1.0])
        trust_radius = 10.0

        t⁻, t⁺ = stepsizes_to_bound_trust_region(x, s, trust_radius)

        @test typeof(t⁻) == typeof(t⁺) == typeof(trust_radius)
        @test t⁻ < t⁺
        @test t⁺ ≈ positive_stepsize_to_bound_trust_region(x, s, trust_radius)
    else
        @info "CUDA not available, skipping CuArray tests."
    end
end

@testset "compute_newton_step tests" begin
    using TrustRegionReflective: compute_newton_step
    
    # Test helper function to create a simple Hessian operator
    function make_hessian(matrix)
        return x -> matrix * x
    end
    
    # Test case 1: Newton step is feasible
    x = [1.0, 1.0]
    gn = [0.5, 0.5]
    gn_hat = [0.5, 0.5]
    g_hat = [-1.0, -1.0]
    H_hat = make_hessian([1.0 0.0; 0.0 1.0])
    LB = [0.0, 0.0]
    UB = [2.0, 2.0]
    
    is_feasible, step, step_hat, step_value = compute_newton_step(x, gn, gn_hat, g_hat, H_hat, LB, UB)
    @test is_feasible == true
    @test step == gn
    @test step_hat == gn_hat
    @test step_value isa Real
    
    # Test case 2: Newton step is not feasible
    x = [1.0, 1.0]
    gn = [2.0, 2.0]
    gn_hat = [2.0, 2.0]
    g_hat = [-1.0, -1.0]
    H_hat = make_hessian([1.0 0.0; 0.0 1.0])
    LB = [0.0, 0.0]
    UB = [2.0, 2.0]
    
    is_feasible, step, step_hat, step_value = compute_newton_step(x, gn, gn_hat, g_hat, H_hat, LB, UB)
    @test is_feasible == false
    @test step === nothing
    
    # Test case 3: Float32 inputs
    x = Float32[1.0, 1.0]
    gn = Float32[0.5, 0.5]
    gn_hat = Float32[0.5, 0.5]
    g_hat = Float32[-1.0, -1.0]
    H_hat = make_hessian(Float32[1.0 0.0; 0.0 1.0])
    LB = Float32[0.0, 0.0]
    UB = Float32[2.0, 2.0]
    
    is_feasible, step, step_hat, step_value = compute_newton_step(x, gn, gn_hat, g_hat, H_hat, LB, UB)
    @test is_feasible == true
    @test step == gn
    @test step_hat == gn_hat
    @test step_value isa Real
    @test eltype(step) == Float32
    
    # Test case 4: CuArray inputs
    if CUDA.has_cuda_gpu()
        x = CuArray([1.0, 1.0])
        gn = CuArray([0.5, 0.5])
        gn_hat = CuArray([0.5, 0.5])
        g_hat = CuArray([-1.0, -1.0])
        H_hat = x -> CUDA.CUBLAS.gemm('N', 'N', CuArray(Float64[1.0 0.0; 0.0 1.0]), x)
        LB = CuArray([0.0, 0.0])
        UB = CuArray([2.0, 2.0])
        
        is_feasible, step, step_hat, step_value = compute_newton_step(x, gn, gn_hat, g_hat, H_hat, LB, UB)
        @test is_feasible == true
        @test step isa CuArray
        @test step_hat isa CuArray
        @test step_value isa Real
    else
        @info "CUDA not available, skipping CuArray tests."
    end
end

@testset "compute_interior_newton_step tests" begin
    using TrustRegionReflective: compute_interior_newton_step
    
    # Test helper function to create a simple Hessian operator
    function make_hessian(matrix)
        return x -> matrix * x
    end
    
    # Test case 1: Basic functionality
    x = [1.0, 1.0]
    gn = [2.0, 2.0]
    gn_hat = [2.0, 2.0]
    g_hat = [-1.0, -1.0]
    H_hat = make_hessian([1.0 0.0; 0.0 1.0])
    D = [1.0, 1.0]
    theta = 0.95
    LB = [0.0, 0.0]
    UB = [2.0, 2.0]
    
    step, step_hat, step_value = compute_interior_newton_step(x, gn, gn_hat, g_hat, H_hat, D, theta, LB, UB)
    @test step isa Vector
    @test step_hat isa Vector
    @test step_value isa Real
    @test all(x .+ step .>= LB)
    @test all(x .+ step .<= UB)
    
    # Test case 2: Float32 inputs
    x = Float32[1.0, 1.0]
    gn = Float32[2.0, 2.0]
    gn_hat = Float32[2.0, 2.0]
    g_hat = Float32[-1.0, -1.0]
    H_hat = make_hessian(Float32[1.0 0.0; 0.0 1.0])
    D = Float32[1.0, 1.0]
    theta = Float32(0.95)
    LB = Float32[0.0, 0.0]
    UB = Float32[2.0, 2.0]
    
    step, step_hat, step_value = compute_interior_newton_step(x, gn, gn_hat, g_hat, H_hat, D, theta, LB, UB)
    @test step isa Vector{Float32}
    @test step_hat isa Vector{Float32}
    @test step_value isa Real
    @test all(x .+ step .>= LB)
    @test all(x .+ step .<= UB)
    
    # Test case 3: CuArray inputs
    if CUDA.has_cuda_gpu()
        x = CuArray([1.0, 1.0])
        gn = CuArray([2.0, 2.0])
        gn_hat = CuArray([2.0, 2.0])
        g_hat = CuArray([-1.0, -1.0])
        H_hat = x -> CUDA.CUBLAS.gemm('N', 'N', CuArray(Float64[1.0 0.0; 0.0 1.0]), x)
        D = CuArray([1.0, 1.0])
        theta = 0.95
        LB = CuArray([0.0, 0.0])
        UB = CuArray([2.0, 2.0])
        
        step, step_hat, step_value = compute_interior_newton_step(x, gn, gn_hat, g_hat, H_hat, D, theta, LB, UB)
        @test step isa CuArray
        @test step_hat isa CuArray
        @test step_value isa Real
        @test all(Array(x .+ step .>= LB))
        @test all(Array(x .+ step .<= UB))
    else
        @info "CUDA not available, skipping CuArray tests."
    end
end

@testset "compute_reflected_step tests" begin
    using TrustRegionReflective: compute_reflected_step
    
    # Test helper function to create a simple Hessian operator
    function make_hessian(matrix)
        return x -> matrix * x
    end
    
    # Test case 1: Basic functionality
    x = [1.0, 1.0]
    gn = [2.0, 2.0]
    gn_hat = [2.0, 2.0]
    g_hat = [-1.0, -1.0]
    H_hat = make_hessian([1.0 0.0; 0.0 1.0])
    D = [1.0, 1.0]
    trust_radius = 5.0
    theta = 0.95
    LB = [0.0, 0.0]
    UB = [2.0, 2.0]
    
    step, step_hat, step_value = compute_reflected_step(x, gn, gn_hat, g_hat, H_hat, D, trust_radius, theta, LB, UB)
    @test step isa Vector
    @test step_hat isa Vector
    @test step_value isa Real
    
    # Test case 2: Float32 inputs
    x = Float32[1.0, 1.0]
    gn = Float32[2.0, 2.0]
    gn_hat = Float32[2.0, 2.0]
    g_hat = Float32[-1.0, -1.0]
    H_hat = make_hessian(Float32[1.0 0.0; 0.0 1.0])
    D = Float32[1.0, 1.0]
    trust_radius = Float32(5.0)
    theta = Float32(0.95)
    LB = Float32[0.0, 0.0]
    UB = Float32[2.0, 2.0]
    
    step, step_hat, step_value = compute_reflected_step(x, gn, gn_hat, g_hat, H_hat, D, trust_radius, theta, LB, UB)
    @test step isa Vector{Float32}
    @test step_hat isa Vector{Float32}
    @test step_value isa Real
    
    # Test case 3: CuArray inputs
    if CUDA.has_cuda_gpu()
        x = CuArray([1.0, 1.0])
        gn = CuArray([2.0, 2.0])
        gn_hat = CuArray([2.0, 2.0])
        g_hat = CuArray([-1.0, -1.0])
        H_hat = x -> CUDA.CUBLAS.gemm('N', 'N', CuArray(Float64[1.0 0.0; 0.0 1.0]), x)
        D = CuArray([1.0, 1.0])
        trust_radius = 5.0
        theta = 0.95
        LB = CuArray([0.0, 0.0])
        UB = CuArray([2.0, 2.0])
        
        step, step_hat, step_value = compute_reflected_step(x, gn, gn_hat, g_hat, H_hat, D, trust_radius, theta, LB, UB)
        @test step isa CuArray
        @test step_hat isa CuArray
        @test step_value isa Real
    else
        @info "CUDA not available, skipping CuArray tests."
    end
end

@testset "compute_steepest_descent_step tests" begin
    using TrustRegionReflective: compute_steepest_descent_step
    
    # Test helper function to create a simple Hessian operator
    function make_hessian(matrix)
        return x -> matrix * x
    end
    
    # Test case 1: Basic functionality
    x = [1.0, 1.0]
    g_hat = [1.0, 1.0]
    H_hat = make_hessian([1.0 0.0; 0.0 1.0])
    D = [1.0, 1.0]
    trust_radius = 5.0
    theta = 0.95
    LB = [0.0, 0.0]
    UB = [2.0, 2.0]
    
    step, step_hat, step_value = compute_steepest_descent_step(x, g_hat, H_hat, D, trust_radius, theta, LB, UB)
    @test step isa Vector
    @test step_hat isa Vector
    @test step_value isa Real
    @test all(x .+ step .>= LB)
    @test all(x .+ step .<= UB)
    
    # Test case 2: Float32 inputs
    x = Float32[1.0, 1.0]
    g_hat = Float32[1.0, 1.0]
    H_hat = make_hessian(Float32[1.0 0.0; 0.0 1.0])
    D = Float32[1.0, 1.0]
    trust_radius = Float32(5.0)
    theta = Float32(0.95)
    LB = Float32[0.0, 0.0]
    UB = Float32[2.0, 2.0]
    
    step, step_hat, step_value = compute_steepest_descent_step(x, g_hat, H_hat, D, trust_radius, theta, LB, UB)
    @test step isa Vector{Float32}
    @test step_hat isa Vector{Float32}
    @test step_value isa Real
    @test all(x .+ step .>= LB)
    @test all(x .+ step .<= UB)
    
    # Test case 3: CuArray inputs
    if CUDA.has_cuda_gpu()
        x = CuArray([1.0, 1.0])
        g_hat = CuArray([1.0, 1.0])
        H_hat = x -> CUDA.CUBLAS.gemm('N', 'N', CuArray(Float64[1.0 0.0; 0.0 1.0]), x)
        D = CuArray([1.0, 1.0])
        trust_radius = 5.0
        theta = 0.95
        LB = CuArray([0.0, 0.0])
        UB = CuArray([2.0, 2.0])
        
        step, step_hat, step_value = compute_steepest_descent_step(x, g_hat, H_hat, D, trust_radius, theta, LB, UB)
        @test step isa CuArray
        @test step_hat isa CuArray
        @test step_value isa Real
    else
        @info "CUDA not available, skipping CuArray tests."
    end
end

@testset "choose_step tests" begin
    using TrustRegionReflective: choose_step
    
    # Test helper function to create a simple Hessian operator
    function make_hessian(matrix)
        return x -> matrix * x
    end
    
    # Test case 1: When Newton step is feasible
    x = [1.0, 1.0]
    gn = [0.5, 0.5]
    gn_hat = [0.5, 0.5]
    g_hat = [-1.0, -1.0]
    H_hat = make_hessian([1.0 0.0; 0.0 1.0])
    D = [1.0, 1.0]
    trust_radius = 5.0
    theta = 0.95
    LB = [0.0, 0.0]
    UB = [2.0, 2.0]
    
    step, step_hat, step_value = choose_step(x, H_hat, g_hat, gn, gn_hat, D, trust_radius, theta, LB, UB)
    @test step isa Vector
    @test step_hat isa Vector
    @test step_value isa Real
    @test all(x .+ step .>= LB)
    @test all(x .+ step .<= UB)
    
    # Test case 2: When Newton step is not feasible
    x = [1.0, 1.0]
    gn = [2.0, 2.0]
    gn_hat = [2.0, 2.0]
    g_hat = [-1.0, -1.0]
    H_hat = make_hessian([1.0 0.0; 0.0 1.0])
    D = [1.0, 1.0]
    trust_radius = 5.0
    theta = 0.95
    LB = [0.0, 0.0]
    UB = [2.0, 2.0]
    
    step, step_hat, step_value = choose_step(x, H_hat, g_hat, gn, gn_hat, D, trust_radius, theta, LB, UB)
    @test step isa Vector
    @test step_hat isa Vector
    @test step_value isa Real
    @test all(x .+ step .>= LB)
    @test all(x .+ step .<= UB)
    @test norm(step_hat) <= trust_radius
    
    # Test case 3: Float32 inputs
    x = Float32[1.0, 1.0]
    gn = Float32[0.5, 0.5]
    gn_hat = Float32[0.5, 0.5]
    g_hat = Float32[-1.0, -1.0]
    H_hat = make_hessian(Float32[1.0 0.0; 0.0 1.0])
    D = Float32[1.0, 1.0]
    trust_radius = Float32(5.0)
    theta = Float32(0.95)
    LB = Float32[0.0, 0.0]
    UB = Float32[2.0, 2.0]
    
    step, step_hat, step_value = choose_step(x, H_hat, g_hat, gn, gn_hat, D, trust_radius, theta, LB, UB)
    @test step isa Vector{Float32}
    @test step_hat isa Vector{Float32}
    @test step_value isa Real
    @test all(x .+ step .>= LB)
    @test all(x .+ step .<= UB)
    
    # Test case 4: CuArray inputs
    if CUDA.has_cuda_gpu()
        x = CuArray([1.0, 1.0])
        gn = CuArray([0.5, 0.5])
        gn_hat = CuArray([0.5, 0.5])
        g_hat = CuArray([-1.0, -1.0])
        H_hat = x -> CUDA.CUBLAS.gemm('N', 'N', CuArray(Float64[1.0 0.0; 0.0 1.0]), x)
        D = CuArray([1.0, 1.0])
        trust_radius = 5.0
        theta = 0.95
        LB = CuArray([0.0, 0.0])
        UB = CuArray([2.0, 2.0])
        
        step, step_hat, step_value = choose_step(x, H_hat, g_hat, gn, gn_hat, D, trust_radius, theta, LB, UB)
        @test step isa CuArray
        @test step_hat isa CuArray
        @test step_value isa Real
        @test all(Array(x .+ step .>= LB))
        @test all(Array(x .+ step .<= UB))
    else
        @info "CUDA not available, skipping CuArray tests."
    end
end