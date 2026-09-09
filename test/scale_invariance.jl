@testset "Outward trust-boundary root accuracy" begin
    using TrustRegionReflective: positive_stepsize_to_bound_trust_region
    for T in (Float32, Float64), device in (identity, CuArray)
        device === CuArray && !CUDA.functional() && continue
        # Choose a radius exactly one representable number above ‖x‖. Check the
        # root itself, not just the final norm, which can hide a large relative error.
        x = T[0.6, 0.8]
        p = T[1e-4, 0]
        radius = nextfloat(norm(device(x)))
        xp = BigFloat(dot(device(x), device(p))) / BigFloat(norm(device(p)))
        gap = (BigFloat(radius) - BigFloat(norm(device(x)))) *
            (BigFloat(radius) + BigFloat(norm(device(x))))
        expected = gap / (sqrt(xp^2 + gap) + xp) / BigFloat(norm(device(p)))
        step = positive_stepsize_to_bound_trust_region(device(x), device(p), radius)
        @test step ≈ T(expected) rtol=10eps(T)
        @test step > 0 && isfinite(step)
    end
end

@testset "Steihaug objective scaling" begin
    using TrustRegionReflective: steihaug_store_steps
    for T in (Float32, Float64), device in (identity, CuArray)
        device === CuArray && !CUDA.functional() && continue
        # Rescale both g and H: the minimizer and trust-region geometry stay fixed.
        for scale in T[1, 1e-5, 1e-9]
            diagonal = device(T[1, 3])
            g = device(T[0.1, -0.2]) .* scale
            H = x -> scale .* diagonal .* x
            steps, _ = steihaug_store_steps(H, g, T(2), identity, 10, T(1e-5), zero(g))
            @test Array(last(steps)) ≈ T[-0.1, 0.2/3] rtol=50eps(T)
            @test all(isfinite, last(steps))
        end
        # A small positive eigenvalue still permits an interior Newton step.
        g = device(T[1e-3])
        steps, _ = steihaug_store_steps(x -> T(1e-9) .* x, g, T(1e8), identity,
            5, T(1e-4), zero(g))
        @test Array(last(steps)) ≈ T[-1e6] rtol=10eps(T)
        # A vanishing preconditioner must not form an infinite boundary step times zero.
        steps, _ = steihaug_store_steps(identity, g, one(T), zero, 5, T(1e-4), zero(g))
        @test iszero(norm(last(steps)))
        @test all(isfinite, last(steps))
        # Actual zero and negative curvature still take finite boundary steps.
        for curvature in T[0, -1]
            steps, _ = steihaug_store_steps(x -> curvature .* x, g, one(T), identity,
                5, T(1e-4), zero(g))
            @test norm(last(steps)) ≈ one(T) rtol=10eps(T)
            @test dot(g, last(steps)) < 0
        end
    end
end

@testset "TRF stops at parameter precision" begin
    for T in (Float32, Float64), device in (identity, CuArray)
        device === CuArray && !CUDA.functional() && continue
        # The exact minimum is x0 + 1, which is closer than one representable increment.
        # Subtract x0 before 1 so the residual still exposes that nonzero gradient.
        x0 = device(T[T === Float32 ? 1e8 : 1e16])
        calls = Ref(0)
        accepted = Ref(0)
        objective = function (x, mode)
            calls[] += 1
            r = (x .- x0) .- one(T)
            f = sum(abs2, r) / 2
            mode == "fr" && return f, r
            return f, r, r, Diagonal(device(ones(T, 1))), nothing
        end
        options = TRFOptions{T}(max_iter_trf=3)
        result = trust_region_reflective(objective, x0, device(T[-Inf]), device(T[Inf]),
            (iteration, state) -> (accepted[] += 1), TimerOutput(), options)
        @test result == x0
        @test calls[] == 1
        @test accepted[] == 0
    end
end
