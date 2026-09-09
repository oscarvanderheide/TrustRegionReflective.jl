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
