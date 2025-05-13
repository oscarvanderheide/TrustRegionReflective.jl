using TrustRegionReflective
using Test
using LinearAlgebra
using CUDA
using Random
using LinearMaps
using TimerOutputs

@testset "Convergence Tests" begin
    # Set random seed for reproducible results
    Random.seed!(123)

    # Helper function to set up large bounds (effectively unconstrained)
    function unconstrained_bounds(n, T=Float64)
        LB = fill(convert(T, -Inf), n)
        UB = fill(convert(T, Inf), n)
        return LB, UB
    end

    # Basic timer for all tests
    to = TimerOutputs.TimerOutput()
    
    # Empty callback function
    callback_fn(iter, state) = nothing

    @testset "Quadratic Function - Unconstrained" begin
        # Minimization of f(x) = (x - [2, 3])^2 = (x₁ - 2)² + (x₂ - 3)²
        # Global minimum at x = [2, 3] with f(x) = 0
        
        function quad_objective(x, mode)
            target = [2.0, 3.0]
            residuals = x - target
            f = sum(residuals.^2)
            
            if mode == "f"
                return f
            elseif mode == "fr"
                return f, residuals
            elseif mode == "frg"
                g = 2.0 * residuals
                return f, residuals, g
            elseif mode == "frgH"
                g = 2.0 * residuals
                H = LinearMap(x -> 2.0 * x, 2, 2)
                return f, residuals, g, H
            end
        end
        
        x0 = [0.0, 0.0]  # Starting point
        LB, UB = unconstrained_bounds(2)
        options = TRFOptions{Float64}()
        
        # Solve the problem
        result = trust_region_reflective(quad_objective, x0, LB, UB, callback_fn, to, options)
        
        # Check convergence to the known minimum
        @test isapprox(result, [2.0, 3.0], rtol=1e-5)
        @test isapprox(quad_objective(result, "f"), 0.0, atol=1e-10)
    end

    @testset "Rosenbrock Function - Unconstrained" begin
        # Minimization of Rosenbrock function f(x,y) = 100(y - x²)² + (1 - x)²
        # Global minimum at x = [1, 1] with f(x) = 0
        
        function rosenbrock_objective(x, mode)
            a, b = 1.0, 100.0
            
            # Direct calculation of the function value
            f = b * (x[2] - x[1]^2)^2 + (x[1] - a)^2
            
            # For residual form, we use this decomposition:
            r1 = x[1] - a
            r2 = sqrt(b) * (x[2] - x[1]^2)
            residuals = [r1, r2]
            
            # Function value should match residual sum of squares
            @assert isapprox(f, sum(residuals.^2))
            
            if mode == "f"
                return f
            elseif mode == "fr"
                return f, residuals
            elseif mode == "frg"
                # Gradient of f = b*(y-x^2)^2 + (x-a)^2
                # ∂f/∂x = -4*b*x*(y-x^2) + 2*(x-a)
                # ∂f/∂y = 2*b*(y-x^2)
                g1 = -4*b*x[1]*(x[2] - x[1]^2) + 2*(x[1] - a)
                g2 = 2*b*(x[2] - x[1]^2)
                g = [g1, g2]
                return f, residuals, g
            elseif mode == "frgH"
                # Gradient of f
                g1 = -4*b*x[1]*(x[2] - x[1]^2) + 2*(x[1] - a)
                g2 = 2*b*(x[2] - x[1]^2)
                g = [g1, g2]
                
                # Hessian operator
                function H_op(v)
                    # Hessian components:
                    # ∂²f/∂x² = -4*b*(y-x^2) + 8*b*x^2 + 2
                    # ∂²f/∂y² = 2*b
                    # ∂²f/∂x∂y = ∂²f/∂y∂x = -4*b*x
                    H11 = -4*b*(x[2] - x[1]^2) + 8*b*x[1]^2 + 2
                    H22 = 2*b
                    H12 = -4*b*x[1]
                    H21 = H12  # Symmetric matrix
                    
                    # Matrix-vector product
                    return [H11*v[1] + H12*v[2], H21*v[1] + H22*v[2]]
                end

                H = LinearMap(H_op, 2, 2)
                
                return f, residuals, g, H
            end
        end
        
        x0 = [-1.2, 1.0]  # Standard starting point for Rosenbrock
        LB, UB = unconstrained_bounds(2)
        # Increase max iterations to allow convergence to the true minimum
        options = TRFOptions{Float64}(max_iter_trf=50, init_scale_radius=1.0)
        
        # Solve the problem
        result = trust_region_reflective(rosenbrock_objective, x0, LB, UB, callback_fn, to, options)
        
        # Check convergence to the known minimum
        @test isapprox(result, [1.0, 1.0], rtol=1e-4)
        @test rosenbrock_objective(result, "f") < 1e-8
    end

    @testset "Quadratic Function - Box Constrained" begin
        # Minimization of f(x) = (x - [3, 4])^2 = (x₁ - 3)² + (x₂ - 4)²
        # With box constraints 0 ≤ x₁ ≤ 2, 0 ≤ x₂ ≤ 3
        # Solution is at the nearest feasible point to the unconstrained minimum: x = [2, 3]
        
        function constrained_quad_objective(x, mode)
            target = [3.0, 4.0]  # Unconstrained minimum is outside feasible region
            residuals = x - target
            f = sum(residuals.^2)
            
            if mode == "f"
                return f
            elseif mode == "fr"
                return f, residuals
            elseif mode == "frg"
                g = 2.0 * residuals
                return f, residuals, g
            elseif mode == "frgH"
                g = 2.0 * residuals
                H = LinearMap(x -> 2.0 * x, 2, 2)
                return f, residuals, g, H
            end
        end
        
        x0 = [1.0, 1.0]  # Starting point
        LB = [0.0, 0.0]  # Lower bounds
        UB = [2.0, 3.0]  # Upper bounds
        options = TRFOptions{Float64}()
        
        # Solve the problem
        result = trust_region_reflective(constrained_quad_objective, x0, LB, UB, callback_fn, to, options)
        
        # Check convergence to the known constrained minimum
        @test isapprox(result, [2.0, 3.0], rtol=1e-5)
    end

    @testset "Float32 Precision Test" begin
        # Test with Float32 precision to ensure the solver works with different float types
        function quad_objective_f32(x, mode)
            target = Float32[2.0, 3.0]
            residuals = x - target
            f = sum(residuals.^2)
            
            if mode == "f"
                return f
            elseif mode == "fr"
                return f, residuals
            elseif mode == "frg"
                g = 2.0f0 * residuals
                return f, residuals, g
            elseif mode == "frgH"
                g = 2.0f0 * residuals
                H = LinearMap(x -> 2.0f0 * x, 2,2)
                return f, residuals, g, H
            end
        end
        
        x0 = Float32[0.0, 0.0]
        LB, UB = unconstrained_bounds(2, Float32)
        options = TRFOptions{Float32}()
        
        # Solve the problem
        result = trust_region_reflective(quad_objective_f32, x0, LB, UB, callback_fn, to, options)
        
        # Check convergence and type
        @test eltype(result) == Float32
        @test isapprox(result, Float32[2.0, 3.0], rtol=1e-4)
    end
    
    @testset "CuArray Test" begin
        # Only run if CUDA is available
        if CUDA.functional()
            function quad_objective_cuda(x, mode)
                target = CuArray([2.0, 3.0])
                residuals = x - target
                f = sum(residuals.^2)
                
                if mode == "f"
                    return f
                elseif mode == "fr"
                    return f, residuals
                elseif mode == "frg"
                    g = 2.0 * residuals
                    return f, residuals, g
                elseif mode == "frgH"
                    g = 2.0 * residuals
                    H = LinearMap(x -> 2.0 * x, 2, 2)
                    return f, residuals, g, H
                end
            end
            
            x0 = CuArray([0.0, 0.0])
            LB = CuArray(fill(-Inf, 2))
            UB = CuArray(fill(Inf, 2))
            options = TRFOptions{Float64}()
            
            # Solve the problem
            result = trust_region_reflective(quad_objective_cuda, x0, LB, UB, callback_fn, to, options)
            
            # Check convergence
            @test isapprox(Array(result), [2.0, 3.0], rtol=1e-5)
        else
            @info "CUDA not available, skipping CuArray test."
        end
    end
end