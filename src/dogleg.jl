"""
    dogleg(obj, lb, ub, x0, plotfun)

The `dogleg` function implements the dogleg trust region algorithm for constrained optimization problems. It finds the minimum of the objective function `obj` subject to the lower bounds `lb` and upper bounds `ub`, starting from the initial guess `x0`. The function `plotfun` is used to visualize the optimization process.

# Arguments
- `obj`: The objective function to be minimized. It takes a vector `x` as input and returns a tuple `(f, r, g, H)`, where `f` is the cost function value, `r` is the residual vector, `g` is the gradient vector, and `H` is the Hessian matrix.
- `lb`: The lower bounds for the parameters.
- `ub`: The upper bounds for the parameters.
- `x0`: The initial guess for the parameters.
- `plotfun`: A function used to visualize the optimization process. It takes a vector `x` as input and produces a plot.

# Returns
- `x`: The optimized parameters.
- `f_history`: A vector containing the cost function values at each iteration.
"""
function dogleg(obj, lb, ub, x0, plotfun)

    # Define projection operator
    project = x -> map(median, zip(lb, x, ub))

    # Function to check whether parameters are within feasible region
    within_bounds = x -> all(lb .<= x .<= ub)

    # Check that initial guess is within feasible region
    @assert within_bounds(x0)

    # Compute boundary indices
    # activate_set = (x,g) -> (x .<= lb .& g .> 0) .| (x .>= ub .& g .< 0)
    activate_set = x -> (x .<= lb) .| (x .>= ub)

    # Create a reduced version of (approximate) Hessian that does not modify parameters that are on the boundaries already.
    function reduced_operator(H, p, activate_set)
        p̃ = copy(p)
        p̃[activate_set] .= 0
        Hp̃ = H * p̃
        Hp̃[activate_set] .= p[activate_set]
        return Hp̃
    end

    # Modify the preconditioner, if available.
    PC(x) = x

    # Check the options structure.
    CGMaxIter = 20
    CGTol = 1e-6
    Δ = 0.1 * norm(x0)
    max_iter = 10
    ρ_min = 0.1

    # Compute initial cost
    f, r = obj(x0, 0)

    x = x0
    x_new = x
    iter = 0

    f_history = []

    while iter <= max_iter

        # Evaluate objective, compute gradient and build Hessian multiply function
        f, r, g, H = obj(x, 2)
        push!(f_history, f)

        @info "Iteration $iter"
        @info "    Cost = $f"
        @info "    Δ    = $Δ"

        # Compute the Cauchy point pᶜᵖ = -α*g.
        gᵀg = g' * g
        gᵀHg = g' * (H * g)
        α = gᵀg / gᵀHg
        pᶜᵖ = -α * g

        # Compute the Gauss-Newton step pgn.
        H̃ = p -> reduced_operator(H, p, activate_set(x))
        pᵍⁿ = mycg(H̃, -g, CGTol, CGMaxIter, PC, pᶜᵖ)

        # H̃ = Krylov.LinearOperators.LinearOperator(eltype(x), length(x), length(x), true, false, p -> reduced_operator(H,p,activate_set(x)))
        # pᵍⁿ = Krylov.cg(H̃,-g,itmax=CGMaxIter,verbose=true)[1]

        # How much is the curvature information bending the step?
        # dot(normalize(pᵍⁿ), normalize(pᶜᵖ))

        # Check if we actually ever go outside of feasible region
        !within_bounds(x + pᵍⁿ) && @info "x + pᵍⁿ outside of bounds"
        !within_bounds(x + pᶜᵖ) && @info "x + pᶜᵖ outside of bounds"

        # Project pᵍⁿ and pᶜᵖ.
        pᵍⁿ = project(x + pᵍⁿ) - x
        pᶜᵖ = project(x + pᶜᵖ) - x

        # Plane search in the plane spanned by {pᵍⁿ,pᵍⁿ}.

        # if isa(options.PlaneSearch,'function_handle')

        #     state = output;
        #     state.grad = grad;

        #     [alpha,outputps] = options.PlaneSearch( ...
        #         F,dF,z,deserialize(pgn,dim),deserialize(pcp,dim), ...
        #         state,options.PlaneSearchOptions);

        #     output.alpha(:,end+1) = alpha;

        #     if length(alpha) < 3, alpha(3) = 1; end
        #     p = alpha(1)*pgn+alpha(2)*pcp;
        #     z1 = deserialize(alpha(3)*(z0+p),dim);
        #     relstep = norm(p)/norm(z0); if isnan(relstep), relstep = 0; end
        #     if isfield(outputps,'fval')
        #         fval = outputps.fval;
        #     else
        #         switch method
        #             case {'F+dFdzc','F+dFdzx+dFdconjzx','F+dFdz','F+dFdzx'}
        #                 Fval = F(z); fval = 0.5*sum(Fval(:)'*Fval(:));
        #             case {'f+JHJ+JHF','f+JHJx+JHF'}
        #                 fval = f(z);
        #         end
        #     end
        #     if isfield(outputps,'info')
        #         output.infops(end+1) = outputps.info;
        #     end
        #     rho = 1;
        # else
        #     rho = -inf;
        # end

        # Dogleg trust region computes pᶜᵖ and pᵍⁿ = pᶜᵖ + something
        # Then we have three options:
        # 1. If pᵍⁿ is within the trust-region, choose pᵍⁿ
        # 2. If pᶜᵖ itself is already outside of the trust-region, scale back pᶜᵖ to be on the boundary
        # 3. Otherwise, take pᶜᵖ + β * (pᵍⁿ-pᶜᵖ), such that it is on the boundary

        ρ = -1

        while ρ <= ρ_min

            # Compute the dogleg step p.
            # Assume the projection did not alter the pᵍⁿ or pᶜᵖ too much,
            # computing dfval's would otherwise be quite expensive.

            if norm(pᵍⁿ) <= Δ # option 1
                @info "    Gauss-Newton step chosen"
                p = pᵍⁿ
            elseif norm(pᶜᵖ) >= Δ # option 2
                @info "    Cauchy-point chosen"
                p = (Δ / norm(pᶜᵖ)) * pᶜᵖ
            else # option 3
                @info "    pᶜᵖ + β * (pᵍⁿ-pᶜᵖ) chosen"
                γ = pᵍⁿ - pᶜᵖ
                γ² = γ' * γ
                c = pᶜᵖ' * γ
                c² = c * c
                Δ² = Δ * Δ
                if c <= 0
                    β = (-c + sqrt(c² + γ² * (Δ² - norm(pᶜᵖ)^2))) / γ²
                else
                    β = (Δ² - norm(pᶜᵖ)^2) / (c + sqrt(c² + γ² * (Δ² - norm(pᶜᵖ)^2)))
                end
                p = pᶜᵖ + β * (pᵍⁿ - pᶜᵖ)
            end

            # Estimate objective function improvement.
            # dot(normalize(p), normalize(pᶜᵖ))
            expected_reduction = -p' * g - (1 / 2) * p' * (H * p)

            if expected_reduction > 0
                x_new = x + p
                f_new, r_new = obj(x_new, 0)

                # Compute the trustworthiness rho.
                actual_reduction = (f - f_new)
                ρ = actual_reduction / expected_reduction
            end

            # Update trust region radius Δ.
            if ρ > 0.5
                @info "        ρ = $ρ"
                Δ = max(Δ, 2 * norm(p))
                @info "        Increasing Δ to $Δ"
            else
                σ = (1 - 0.25) / (1 + exp(-14 * (ρ - 0.25))) + 0.25
                if norm(pᵍⁿ) < σ * Δ && ρ < ρ_min
                    e = ceil(log2(norm(pᵍⁿ) / Δ) / log2(σ))
                    Δ = σ^e * Δ
                else
                    Δ = σ * Δ
                end
                @info "        Decreasing Δ to $Δ"
            end
        end

        # Update parameters
        @info "    Step accepted"
        x = x_new
        iter += 1
        plotfun(x)
        @info ""

    end

    return x, f_history

end

"""
    mycg(A, b, tol, maxit, M, x0)

Solves a linear system of equations using the Preconditioned Conjugate Gradient (PCG) method.

# Arguments
- `A`: A function that computes the matrix-vector product of `A` and a vector `x`.
- `b`: The right-hand side vector.
- `tol`: The tolerance for convergence. The algorithm stops when the relative residual is below this value.
- `maxit`: The maximum number of iterations.
- `M`: A function that applies a preconditioner to a vector.
- `x0`: The initial guess for the solution.

# Returns
The solution vector `x` that approximately solves the linear system `Ax = b`.
"""
function mycg(A, b, tol, maxit, M, x0)

    # Initialize PCG.

    x = x0
    r = A(x) - b
    y = M(r)
    d = -y
    rr = r' * y
    normb = norm(b)

    flag = 1

    @info "    "

    # Preconditioned conjugate gradients.
    for iter = 1:maxit
        print(".")

        Ad = A(d)
        α = rr / (d' * Ad)

        x = x + α * d
        r = r + α * Ad
        rr1 = rr
        y = M(r)
        rr = r' * y

        relres = norm(r) / normb
        if relres < tol
            flag = 0
            break
        end

        β = rr / rr1
        d = -y + β * d

    end
    print("\n")

    flag == 0 && @info "    CG converged to tolerance"
    flag == 1 && @info "    CG maxit reached"


    return x
end
