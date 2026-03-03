# Understanding the Trust Region Reflective Algorithm

A guide for MRI researchers. No optimization background required.

---

## Part 1: The Big Picture (No Math)

### What problem are we solving?

You have an MRI model — some mathematical equation that predicts what your scanner measures, given a set of tissue parameters (like T1, T2, proton density, etc.). You measured actual data from the scanner, and now you want to find the tissue parameters that make your model's predictions match the measured data as closely as possible.

This is called **least-squares fitting**: you're minimizing the sum of squared differences between your model's predictions and the real measurements.

There's a catch: tissue parameters have **physical limits**. T1 and T2 can't be negative and should not be higher than - say - five seconds. These are your **box constraints** — each parameter has a lower bound (LB) and an upper bound (UB), forming a "box" the solution must stay inside.

### The core idea: trust regions

Imagine you're standing on a hilly landscape in fog. You can't see the whole landscape, but you can feel the slope under your feet and estimate the curvature nearby. You want to walk downhill to the lowest point.

A **trust region** is like saying: "I trust my local map of the terrain for about 10 meters around me. I'll find the best step within that 10-meter circle, take it, then re-survey."

- If the step worked well (the terrain really went down as predicted), you **grow** the trusted circle — maybe to 20 meters.
- If the step was disappointing (terrain didn't go down as much as expected), you **shrink** the circle — maybe to 5 meters — and try a more cautious step.
- You repeat until you can't improve anymore.

### What makes this "reflective"?

The bounds (LB, UB) act like walls. When the algorithm's preferred step would go through a wall, it doesn't just stop at the wall — it **reflects** off the wall, like a billiard ball bouncing off the rail. This reflection strategy helps the algorithm continue making progress even when it's near a constraint boundary.

### Why this algorithm for MRI?

In MRI parameter fitting, you often have **very large problems** — thousands or even millions of parameters (e.g., fitting a map for every voxel simultaneously, or fitting a model with many spatial basis functions). In these cases, the Jacobian matrix $J$ and the Hessian matrix $H = J^T J$ are far too large to store in memory, let alone decompose (e.g., via SVD or Cholesky) to solve the normal equations directly.

The Trust Region Reflective algorithm is designed for exactly this situation. It only ever needs **matrix-vector products** — given a vector $v$, it needs someone to compute $H \cdot v$ — and never needs the full matrix. This is why it uses an iterative subproblem solver (Steihaug conjugate gradient) rather than a direct matrix factorization. As long as you can compute the product $J^T (J \cdot v)$ for any vector $v$ (which is much cheaper than forming $J^T J$), this algorithm works.

### The high-level loop

```
1. Start at initial guess x₀
2. Evaluate: How good is x? What's the gradient (downhill direction)? What's the curvature?
3. Compute scaling factors that slow down parameters near their bounds
4. Solve a local subproblem: "What's the best step within my trust region?"
5. Try the step — did it actually improve things?
   - Yes → accept it, possibly grow the trust region
   - No  → reject it, shrink the trust region, try a smaller step
6. Repeat until converged or max iterations reached
```

---

## Part 2: More Detail (Some Math, Still Accessible)

### The objective function

Your fitting problem looks like this:

$$\min_x \; f(x) = \frac{1}{2} \|r(x)\|^2 = \frac{1}{2} \sum_i r_i(x)^2$$

where $r(x)$ is the vector of **residuals** (differences between model predictions and measured data), and $x$ is your parameter vector (the tissue properties you're solving for).

The constraints are:

$$\text{LB}_i \leq x_i \leq \text{UB}_i \quad \text{for each parameter } i$$

### Gradient and Hessian

To navigate the cost landscape, the algorithm needs:

- **Gradient** $g = \nabla f(x)$: points in the direction of steepest ascent. We want to step in the opposite direction.
- **Hessian** $H = \nabla^2 f(x)$: describes the curvature. Tells us whether the landscape curves up (good — it's a valley) or is flat (need to be careful).

In least-squares problems, the Hessian is typically approximated by $J^T J$ where $J$ is the Jacobian of the residuals. This approximation is always positive semi-definite, which means the **approximate** landscape (the quadratic model the algorithm builds at each step) is always bowl-shaped. The *actual* cost landscape of your MRI fitting problem is generally not bowl-shaped — it can have multiple valleys and ridges — but the algorithm only ever works with the local bowl-shaped approximation, which is valid in a small neighborhood around the current point.

Importantly, this algorithm never needs to store the full Jacobian $J$ or the Hessian $J^T J$ in memory. It only requires the ability to compute matrix-vector products $J^T(J \cdot v)$ for arbitrary vectors $v$. This makes it suitable for large-scale MRI problems where the Jacobian may have millions of entries.

### Coleman-Li scaling: respecting the bounds

This is the key innovation. Near a boundary, the algorithm creates **scaling factors** $v$ for each parameter:

- If parameter $x_i$ is far from any bound: $v_i = 1$ (no scaling, move freely).
- If $x_i$ is near its lower bound and the gradient wants to push it even lower: $v_i = x_i - \text{LB}_i$ (small number → tiny steps).
- If $x_i$ is near its upper bound and the gradient wants to push it even higher: $v_i = \text{UB}_i - x_i$ (small number → tiny steps).
- If the gradient would pull $x_i$ *away* from a nearby bound: $v_i = 1$ (no need to slow down).

The scaling matrix is $D = \sqrt{v}$, applied to transform the problem into a new coordinate system where taking a unit step is "safe" for all parameters.

**Intuition**: imagine parameter T1 has bounds $[0, 5000]$ ms and the gradient wants to push it upward (toward the upper bound).

- If T1 is at 2500 ms (middle of the range): scaling factor $v = 5000 - 2500 = 2500$. The algorithm moves freely — plenty of room.
- If T1 is at 4900 ms (getting close): scaling factor $v = 5000 - 4900 = 100$. Steps are scaled down proportionally.
- If T1 is at 4999 ms (very close): scaling factor $v = 5000 - 4999 = 1$. Steps become very small.
- If T1 is at 4999.99 ms (nearly at the wall): scaling factor $v = 0.01$. Steps are tiny, naturally preventing the parameter from crashing into the bound.

The closer to the wall, the more the brakes are applied — automatically and smoothly.

### The trust region subproblem

At each iteration, the algorithm solves this local problem in the scaled space:

$$\min_{\hat{s}} \; \hat{g}^T \hat{s} + \frac{1}{2} \hat{s}^T \hat{H} \hat{s} \quad \text{subject to} \quad \|\hat{s}\| \leq \Delta$$

where $\hat{g} = D \cdot g$ and $\hat{H}$ is the scaled Hessian operator, and $\Delta$ is the trust region radius. This is a quadratic approximation of the cost function — it's the "local map of the terrain" we talked about.

This subproblem is solved using the **Steihaug conjugate gradient method** (more on this below).

### Choosing the best step

Once the subproblem gives us a step direction, the algorithm considers three candidates:

1. **Full Newton step**: the subproblem solution, if it stays within bounds.
2. **Reflected Newton step**: if the Newton step hits a bound, bounce off it like a billiard ball and continue in the reflected direction.
3. **Steepest descent step**: just go straight downhill (the safest, most conservative option).

The algorithm evaluates all three using the quadratic model and picks whichever gives the most reduction. This ensures progress even in tricky geometric situations near corners of the constraint box.

### Accepting or rejecting the step

After choosing a step and computing $x_\text{new} = x + \text{step}$:

1. Evaluate the *actual* cost reduction: $f(x) - f(x_\text{new})$.
2. Compare to the *predicted* reduction from the quadratic model.
3. Compute the **ratio**: $\rho = \text{actual reduction} / \text{predicted reduction}$.

The ratio tells us how trustworthy our local model is:

| Ratio $\rho$  | Meaning | Action |
|---|---|---|
| $\rho > 0.75$ | Model was very accurate | Accept step, double trust radius |
| $0.1 < \rho < 0.75$ | Model was okay | Accept step, keep trust radius |
| $\rho < 0.25$ | Model overestimated improvement | Accept (if reduction > 0), shrink trust radius |
| $\rho < 0.1$ or $f$ got worse | Model was wrong | Reject step, halve trust radius, try smaller step |

### Convergence

The algorithm stops when either:

- The **scaled gradient norm** drops below a tolerance — meaning we're at a (possibly constrained) optimum.
- The **change in cost function** between iterations is negligibly small.
- The maximum number of iterations is reached.

The scaling factors from Coleman-Li are important here: at a constrained optimum, a parameter on its bound might still have a nonzero gradient (the wall is "holding it back"), but the scaled gradient will be zero because $v_i = 0$ for that parameter. This correctly identifies the solution.

---

## Part 3: All the Details

### File structure overview

| File | Purpose |
|---|---|
| `TrustRegionReflective.jl` | Module definition, options struct, solver state struct |
| `solver.jl` | Main iteration loop (`trust_region_reflective`) |
| `steihaug.jl` | Steihaug-CG subproblem solver (`steihaug_store_steps`) |
| `utils.jl` | All helper functions (scaling, step selection, quadratic utilities) |

---

### `TRFOptions` — Configuration

```julia
@kwdef struct TRFOptions{T<:Real}
    max_iter_trf::Int = 20         # Max outer iterations
    max_iter_steihaug::Int = 20    # Max inner CG iterations per outer iteration
    tol_steihaug::T = 1E-6         # CG convergence tolerance
    tol_convergence::T = 1E-6      # Overall convergence tolerance
    init_scale_radius::T = 0.1     # Initial trust radius = 0.1 * ‖x₀‖
    save_every_iter::Bool = false   # Whether to store x/f/r history at each step
    modified_reduction_for_ratio::Bool = false  # Whether to include CL terms in ratio
end
```

`max_iter_steihaug` controls how accurately the subproblem is solved. Higher = more accurate but slower per iteration. 

`init_scale_radius` sets the initial trust region as a fraction of the starting point's magnitude. If your initial guess is far from the solution, a smaller value (e.g., 0.01) makes the first steps more cautious.

---

### `trust_region_reflective` — Main solver (solver.jl)

#### Initialization (lines 1–38)

```julia
x = x0
f, r, g, H, H⁻¹_approx = objective(x, "frgH")
```

The user-provided `objective` function is called with mode `"frgH"` to get:
- `f`: scalar cost value ($\frac{1}{2}\|r\|^2$)
- `r`: residual vector
- `g`: gradient vector
- `H`: Hessian operator (a function that computes $H \cdot v$ for any vector $v$, rather than storing the full matrix — critical for large problems)
- `H⁻¹_approx`: an approximate inverse Hessian, used as a **preconditioner** for the CG solver

The initial trust radius is:

$$\Delta = \text{init\_scale\_radius} \times \|x_0\|$$

If $x_0 = 0$, it falls back to $\sqrt{n}$ where $n$ is the number of parameters.

A lower limit `Δ_limit = Δ × 10⁻¹⁰` is set. When the trust radius shrinks below this, the algorithm force-accepts the step (as an escape hatch to avoid infinite shrinking).

#### Main loop — each iteration (lines 39–175)

**Step 1: Evaluate objective and compute scaling**

```julia
v, dv = coleman_li_scaling_factors(x, g, LB, UB)
```

Returns scaling vector `v` and its derivative `dv` (see detailed description below).

**Step 2: Check convergence**

```julia
g_norm = norm(v .* g, Inf)
```

The infinity norm of the *scaled* gradient. This is zero at a constrained optimum because `v[i] = 0` for any parameter sitting on its active bound.

Convergence is declared if:
- `g_norm < tol`, or
- the cost function barely changed: $|f_{k-1} - f_k| < \text{tol} \times \max(1, |f_k|)$

**Step 3: Build scaled problem**

```julia
D = sqrt.(v)                           # Scaling factors
ĝ = D .* g                             # Scaled gradient
C = dv .* g                            # Coleman-Li derivative correction
H_scaled = x -> (D .* (H * (D .* x))) + (C .* x)   # Scaled Hessian operator
D⁻¹ = map(d -> d == 0 ? 0 : inv(d), D)             # Safe inverse (0→0, not 0→Inf)
```

The scaled Hessian operator is `H_scaled(x) = D .* (H * (D .* x)) + C .* x`. It has two parts:

1. **The similarity transform** $D \cdot H \cdot D$: this is the straightforward scaling of the Hessian into the new coordinate system defined by $D$.

2. **The Coleman-Li correction** $C \cdot x$: this term exists because of a chicken-and-egg problem. The scaling $D$ depends on the current point $x$ (recall $D = \sqrt{v}$ where $v$ measures distance to bounds). Ideally, after taking a step $s$, we'd want the scaling at the new point $D(x+s)$. But we can't compute $D(x+s)$ because **we don't know $s$ yet** — finding $s$ is exactly what the subproblem is solving for! And if we made $D$ depend on $s$, the subproblem would become nonlinear and we could no longer use CG to solve it.

    The solution is to use $D(x)$ (computed at the current point) but add a **first-order Taylor correction** that approximates how $D$ would change as we step. Mathematically, when you differentiate the scaled gradient $\hat{g} = D(x) \cdot g(x)$ with respect to $x$, the product rule gives an extra term involving $\frac{\partial D}{\partial x}$. That extra term simplifies to $dv \cdot g = C$, applied elementwise to the step. This keeps the subproblem quadratic (solvable by CG) while accounting for the fact that the scaling changes as we move — which matters most near bounds where $D$ changes rapidly with $x$.

**Step 4: Solve subproblem with Steihaug CG**

The subproblem $\min_{\hat{s}} \hat{g}^T \hat{s} + \frac{1}{2} \hat{s}^T \hat{H} \hat{s}$ subject to $\|\hat{s}\| \leq \Delta$ is solved using the **Steihaug conjugate gradient** method. This is an **inexact** solver: it does not solve the underlying linear system to full precision, but instead iterates only until the solution would leave the trust region, at which point it stops and returns the best step found so far. This is desirable because spending effort on a highly precise subproblem solution is wasteful when the trust region limits the step anyway.

Critically, Steihaug CG only uses **matrix-vector products** with the Hessian (i.e., computing $\hat{H} \cdot v$ for vectors $v$). It never needs to store, factorize, or invert the Hessian matrix. This is what makes the algorithm suitable for large-scale problems where the Hessian is too large to fit in memory. See the detailed description of Steihaug in the section below.

```julia
P = y -> D⁻¹ .* (H⁻¹_approx * (D⁻¹ .* y))   # Preconditioner
steps = steihaug_store_steps(H_scaled, ĝ, Δ, P, ...)
ŝ = steps[end]                                   # Best step in scaled space
s = D .* ŝ                                        # Unscale back to original space
```

If you have domain knowledge that lets you build a cheap approximation to the inverse Hessian (e.g., a diagonal approximation of $(J^T J)^{-1}$ based on known parameter sensitivities), you can supply it as `H⁻¹_approx` and it will be used as a **preconditioner** `P` for the CG iterations. This can speed up convergence of the inner loop. If no good approximation is available, the identity can be used (i.e., no preconditioning).

All intermediate CG steps are stored. If the step is rejected later, the algorithm can backtrack to a shorter intermediate step instead of re-running CG from scratch.

**Step 5: Choose the best feasible step**

```julia
theta = max(0.995, 1 - norm(v .* g, Inf))
step, step_hat, step_value = choose_step(x, H_scaled, ĝ, s, ŝ, D, Δ, theta, LB, UB)
```

`theta` is a safety factor (between 0.995 and 1.0) that **scales down candidate steps that would otherwise reach all the way to a constraint boundary**. Specifically, when computing the reflected Newton step and the steepest descent step, the algorithm first finds how far it *could* step before hitting a bound, then multiplies that maximum step length by `theta` to stay slightly inside the feasible region. This prevents numerical issues from landing exactly on a bound (where Coleman-Li scaling becomes zero).

When the scaled gradient norm is large (far from optimum), `theta ≈ 0.995` — a 0.5% safety margin from the boundary. When the gradient is small (near optimum), `theta` approaches 1.0, allowing the algorithm to get closer to bounds when it needs to for the final solution.

**Step 6: Evaluate and decide**

```julia
x_new = clamp.(x + step, LB, UB)    # Project onto feasible box
f_new, r_new = objective(x_new, "fr")
actual_reduction = f - f_new
ratio = _calculate_ratio(actual_reduction, g, H, s, ŝ, C, ...)
```

The step is accepted if:
- `actual_reduction > 0` and `ratio > 0.1`, OR
- the trust radius has shrunk below `Δ_limit` (escape hatch).

And it is *never* accepted if `f_new` is NaN (numerical failure).

If rejected, the trust radius is halved repeatedly, and the algorithm reuses an earlier, shorter CG step from the stored intermediate steps.

---

### `steihaug_store_steps` — Subproblem Solver (steihaug.jl)

This solves the **trust region subproblem**: minimize a quadratic function subject to $\|\hat{s}\| \leq \Delta$. It's Algorithm 7.2 from Nocedal & Wright's "Numerical Optimization."

The Steihaug method is a modified **conjugate gradient (CG)** solver. Standard CG solves the linear system $\hat{H} \hat{s} = -\hat{g}$ iteratively. Steihaug adds two extra rules:

1. **Trust region boundary**: if a CG step would leave the trust region, instead step to the trust boundary along the current direction and stop.
2. **Negative curvature**: if the Hessian has negative curvature along the current direction ($d^T H d < 0$), step to the trust boundary and stop. (This shouldn't happen with Gauss-Newton Hessians, but is handled for safety.)

#### The CG loop explained

```
Initialize: z = 0 (zero step), r = ĝ (residual = gradient)
            d = -P(r) (preconditioned steepest descent direction)

For each iteration:
    1. Compute Hd (Hessian times direction)
    2. Check if d'Hd < ε  →  negative curvature, step to trust boundary, stop
    3. Compute step size: α = (r'·P(r)) / (d'·H·d)
    4. New point: z_new = z + α·d
    5. Check if ‖z_new‖ > Δ  →  stepped outside trust region, project to boundary, stop
    6. New residual: r_new = r + α·Hd
    7. Check if ‖r_new‖ < tol  →  converged, stop
    8. Compute β for conjugacy: β = (P(r_new)'·r_new) / (P(r)'·r)
    9. New direction: d_new = -P(r_new) + β·d
    10. Store z_new as an intermediate step
```

If you have a cheap approximation to $\hat{H}^{-1}$ (see Step 4 in the main solver description above), it is used as the **preconditioner** $P$. This transforms the problem so the CG iteration sees a better-conditioned system and may converge in fewer iterations. If no good approximation is available, $P$ can simply be the identity (no preconditioning), and CG will still work — it may just take more iterations.

Every intermediate step is stored, so if the outer loop later decides the final step was too aggressive, it can pick an earlier, shorter step without re-solving.

#### Adaptive tolerance

```julia
η = min(tol, norm(g) / length(g))
tol = η * norm(g)
```

When the gradient is large (the algorithm is far from the solution), the subproblem doesn't need to be solved very accurately — a rough direction is fine. As the gradient shrinks near the solution, the tolerance tightens, giving more accurate subproblem solutions that enable faster final convergence. This is called the **Eisenstat-Walker strategy** and gives quadratic convergence near the solution.

---

### Utility Functions (utils.jl) — Detailed Reference

#### `coleman_li_scaling_factors(x, g, LB, UB)`

For each parameter $x_i$:

$$v_i = \begin{cases} \text{UB}_i - x_i & \text{if } g_i < 0 \text{ and } \text{UB}_i < \infty \\ x_i - \text{LB}_i & \text{if } g_i > 0 \text{ and } \text{LB}_i > -\infty \\ 1 & \text{otherwise} \end{cases}$$

The derivative `dv` is:
$$dv_i = \begin{cases} -1 & \text{if using upper bound} \\ +1 & \text{if using lower bound} \\ 0 & \text{otherwise} \end{cases}$$

The matrix $D = \text{diag}(\sqrt{v})$ defines a coordinate transformation that simultaneously:
- Slows parameters near their bounds (small $v_i$ → short steps)
- Leaves unconstrained parameters alone ($v_i = 1$)
- Freezes parameters *at* their bounds when the gradient pushes into the bound ($v_i = 0$) — they literally cannot move

#### `choose_step(x, Ĥ, ĝ, gn, gn_hat, D, Δ, theta, LB, UB)`

Selects the best step from three candidates by evaluating each with the quadratic model $q(\hat{s}) = \hat{g}^T \hat{s} + \frac{1}{2} \hat{s}^T \hat{H} \hat{s}$:

**Candidate 1: Full Newton step** (`compute_newton_step`)

Just check if $x + s$ (the Steihaug solution) stays within bounds. If yes, use it — it's the best local step.

**Candidate 2: Reflected Newton step** (`compute_reflected_step`)

When the Newton step would leave the feasible box:

1. Walk along the Newton direction until you hit a bound wall.
2. At the wall, **flip** ("reflect") the components of the step that hit the wall.
3. Continue along this reflected direction, but limit the distance by:
   - The trust region boundary
   - The next feasible boundary
4. Optimize the step length along this reflected direction using a 1D quadratic minimization.

Think of it as a billiard shot: the ball (parameter update) bounces off the constraint wall and continues in a modified direction. This often recovers most of the benefit of the Newton step while staying feasible.

**Candidate 3: Steepest descent step** (`compute_steepest_descent_step`)

The safest fallback: step in the negative scaled gradient direction $-\hat{g}$. The step length is optimized via 1D quadratic minimization, bounded by both the trust region and the feasible region (scaled by `theta`).

The algorithm evaluates all three candidates' quadratic model values and picks the one with the smallest (most negative) value — i.e., the one that predicts the most cost reduction.

#### `stepsize_to_bound_feasible_region(x, s, LB, UB)`

Given current position $x$ and direction $s$, find the smallest positive scalar $t$ such that $x + t \cdot s$ hits a bound. For each parameter $i$ with $s_i \neq 0$:

$$t_i = \max\!\left(\frac{\text{LB}_i - x_i}{s_i}, \; \frac{\text{UB}_i - x_i}{s_i}\right)$$

The `max` ensures we pick the relevant bound (the one the step is moving *toward*). The function returns the minimum over all parameters, plus which parameter(s) hit a bound.

#### `positive_stepsize_to_bound_trust_region(x, p, Δ)`

Find $\tau > 0$ such that $\|x + \tau p\| = \Delta$. This means solving:

$$\|p\|^2 \tau^2 + 2(x^T p)\tau + (\|x\|^2 - \Delta^2) = 0$$

Returns the positive root. Used in Steihaug CG when a step would leave the trust region — we project onto the trust boundary.

#### `stepsizes_to_bound_trust_region(x, s, Δ)`

Similar but returns *both* roots $t^-$ and $t^+$ (negative and positive intersection points). Uses the numerically stable formula from *Numerical Recipes* to avoid catastrophic cancellation when $b^2 \gg |4ac|$.

Given $a = s^T s$, $b = x^T s$, $c = x^T x - \Delta^2$:

$$q = -(b + |d| \cdot \text{sign}(b)), \quad d = \sqrt{b^2 - ac}$$
$$t_1 = q/a, \quad t_2 = c/q$$

#### `adjust_trust_radius(ratio, step, Δ)`

Updates $\Delta$ based on how well the quadratic model matched reality:

- $\rho < 0.25$: shrink to $\Delta = \frac{1}{4} \|\text{step}\|$ (but never to zero)
- $\rho > 0.75$ and $\|\text{step}\| > 0.95\Delta$: double to $\Delta = 2\Delta$
- Otherwise: keep $\Delta$ unchanged

The "step near trust boundary" condition for growth prevents unnecessary radius expansion when the step didn't need the full trust region.

#### `evaluate_quadratic(H, g, s)`

Evaluates $q(s) = g^T s + \frac{1}{2} s^T H s$. This is the quadratic model's prediction change at step $s$. Negative values mean predicted cost decrease.

#### `build_quadratic_1d(H, g, s, s0)` and `minimize_quadratic_1d(a, b, lb, ub, c)`

Used together to optimize step length along a line. Given a direction $s$ from starting point $s_0$:

$$q(t) = at^2 + bt + c$$

where $a = \frac{1}{2} s^T H s$, $b = g^T s + s^T H s_0$, $c = g^T s_0 + \frac{1}{2} s_0^T H s_0$.

The minimum is found by checking the boundary points `lb`, `ub` and the unconstrained optimum $t^* = -b/(2a)$ if it falls within bounds.

#### `_calculate_ratio(actual_reduction, g, H, s, ŝ, C, modified, to)`

Computes the ratio $\rho$ = actual improvement / predicted improvement.

$$\text{predicted} = -\left(g^T s + \frac{1}{2} s^T H s\right)$$

With `modified_reduction_for_ratio = true`, the actual reduction is adjusted:

$$\text{actual}_\text{mod} = (f - f_\text{new}) - \frac{1}{2} \hat{s}^T (C \odot \hat{s})$$

This accounts for the fact that the Coleman-Li scaling itself changes the model's predictions, giving a more reliable ratio near bounds.

#### `all_within_bounds(x, LB, UB, tol)`

Simple check: is every element of $x$ between its lower and upper bound (within numerical tolerance)?

#### `snap_to_bounds(x, LB, UB, tol)`

If a parameter has drifted slightly outside a bound due to floating-point arithmetic (e.g., $x_i = -1\text{e-}16$ when $\text{LB}_i = 0$), snap it back to the bound. Only snaps if the violation is smaller than `tol`.

---

### Summary Flow Diagram

```
trust_region_reflective(objective, x₀, LB, UB)
│
├─ Evaluate f, r, g, H at x₀
├─ Set initial Δ
│
└─ FOR each iteration:
    │
    ├─ (Re-)evaluate f, r, g, H at current x
    ├─ Compute Coleman-Li scaling (v, dv, D)
    ├─ Check convergence (scaled gradient & cost change)
    │
    ├─ Build scaled problem (ĝ, Ĥ_scaled)
    │
    ├─ WHILE step not accepted:
    │   │
    │   ├─ Solve subproblem via Steihaug CG → ŝ (in scaled space)
    │   │       └─ steihaug_store_steps()
    │   │           ├─ CG iterations with trust region constraint
    │   │           └─ Stores all intermediate steps for backtracking
    │   │
    │   ├─ Unscale: s = D · ŝ
    │   │
    │   ├─ Choose best step from 3 candidates
    │   │       └─ choose_step()
    │   │           ├─ Full Newton (if feasible)
    │   │           ├─ Reflected Newton (bounces off bounds)
    │   │           └─ Steepest descent (conservative fallback)
    │   │
    │   ├─ Evaluate f_new at x + step
    │   ├─ Compute ratio ρ = actual_reduction / predicted_reduction
    │   │
    │   ├─ IF ρ good enough → ACCEPT step, adjust Δ
    │   └─ ELSE → shrink Δ, try shorter intermediate CG step
    │
    └─ Continue to next iteration
```

---

### Differences from SciPy

This implementation follows SciPy's `scipy.optimize.least_squares` with `method='trf'` closely, including:

- Coleman-Li scaling with the derivative correction term
- The three-candidate step selection (Newton, reflected, steepest descent)
- The θ safety factor for bounds
- The trust radius update strategy

Key differences:
- Uses operator-form Hessian ($H$ is a function, not a matrix) — essential for large MRI problems
- Supports GPU arrays via CUDA.jl
- Stores intermediate Steihaug steps for efficient backtracking
- Uses a user-supplied preconditioner for the Steihaug CG method
