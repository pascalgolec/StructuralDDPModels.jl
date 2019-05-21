@doc raw"""
	CooperHaltiwanger2006(;
        β = 0.9, # discount rate
        θ = 0.67, # returns to scale parameter
        ρ = 0.6, # autocorrelation of productivity
        σ = 0.3, # volatlity of productivity
        δ = 0.15, # capital depreciation rate
        γ = 2.0, # convex adjustment cost parameter
        F = 0.01, # fixed adjustment costs that scale with capital (F>=0)
        λ = 0.95, # fixed adjustment costs that scale with sales (λ<=1)
        p_b = 1., # unit price of capital purchases
        p_s = 1., # unit price of capital sales
        nK = 100, # number of nodes capital
        na = 5, # number of nodes for productivity
        ni = 50, # number of choices for investment rate
        mini = -0.5, # minimum possible investment rate
        maxi = 2.0) # maximum possible investment rate

Model of [Cooper and Haltiwanger 2006 RES](https://doi.org/10.1111/j.1467-937X.2006.00389.x). The recursive formulation is:

```math
\begin{equation*}
V(K,a) = \max_I K^\theta e^a (1-\mathbb{1}_{I\neq0}\lambda) - p_i I - \mathbb{1}_{I\neq0} F K - \frac{\gamma}{2} I^2 K + \beta \mathbb{E} V(K', a') \\
p_i = \begin{cases} p_b & \mbox{if } I \geq 0 \\
p_s & \mbox{if } I < 0 \end{cases} \\
\text{where } K' = (1-\delta) K + I \\
a' = \rho a + \sigma \varepsilon, \quad \varepsilon \sim \mathcal{N}(0,1)
\end{equation*}
```


Example:

```julia
using StructuralDDP, StructuralDDPModels
prob = CooperHaltiwanger2006(nK=150, na=8, ni=100, ρ=0.5, σ=0.3, γ=2.)
```

"""
function CooperHaltiwanger2006(;
    β = 0.9, # discount rate
    θ = 0.67, # returns to scale parameter
    ρ = 0.6,
    σ = 0.3,
    δ = 0.15, # capital depreciation rate
    γ = 2.0, # convex adjustment cost parameter
    F = 0.01, # fixed adjustment costs that scale with capital (F>=0)
    λ = 0.95, # fixed adjustment costs that scale with sales (λ<=1)
    p_b = 1., # cost of capital purchases
    p_s = 1., # cost of capital sales
    nK = 100,
    na = 5,
    ni = 50) # number of choices for investment rate

    stda = sqrt(σ^2/(1-ρ^2))
    mina = -3*stda
    maxa =  3*stda
    va = collect(LinRange(mina, maxa, na))

    function steadystate(z)
        function f!(F,x)
            (V_K, K_log) = x

            F[1] = - 1 - γ + V_K
            F[2] = V_K - β*( exp(z)*θ*exp(K_log)^(θ-1) + (1-δ)*V_K)
        end
        nlsolve(f!, [1.1; 1.])
    end

    K_ss = exp(steadystate(stda^2/2).zero[2])
    minK_log = steadystate(-2*stda).zero[2]
    maxK_log = steadystate(3*stda).zero[2]
    vK   = exp.(collect(LinRange(minK_log, maxK_log, nK)))

    tStateVectors = (vK, va)

    vi = collect(LinRange(-0.5, 2., ni))
    iszero = vi .== 0.
    if !any(iszero)
        vi = collect(LinRange(-0.5, 2., ni-1))
        vi = [0., vi...]
        sort!(vi)
    end
    tChoiceVector = (vi,)

    function reward(vStates, i)
        K, a = vStates
        action = i != 0.
        if i > 0.
            price = p_b
        else
            price = p_s
        end
        K^θ * exp(a)*(1-λ*action) - i*K*price - F*K*action - γ/2 * i^2 * K
    end

    function transition(vStates, i, ε)
        K, a = vStates
        Kprime = (1-δ+i)*K
        aprime  = ρ*a + σ * ε
        return Kprime, aprime
    end

    DiscreteDynamicProblem(
            tStateVectors,
            tChoiceVector,
            reward,
            transition,
            Normal(),
            β;
            intdim = :All) # can not separate because of fixed adjustment costs
            # and discrete choices
end
