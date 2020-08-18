# Local Max. Entropy chape function methods.

# Written by Avery Rock, <avery_rock@berkeley.edu>
# Written for Julia 1.4.2, Summer 2020

module LME
using PyPlot, PyCall, LinearAlgebra, Random, Printf
include("solvers.jl")
include("utils/utils.jl")

"""
    shape_functions(xa, xp, β)
Construct local maximum entropy shape functions and derivatives.
# Arguments:
- `λ::Array{Float64}`: lagrange multiplier (current)
- `xa::Array{Float64}`: nodal points for functions
- `xp::Float64`: point to evaluate.
- `β::Float64`: locality weight
"""
function shape_functions(xa, xp, β; singular = direct, line = false, kmax = 1e12)
    np = size(xp, 1) # number of material points
    nn = size(xa, 1) # number of nodes
    dim = size(xa, 2) # spatial dimensions
    NN = zeros(nn, np) # shape function values
    dNN = zeros(nn, dim, np) # shape function gradient
    iters = zeros(np) # number of iterations per point

    Threads.@threads for p ∈ 1:np
        # println(Threads.threadid())
        # for p ∈ 1:np
        x = xp[p, :]
        N, dN, λ, its = multiplier(x, xa, β, singular = singular, line = line, kmax = kmax)
        NN[:, p] = N
        dNN[:, :, p] = dN
        iters[p] = its
    end
    return NN, dNN, iters
end

"""
    residual(λ, x, xa, β)

Use with multiplier() to solve for shape functions and derivatives.
Calculate value and properties of max ent shape functions.

# Arguments:
- `λ::Array{Float64}`: lagrange multiplier (current)
- `x::Float64`: point to evaluate.
- `xa::Array{Float64}`: nodal points for functions
- `β::Float64`: locality weight
"""
function residual(λ, x, xa, β)

    N = size(xa, 1)
    dim = size(xa, 2)
    DDF = zeros(dim, dim)
    DF = zeros(dim)
    F = 0
    p = zeros(N)
    dp = zeros(N, dim)
    r = zeros(dim) # bold r

    Z = 0
    for a ∈ 1:N
        dx = xa[a, :] - x
        Za = exp(- β * norm(dx)^2 - dot(λ, dx))
        p[a] = Za
        Z += Za
        r += p[a] * dx # Li 2.48
        DDF -= p[a]*dx*dx'
    end

    p /= Z

    J = -r*r' # Li 2.51
    DF = r

    for a = 1:N
        dx = x - xa[a, :]
        J += p[a] * (dx * dx') # Li 2.51
    end

    for a = 1:N
        try
        dp[a, :] = -p[a] * (J\(x - xa[a, :]))
    catch SingularException
        dp[a, :] = -p[a] * (pinv(J)*(x - xa[a, :]))
    end
    end

    F = .5 * r' * r
    return F, DF, DDF, p, dp
end

"""
    N, DN, λ, iters = multiplier(λ, x, xa, β[, max_iterations, tol])

Use Newton-Raphson method to compute the lagrange multiplier for LME shape functions.
Return the shape functions and their gradients as auxiliary outputs.
"""
function multiplier(x, xa, β; λ = nothing, singular = direct, line = false, maxit = 50, tol = 1e3eps(), kmax = 1e12)
    if λ == nothing
        λ = zeros(size(xa, 2))
    end
    i = 0
    dims = size(x, 1)
    while true
        i +=1
        F, DF, DDF, N, DN = residual(λ, x, xa, β) # "reusing" information versus only calculating what you need at any given time?
        if norm(DF) < tol || i >= maxit # can add more convergence information and cases if needed.
            return N, DN, λ, i
        end

        if cond(DDF) > kmax
            dλ = -singular(DDF, DF)
        else
            dλ = -direct(DDF, DF)
        end

        if line
            f(a) = residual(a, x, xa, β)[1]
            dλ = linesearch(f, DF, λ, dλ)
        end
        λ += dλ
    end
end

"""
    N, DN, λ, iters = multiplier2(λ, x, xa, β[, max_iterations, tol])

Use Newton-Raphson, Nelder Mead, and bisection to resolve lagrange multiplier
"""
function neldermead(x, xa, β; λ = nothing, singular = direct, line = false, maxit = 25, tol = 1e2eps(), kmax = 1e12)
    dims = size(x, 2)
    history = zeros(maxit, dims) # record lagrange multipliers
    f_history = zeros(maxit)
    simplex = zeros(dims + 1, dims) # simplex for Nelder-Mead
    simplex[1:dims, :] = eye(dims)
    simplex_objective = zeros(dims + 1)

    f(a) = residual(a, x, xa, β)[1]

    if λ == nothing
        λ = zeros(size(xa, 2))
    end
    i = 0
    dims = size(x, 1)

    for j ∈ 1:(dims + 1)
        simplex_objective[j] = f(simplex[j, :])
    end

    while true
        p = sortperm(simplex_objective)
        simplex_objective = simplex_objective[p]
        simplex = simplex[p, :]
        xo = mean(simplex[1:dims], dims = 1) # centroid
        i += 1
        I, DF, DDF, N, DN = residual(λ, x, xa, β) # "reusing" information versus only calculating what you need at any given time?
        # println("‖∇F‖: ", norm(DF))
        if norm(DF) < tol || i >= maxit # can add more convergence information and cases if needed.
            return N, DN, λ, i
        end

        error("unimplemented") # finish for future
    end
end

"""
    tests = validate(N, x, ϵ)
Check shape function array N for zeroth- and first-order consistency.
Return boolean for ALL tests.
"""
function validate(N, DN, xa, xm; ϵ = 1e-6)

    t1 = nanabs.(sum(N, dims = 1) .-1)
    t2 = nanabs.(N' * xa - xm )
    t3 = abs(nansum(DN))/size(xm, 1)
    zeroth = all(t1 .< ϵ)
    first = all(t2 .< ϵ)
    grad = t3 < ϵ

    zeroth || println("Shape functions are not a partition of unity. Largest error: "*string(maximum(t1)))
    first || println("Shape functions do not satisfy first-order consistency. Largest error: "*string(maximum(t2)))
    grad || println("Gradients inconsistent with partition of unity. Average error: "*string(t3))

    return zeroth && first && grad
end

"""
    zeroth, first, grad = validate2(N, x, ϵ)
Check shape function array N for zeroth- and first-order consistency.
Return boolean array for each test. Can be used to index.
"""
function validate2(N, DN, xa, xm; ϵ = 1e-6)
    t1 = abs.(sum(N, dims = 1) .-1)
    t2 = sum(abs.(N' * xa - xm ), dims = 2)
    t3 = abs.(sum(DN, dims = (1, 2))) # can potentially break down by direction

    zeroth = t1 .< ϵ
    first = t2 .< ϵ
    grad = t3 .< ϵ

    return all([zeroth[:] first[:] grad[:]], dims = 2)[:], t1[:], t2[:], t3[:]
end
end # end lme
