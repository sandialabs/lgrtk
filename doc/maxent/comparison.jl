# Assess success rate for resolving LME shape functions on benchmark nodes.

# Written by Avery Rock, <avery_rock@berkeley.edu>
# Written for Julia 1.4.2, Summer 2020

using Plots, Printf, LinearAlgebra, Dates, Random, Statistics
pyplot()

include("pmesh.jl")
include("lme.jl")
include("utils/utils.jl")
include("utils/colorutils.jl")
include("utils/geometry.jl")
include("solvers.jl")

"""
    data = successrate(p, x, h; kwargs)
Measure the fraction of successful LME shape function evaluations with varied solution schemes.
# Arguments
- `p::Array{Float}`: node locations.
- `x::Array{Float}`: material point locations
- `h::Float`
"""
function successrate(p, x, h; singular_methods = [pinvs], line = true, γ = 2 .^range(0, stop = 6, step = 1), ϵ = 1e-6, kmax = 1e8)
    nγ = length(γ)
    ns = length(singular_methods)
    solved = zeros(nγ, ns)
    for s ∈ 1:ns
        println(@sprintf("%s, %d points", string(singular_methods[s]), size(x, 1)))
        for i ∈ 1:nγ
            β = γ[i]/(h^2)
            N, DN, its = LME.shape_functions(p, x, β, singular = singular_methods[s], line = line, kmax = kmax)
            valid, z, f, g = LME.validate2(N, DN, p, x, ϵ = ϵ)
            solved[i, s] = count(valid) / length(valid)
        end
    end
    return solved
end

"""
    comparison2()
Compare solution success rates for LME solution methods in 2D and plot results.
No arguments, no returned values.
"""
function comparison2b()
    include("patterns/patterns.jl")

    γ = 2 .^range(3, stop = 6, length = 10)
    res = range(15, step = 5, length = 3)
    ϵ = 1e-8
    kmax = 1e14
    singular_methods = [direct, pinvs, babuska, gd, guess]

    nγ = length(γ)
    nr = length(res)
    ns = length(singular_methods)

    data1 = zeros(Float64, nγ, nr, ns)
    data2 = zeros(Float64, nγ, nr, ns)
    dof = zeros(length(res))

    for (r, i) in zip(res, 1:length(res))

        p = star2d.nodes
        h = star2d.h
        x = .95 * brick(minimum(p, dims = 1), maximum(p, dims = 1), (r, r))

        dof[i] = size(x, 1)
        data1[:, i, :] = successrate(p, x, h, ϵ = ϵ, singular_methods = singular_methods, γ = γ, kmax = kmax, line = false)
        data2[:, i, :] = successrate(p, x, h, ϵ = ϵ, singular_methods = singular_methods, γ = γ, kmax = kmax, line = true)
    end

    cs = colorarray(nr, ns)

    PyPlot.figure()
    PyPlot.suptitle("Solution Success Rate, 2D")
    PyPlot.subplot(1, 2, 1)
    for j ∈ 1:ns, i ∈ 1:nr
        PyPlot.semilogx(γ, 100*data1[:, i, j], "ko-", color = cs[i, j], label = @sprintf("%s, n = %d", string(singular_methods[j]), dof[i]))
    end

    PyPlot.title("Line Search Off")
    PyPlot.xlabel("γ")
    PyPlot.ylabel(@sprintf("%% Success, tol = %g", ϵ))
    PyPlot.legend(ncol = length(singular_methods))

    PyPlot.subplot(1, 2, 2)
    for j ∈ 1:ns, i ∈ 1:nr
        PyPlot.semilogx(γ, 100*data2[:, i, j], "ko-", color = cs[i, j], label = @sprintf("%s, n = %d", string(singular_methods[j]), dof[i]))
    end

    PyPlot.title("Line Search On")
    PyPlot.xlabel("γ")
    PyPlot.ylabel(@sprintf("%% Success, tol = %g", ϵ))
    PyPlot.legend(ncol = length(singular_methods))

    return data1, data2
end

"""
    comparison3()
Compare solution success rates for LME solution methods in 3D and plot results.
No arguments, no returned values.
"""
function comparison3b()
    include("/Users/averyrock/Documents/Sandia/diptera/patterns/patterns.jl")

    γ = 2 .^range(3, stop = 6, length = 10)
    res = range(20, step = 10, length = 3)
    ϵ = 1e-10
    kmax = 1e12
    singular_methods = [direct, pinvs, babuska, gd, guess]

    nγ = length(γ)
    nr = length(res)
    ns = length(singular_methods)

    data1 = zeros(Float64, nγ, nr, ns)
    data2 = zeros(Float64, nγ, nr, ns)
    dof = zeros(length(res))

    for (r, i) in zip(res, 1:length(res))
        p = star3d.nodes
        h = star3d.h
        x = .95 .* maximum(p) .* tetrand(r)
        dof[i] = size(x, 1)
        data1[:, i, :] = successrate(p, x, h, ϵ = ϵ, singular_methods = singular_methods, γ = γ, kmax = kmax, line = false)
        data2[:, i, :] = successrate(p, x, h, ϵ = ϵ, singular_methods = singular_methods, γ = γ, kmax = kmax, line = true)
    end

    cs = colorarray(nr, ns)

    PyPlot.figure()
    PyPlot.suptitle("Solution Success Rate, 3D")
    PyPlot.subplot(1, 2, 1)
    for j ∈ 1:ns, i ∈ 1:nr
        PyPlot.semilogx(γ, 100*data1[:, i, j], "ko-", color = cs[i, j], label = @sprintf("%s, n = %d", string(singular_methods[j]), dof[i]))
    end

    PyPlot.title("Line Search Off")
    PyPlot.xlabel("γ")
    PyPlot.ylabel(@sprintf("%% Success, tol = %g", ϵ))
    PyPlot.legend(ncol = length(singular_methods))

    PyPlot.subplot(1, 2, 2)
    for j ∈ 1:ns, i ∈ 1:nr
        PyPlot.semilogx(γ, 100*data2[:, i, j], "ko-", color = cs[i, j], label = @sprintf("%s, n = %d", string(singular_methods[j]), dof[i]))
    end

    PyPlot.title("Line Search On")
    PyPlot.xlabel("γ")
    PyPlot.ylabel(@sprintf("%% Success, tol = %g", ϵ))
    PyPlot.legend(ncol = length(singular_methods))

    return data1, data2
end

function comparison3r(; points = 100, reps = 5)
    include("/Users/averyrock/Documents/Sandia/diptera/patterns/patterns.jl")

    γ = range(4, stop = 64, step = 4)
    ϵ = 1e-10
    kmax = 1e12
    singular_methods = [direct, pinvs, babuska, gd, guess]

    nγ = length(γ)
    nr = reps
    ns = length(singular_methods)

    data1 = zeros(Float64, nγ, nr, ns)
    data2 = zeros(Float64, nγ, nr, ns)

    for i in 1:reps
        p = star3d.nodes
        h = star3d.h
        x = maximum(p) .* tetrand(points)
        data1[:, i, :] = successrate(p, x, h, ϵ = ϵ, singular_methods = singular_methods, γ = γ, kmax = kmax, line = false)
        data2[:, i, :] = successrate(p, x, h, ϵ = ϵ, singular_methods = singular_methods, γ = γ, kmax = kmax, line = true)
    end

    data1 *= 100
    data2 *= 100

    cs = colorinterp((1, .8, .1), (.8, .1, .9), ns)

    PyPlot.figure()
    PyPlot.suptitle(@sprintf("Solution Success Rate, 3D, n = %d", points))
    PyPlot.subplot(1, 2, 1)
    for j ∈ 1:ns
        m = mean(data1[:, :, j], dims = 2)[:]
        err = hcat(m - minimum(data1[:, :, j], dims = 2)[:],
        maximum(data1[:, :, j], dims = 2)[:] - m)'
        PyPlot.errorbar(γ, m, err, color = cs[j],
        label = string(singular_methods[j]))
    end

    PyPlot.title("Line Search Off")
    PyPlot.xlabel("γ")
    PyPlot.ylabel(@sprintf("%% Success, tol = %g", ϵ))
    PyPlot.legend(loc = 3)

    PyPlot.subplot(1, 2, 2)
    for j ∈ 1:ns
        m = mean(data2[:, :, j], dims = 2)[:]
        err = hcat(m - minimum(data2[:, :, j], dims = 2)[:],
        maximum(data2[:, :, j], dims = 2)[:] - m)'
        PyPlot.errorbar(γ, m, err, color = cs[j],
        label = string(singular_methods[j]))
    end

    PyPlot.title("Line Search On")
    PyPlot.xlabel("γ")
    PyPlot.ylabel(@sprintf("%% Success, tol = %g", ϵ))
    PyPlot.legend(loc = 3)

    return data1, data2
end

"""
    Use cubic symmetry to show failure locations for example node locations in 3D
    Visualize results with 3D scatter plot
"""
function failurestar3(;n = 50, reps = 1, γ = 8)
    timestamp("Starting block: ")
    include("patterns/patterns.jl")
    p = star3d.nodes
    x = .99 .* maximum(p).*tetmap(tetrand(n))
    xn = Array{Float64}(undef, 0, 3)

    ϵ = 1e-8
    h = 20
    β = γ/(h^2)

    for j in 1:reps
        if size(xn, 1) == 0
            N, DN, _ = LME.shape_functions(p, x, β)
            valid, _, _, _ = LME.validate2(N, DN, p, x, ϵ = ϵ)
            x = x[.!valid, :]
        else
            N, DN, _ = LME.shape_functions(p, xn, β)
            valid, _, _, _ = LME.validate2(N, DN, p, xn, ϵ = ϵ)
            xn = xn[.!valid, :]
        end
        x = vcat(x, xn)
        println(@sprintf("Iteration: %d, points found: %d", j, size(x, 1)))
        if size(x, 1) >= 1e4
            break
        end
        xn = convexcombination(x)
    end
    PyPlot.figure()
    PyPlot.title("Symmetric Node Arrangement")
    lmeplot(tetreflect(x))
    timestamp("Done: ")
    return x, p
end


"""
    Use cubic symmetry to show failure locations for example node locations in 3D
    Visualize results with 3D scatter plot
"""
function failurestar3_moving(;t = 0, n = 50, reps = 1, γ = 8)
    include("patterns/patterns.jl")
    p = star3d.nodes
    p = vcat(p, sin(t) * [10.0 10 0])
    x = .99 *brick(minimum(p, dims = 1), maximum(p, dims = 1), 10 .*(1,1,1))
    xn = Array{Float64}(undef, 0, 3)

    ϵ = 1e-12
    h = 20
    β = γ/(h^2)

    for j in 1:reps
        if size(xn, 1) == 0
            N, DN, _ = lme.shape_functions(p, x, β)
            valid, _, _, _ = lme.validate2(N, DN, p, x, ϵ = ϵ)
            x = x[.!valid, :]
        else
            N, DN, _ = lme.shape_functions(p, xn, β)
            valid, _, _, _ = lme.validate2(N, DN, p, xn, ϵ = ϵ)
            xn = xn[.!valid, :]
        end
        x = vcat(x, xn)
        println(@sprintf("Iteration: %d, points found: %d", j, size(x, 1)))
        if size(x, 1) >= 1e4
            break
        end
        xn = convexcombination(x)
    end
    PyPlot.figure()
    PyPlot.title("Symmetric Node Arrangement")
    lmeplot(x)
    timestamp("Done: ")
    return x, p
end
