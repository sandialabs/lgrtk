# Example graphical outputs and tests for LME shape functions.

# Written by Avery Rock, <avery_rock@berkeley.edu>
# Written for Julia 1.4.2, Summer 2020

using Plots, PyPlot, PyCall, Printf, LinearAlgebra

include("pmesh.jl")
include("lme.jl")
include("utils/utils.jl")
include("utils/colorutils.jl")
include("solvers.jl")

@userplot Cloud # point cloud structure for failure visuals
@recipe function f(c::Cloud)
    x, i = c.args
    n = length(x)
    ticks --> false
    grid --> false
    dpi --> 80
    aspect_ratio --> 1
    label --> false
    x[:, 1], x[:, 2], x[:, 3]
end

"""
    oneDSweep()
Show LME shape functions with example node placement in 1D for
a range of γ values. Figure used in August 2020 presentation.
Shows values and derivatives.
"""
function oneDSweep()
    p = range(0, stop = 1, length = 4)
    xp = range(0, stop = 1, length = Int(2e5))
    h = norm(p[1, :] - p[2, :])
    γ = 16*.5 .^(5:-1:0)
    β = γ./(h^2)

    cs = colorinterp((.2, .2, 1), (1, .4, .2), length(β)) # colormap

    PyPlot.figure()
    suptitle("LME Functions with Varied Locality γ")
    for i ∈ 1:length(β)
        N, DN, its = LME.shape_functions(p, xp, β[i])
        mask = LME.validate2(N, DN, p, xp, ϵ = 1e-6)[1]

        N[:, .!mask] .= NaN
        DN[: , :, .!mask] .= NaN
        subplot(length(β), 2, 2*i - 1)

        if i == 1
            title("N")
        end
        a = 2
        PyPlot.plot(p, 0*p, "k.")
        PyPlot.plot(xp, N[a, :]', color = cs[i])
        PyPlot.plot(xp, N[:, :]', color = cs[i], alpha = .25)
        ylabel("γ = "*string(γ[i]), rotation = "vertical")
        subplot(length(β), 2, 2*i)
        if i == 1
            title("dN/dx")
        end
        PyPlot.plot(p, 0*p, "k.")
        PyPlot.plot(xp, DN[a, 1, :]', color = cs[i])
        PyPlot.plot(xp, DN[:, 1, :]', color = cs[i], alpha = .25)
    end
end

"""
    twoDSweep()
Show LME shape functions with example node placement in 2D for
a range of γ values.
Shows values, both gradient components, and iterations used.
"""
function twoDSweep()
p = [-20.0 0; -5 5; 0 20; 5 5; 20 0; 5 -5; 0 -20; -5 -5] # Wriggers star pattern
pv = [-20.0 0; 0 20; 20 0; 0 -20; -20 0]

ns = 3 # number of scales
nγ = 3
γ = range(0, stop = 32.0, length = nγ)
sample = zeros(ns)

h = 20 # Wriggers spacing for star pattern.
a = 2 # shape function to plot

xp, xt, _, _, _ = Mesh.mesh(pv, 1, 1)

for i ∈ 1:nγ

    β = γ[i]/(h^2)
    N, DN, its = LME.shape_functions(p, xp, β)
    mask = LME.validate2(N, DN, p, xp, ϵ = 1e-10)[1]
    N[:, .!mask] .= NaN
    DN[: , :, .!mask] .= NaN

    PyPlot.figure()
    PyPlot.suptitle(@sprintf("LME Shape Functions, γ = %.2f", γ[i]))

    PyPlot.subplot(2, 2, 1)
    Mesh.tplot(xp, xt, u = N[a, :])
    PyPlot.colorbar(ticks = range(0, step = .1, stop = 1))
    PyPlot.plot([p[:, 1]; p[1, 1]], [p[:, 2]; p[1, 2]], "mo--", ms = 10, label = "nodes")
    PyPlot.title("N")
    PyPlot.legend()

    PyPlot.subplot(2, 2, 2)
    Mesh.tplot(xp, xt, u = its, shownan = false)
    PyPlot.colorbar(ticks = range(minimum(its), stop = maximum(its), step = 1))
    PyPlot.title("Iterations")

    PyPlot.subplot(2, 2, 3)
    Mesh.tplot(xp, xt, u = DN[a, 1, :], shownan = false)
    PyPlot.colorbar()
    PyPlot.plot([p[:, 1]; p[1, 1]], [p[:, 2]; p[1, 2]], "mo--", ms = 10, label = "nodes")
    PyPlot.title("δN/δx")

    PyPlot.subplot(2, 2, 4)
    Mesh.tplot(xp, xt, u = DN[a, 2, :], shownan = false)
    PyPlot.colorbar()
    PyPlot.plot([p[:, 1]; p[1, 1]], [p[:, 2]; p[1, 2]], "mo--", ms = 10, label = "nodes")
    PyPlot.title("δN/δy")
end
end
"""
    threeDSweep()
Produce animations of point clouds of failure locations. Used for presentation Aug. 2020
"""
function threeDSweep(; n = 90, points = Int(1e3))
    for γ ∈ range(12, stop = 18, length = 3)
        x, p = failurestar3(n = points, γ = γ)
        anim = @animate for i ∈ range(0, stop = 90 - 90/n, length = n)
            cloud(p, i, camera = (i, 22.5), markershape = :circle, markersize = 5,
            line = nothing, markercolor = :cyan, markerstrokecolor = :black,
            legend = false, cbar = false, framestyle = :none)
            cloud!(tetreflect(x), i, camera = (i, 22.5), markeralpha = .5,
            line = nothing, markershape = :circle, markersize = 1,
            markercolor = :red, markerstrokecolor = :red,
            legend = false, cbar = false, framestyle = :none)
        end
        gif(anim, @sprintf("%d_%d_rotate.gif", γ, points), fps = n÷10)
    end
end
