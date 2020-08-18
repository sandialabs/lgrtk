include("pmesh.jl")
include("patterns/circle.jl")
include("lme.jl")
include("utils/utils.jl")
include("solvers.jl")

using PyPlot, PyCall, Printf, LinearAlgebra

function matlab_comparison(; γ = 8.0, a = 5)
    p = hm.p
    t = hm.t

    _, c, _ = Mesh.tristats(p, t)
    xp, _ = Mesh.triquad(p, t, c)
    xp = vcat(xp, p)
    tp = Mesh.delaunay(xp)

    xp, tp = Mesh.trefine(xp, tp, reps = 2)

    h = 1/2.2
    β = γ/(h^2)
    N, DN = LME.shape_functions(p, xp, β)
    mask = LME.validate2(N, DN, p, xp, ϵ = 1e-10)[1]

    N[:, .!mask] .= NaN
    DN[: , :, .!mask] .= NaN

    figure()
    suptitle(@sprintf("LME Shape Functions, γ = %.2f", γ))
    PyPlot.subplot(2, 2, 1)
    Mesh.tplot(p, t)
    PyPlot.plot(p[:, 1], p[:, 2], "ko")
    PyPlot.plot(xp[:, 1], xp[:, 2], "k.")
    title("Original Mesh")

    subplot(2, 2, 2)
    Mesh.tplot(xp, tp, u = N[a, :], limits = (0, 1))
    colorbar(ticks = range(0, step = .1, stop = 1))
    # plot(xp[:, 1], xp[:, 2], "k.", label = "material points", ms = 1)
    PyPlot.plot(p[:, 1], p[:, 2], "mo", label = "nodes")
    title("N")
    legend()

    subplot(2, 2, 3)
    u = DN[a, 1, :]
    Mesh.tplot(xp, tp, u = u, shownan = false)
    colorbar(ticks = range(floor(nanmin(u)), stop = ceil(nanmax(u)), step = 1))
    # plot(xp[:, 1], xp[:, 2], "k.", label = "material points", ms = 1)
    PyPlot.plot(p[:, 1], p[:, 2], "mo", label = "nodes")
    title("δN/δx")
    legend()

    subplot(2, 2, 4)
    u = DN[a, 2, :]
    Mesh.tplot(xp, tp, u = u, shownan = false)
    colorbar(ticks = range(floor(nanmin(u)), stop = ceil(nanmax(u)), step = 1))
    PyPlot.plot(xp[:, 1], xp[:, 2], "k.", label = "material points", ms = 1)
    PyPlot.plot(p[:, 1], p[:, 2], "mo", label = "nodes")
    title("δN/δy")
    legend()

    show()
end
