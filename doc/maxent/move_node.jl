# Figures to visualize changing node location in 2D.

include("pmesh.jl")
include("lme.jl")
include("utils/utils.jl")

using PyPlot, PyCall, Printf, LinearAlgebra

function movenode(s, a)
    p = hm.p
    t = hm.t
    d = [cos(s); sin(s)]
    p[a, :] += d

    _, c, _ = Mesh.tristats(p, t)
    xp, _ = Mesh.triquad(p, t, c)
    xp = vcat(xp, p)
    tp = Mesh.delaunay(xp)

    xp, tp = Mesh.trefine(xp, tp, reps = 5)

    γ = 1.0
    h = 1/2.2
    β = γ/(h^2)
    N, DN = lme.shape_functions(p, xp, β)

    a = 5
    tp = Mesh.delaunay(xp)

    figure()
    Mesh.tplot(p, t)
    plot(p[:, 1], p[:, 2], "ko")
    plot(xp[:, 1], xp[:, 2], "k.")
    title("Original Mesh")

    figure()
    Mesh.tplot(xp, tp, u = N[a, :])
    colorbar(ticks = range(0, step = .1, stop = 1))
    # plot(xp[:, 1], xp[:, 2], "k.", label = "material points", ms = 1)
    plot(p[:, 1], p[:, 2], "mo", label = "nodes")
    title(@sprintf("LME Shape Functions, γ = %.2f", γ))
    legend()


    figure()
    subplot(1, 2, 1)
    Mesh.tplot(xp, tp, u = DN[a, 1, :], shownan = false)
    colorbar()
    # plot(xp[:, 1], xp[:, 2], "k.", label = "material points", ms = 1)
    plot(p[:, 1], p[:, 2], "mo", label = "nodes")
    title(@sprintf("δN/δx, γ = %.2f", γ))
    legend()

    subplot(1, 2, 2)
    Mesh.tplot(xp, tp, u = DN[a, 2, :], shownan = false)
    colorbar()
    # plot(xp[:, 1], xp[:, 2], "k.", label = "material points", ms = 1)
    plot(p[:, 1], p[:, 2], "mo", label = "nodes")
    title(@sprintf("δN/δy, γ = %.2f", γ))
    legend()

    tests = lme.validate(N, DN, p, xp)
    println(tests)
end

for t in range(0, stop = 2*pi, length = 10)
    movenode(t, 5)
end
