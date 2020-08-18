# Test LME implementation for numerical correctness.

# Written by Avery Rock, <avery_rock@berkeley.edu>
# Written for Julia 1.4.2, Summer 2020

using PyPlot, PyCall, Printf, LinearAlgebra

include("pmesh.jl")
include("lme.jl")
include("utils/utils.jl")

tests = []

## 1D tests ##

nf = 5
p = range(0, stop = 1, length = nf)
xp = range(0, stop = 1, length = 1000)

h = norm(p[1, :] - p[2, :])
γ = 1.0
β = γ/(h^2)

N, DN = LME.shape_functions(p, xp, β)
push!(tests, LME.validate(N, DN, p, xp))

h = norm(p[1, :] - p[2, :])
γ = 8*.5 .^[0:4; 3:-1:0]
β = γ./(h^2)
p = range(0, stop = 1, length = length(β))

for i in 1:length(β)
    N, DN = LME.shape_functions(p, xp, β[i])
    push!(tests, LME.validate(N, DN, p, xp))
end

## 2D tests ##

p = [-20.0 0; -5 5; 0 20; 5 5; 20 0; 5 -5; 0 -20; -5 -5] # Wriggers star pattern
# pv = [-20.0 20; 20 20; 20 -20; -20 -20; -20 20] # outer hull
pv = copy(p)
# pv = [-30.0 30; 30 30; 30 -30; -30 30] # outer hull, use symmetry and move beyond support domain

xp, xt, _, _, _ = Mesh.mesh(pv, 2.0, 3)

γ = range(1, step = 1, stop = 10)
h = 20 # Wriggers spacing
a = 2 # shape function to plot

for i = 1:length(γ)
    β = γ[i]/(h^2)
    N, DN = LME.shape_functions(p, xp, β)
    push!(tests, LME.validate(N, DN, p, xp))
    tests[end] || println(@sprintf("Failed for γ = %.2f", γ[i]))
end

## 3D

γ = range(0, step = 1, stop = 6)
β = γ./(h^2)
p = [0 0 0 ; 0 0 1.0; 0 1 0; 0 1 1; 1 0 0; 1 0 1; 1 1 0; 1 1 1]
# p = [0 0 0 ; 0 0 1; 0 1.0 0; 1 1 1]
h = norm(p[1, :] - p[2, :])
a = range(0, stop = 1, length = 10)

xp = zeros(length(a)^3, 3)
i = 0
for xx = a, yy = a, zz = a
    global i += 1
    xp[i, :] = [xx, yy, zz]
end

for b in β
    N, DN = LME.shape_functions(p, xp, b)
    push!(tests, LME.validate(N, DN, p, xp))
end

## 4D (for generality)

a = 0:1
h = 1
γ = range(0, step = 1, stop = 6)
β = γ./(h^2)
p = zeros(length(a)^4, 4)
i = 0
for xx = a, yy = a, zz = a, qq = a
    global i += 1
    p[i, :] = [xx, yy, zz, qq]
end

a = range(0, stop = 1, length = 10)
xp = zeros(length(a)^4, 4)

i = 0
for xx = a, yy = a, zz = a, qq = a
    global i += 1
    xp[i, :] = [xx, yy, zz, qq]
end

for b in β
    N, DN = LME.shape_functions(p, xp, b)
    push!(tests, LME.validate(N, DN, p, xp))
end

## Postprocessing
println()
println(@sprintf("Passed %d of %d tests.", count(tests), length(tests)))
