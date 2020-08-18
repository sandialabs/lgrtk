# Node placement, sampling, and manipulation functions.

# Written by Avery Rock, <avery_rock@berkeley.edu>
# Written for Julia 1.4.2, Summer 2020

using Plots, PyCall, Random, Combinatorics
pyplot()

"""
    A = brick(min, max, res)
returns n-dimensional grid of points with lower bound min, upper bound max, and number of points res
"""
function brick(lower, upper, res)
    b = zeros(prod(res), length(res))
    for (l, u, r, i) in zip(lower, upper, res, 1:length(res))
        b[:, i] = repeat(range(l, stop = u, length = r), inner = prod(res[1:i-1]), outer = prod(res[i+1:end]))
    end
    return b
end

"""
    X = tetgrid(n)
Return coordinates X = [x y z] such that 1 ≧ x, y, z ≥ 0, x + y + z ≦ 1
x, y, z ∈ range(0, stop = 1, length = n),
where n is the next-lowest tetrahedral number to p
"""
function tetgrid(p)
    X = zeros(tetnumber(n), 3)
    h = 1/(n - 1)
    i = 0
    for x ∈ range(0, stop = 1, step = h)
        for y ∈ range(0, stop = 1 - x, step = h)
            for z ∈ range(0, stop = 1 - x - y, step = h)
                i += 1
                X[i, :] = [x, y, z]
            end
        end
    end
    return X
end

"""
    X = tetrand(n)
Return coordinates X = [x y z] such that 1 ≧ x ≧ y ≧ z ≥ 0,
x, y, z ∈ range(0, stop = 1, length = n)
Represents points in a tetrahendron comprising
1/48th of a cube with -1 ≦ x, y, z ≦ 1 with p total points
Using method from: http://vcg.isti.cnr.it/jgt/tetra.htm
"""
function tetrand(p)
    p = Int(p)
    X = zeros(p, 3)
    Threads.@threads for d ∈ 1:p
        s, t, u = rand(3)
        if s + t + u > 1
            if s + t > 1
                s, t, u = 1 - s, 1 - t, u
            end
            if s + t + u > 1
                if t + u > 1
                    s, t, u = s, 1 - u, 1 - s - t
                else
                    s, t, u = 1 - t - u, t, s + t + u - 1
                end
            end
        end
        X[d, :] = [s, t, u]
    end
    return X
end

"""
    Tn = tetnumber(n)
Return the n'th tetrahedral number
"""
function tetnumber(n)
    return Int((n * (n + 1) * (n + 2)) / 6)
end

"""
    X = tetreflect(x)
undo symmetry reduction from threeDsym to make a full cube
will duplicate points on the iterior surfaces of the subtet
"""
function tetreflect(x)
    n = size(x, 1)
    X = zeros(48*n, 3) # preallocate
    for (i, p) ∈ zip(1:6, permutations(1:3) |> collect)
        X[(i - 1) * n + 1: i*n, :] = x[:, p]
    end
    for (j, m) ∈ zip(2:8, corners(3, limits = [1, -1])[2:8])
        X[6(j-1)*n + 1 : 6j*n , :] = m' .*X[1:6n, :]
    end
    return X
end

"""
    c = corners(d)
returns the corners of a d-space cube with vertices
at positions defined by limits. Assumed to occupy same interval in all dimensions.
"""
function corners(d; limits = [0 1])
    limits = reshape(limits, (2,))
    c = zeros(2^d, d)
    res = tuple(repeat([2], d)...)
    Threads.@threads for i ∈ 1:d
        inner = prod(res[1:i - 1])
        outer = prod(res[i + 1:end])
        c[:, i] = repeat(limits, inner = inner, outer = outer)
    end
    return [c[i, :] for i in 1:2^d]
end

"""
    Map 'reference tetrahedron' to have corners at origin and points defined by "corners"
"""
function tetmap(X; corners = [1 0 0; 1 1 0; 1 1 1], origin = [0 0 0])
    return X * corners .+ origin
end

"""
    Plot discrete points in 3d space to show regions where LME solution scheme struggles
"""
function lmeplot(x; camera = (30, 30))
    Plots.plot!(x[:, 1], x[:, 2], x[:, 3], markeralpha = .4,
    line = nothing, markershape = :x, markersize = 2,
    markercolor = :red, markestrokecolor = :red, camera = camera,
    legend = false)
end

"""
    Create random convex combinations of pairs of points from array x.
Returns as many points as are in x.
"""
function convexcombination(x)
    ϕ = rand(size(x, 1))
    return ( ϕ .* x + (1 .- ϕ) .* x[Random.randperm(size(x, 1)), :])
end

"""
    Return, at most, nmax rows of array x
"""
function subsample(x, nmax = 1e5)
    nmax = Int(nmax)
    if nmax < size(x, 1)
        return x[randperm(size(x, 1))[1:nmax], :]
    else
        return x
    end
end
