# Mesh utilities for unstructured triangular meshes.

# Original code: UC Berkeley Math 228B, Per-Olof Persson <persson@berkeley.edu>
# Completed by Avery Rock <avery_rock@berkeley.edu> for coursework, Fall 2018.
# Modified for Julia 1.4.2, Summer 2020

module Mesh
using PyPlot, PyCall, LinearAlgebra
include("utils/utils.jl")

"""
    t = delaunay(p)

Delaunay triangulation `t` of N x 2 node array `p`.
"""
function delaunay(p)
    tri = pyimport("matplotlib.tri")
    t = tri.Triangulation(p[:,1], p[:,2])
    return Int64.(t.triangles .+ 1)
end

"""
    edges, boundary_indices, emap = all_edges(t)
Identify and describe edges of the triangulation `t`
`edges`: (ne x 2) integer array, nodes that are connected by each edge
`boundary_indices`: integer array, unique entries of `edges`
`emap`: (nt x 3) integer array, mapping from local triangle edges
to the global edge list, i.e., `emap[i,k]`` is the global edge number
for local edge k (1,2,3) in triangle `i`
"""
function all_edges(t)
    etag = vcat(t[:,[1,2]], t[:,[2,3]], t[:,[3,1]])
    etag = hcat(sort(etag, dims=2), 1:3*size(t,1))
    etag = sortslices(etag, dims=1)
    dup = all(etag[2:end,1:2] - etag[1:end-1,1:2] .== 0, dims=2)[:]
    keep = .![false;dup]
    edges = etag[keep,1:2]
    emap = cumsum(keep)
    invpermute!(emap, etag[:,3])
    emap = reshape(emap,:,3)
    dup = [dup;false]
    dup = dup[keep]
    bndix = findall(.!dup)
    return edges, bndix, emap
end

"""
    e = boundary_nodes(t)

Find all boundary nodes in the triangulation `t`.
"""
function boundary_nodes(t)
    edges, boundary_indices, _ = all_edges(t)
    return unique(edges[boundary_indices,:][:])
end

"""
    tplot(p, t; u=nothing, levels = 51, shownan = true, limits = nothing)
If `u` == nothing: Plot triangular mesh with nodes `p` and triangles `t`.
If `u` == solution vector: Plot filled contour color plot of solution `u`.
"""
function tplot(p, t; u=nothing, levels = 51, shownan = true, limits = nothing)
    axis("equal")
    if u == nothing || isempty(u)
        plt = tripcolor(p[:,1], p[:,2], t .- 1, 0*t[:,1], edgecolors="k", linewidth=1)
    else
        if limits == nothing
            vmax = nanmax(u)
            vmin = nanmin(u)
        else
            vmin, vmax = limits
        end
        t, u, umask = valid(t, u)
        plt = tricontourf(p[:, 1], p[:, 2], t .- 1, u, levels = range(vmin, stop = vmax, length = levels))
        shownan && plot(p[umask, 1], p[umask, 2], "r*", label = "Invalid data")
    end
    return plt
end

"""
    t, u, umask = valid(t, u)
Safely remove NaN's from triangulated data `u` for plotting using tricontourf().

"""
function valid(t, u)
    nt = size(t, 1)
    tmask = trues(nt)
    umask = falses(size(u))
    for i = 1:nt
        for a in t[i, :]
            if isnan(u[a])
                umask[a] = true
                tmask[i] = false
            end
        end
    end
    t = t[tmask, :]
    u[umask] .= 0
    return t, u, umask
end

"""
    inside = inpolygon(p, pv)
Determine if each point in the N x 2 node array `p` is inside the polygon
described by the NE x 2 node array `pv`.
"""
function inpolygon(p::Array{Float64,2}, pv::Array{Float64,2})
    path = pyimport("matplotlib.path")
    poly = path.Path(pv)
    inside = [poly.contains_point(p[ip,:]) for ip = 1:size(p,1)]
end

"""
    circs, cents, areas = tristats(p, t, tol = 1e-5)
Determine circumcenter, centroid, and area for triangles
with nodes `p` and connectivity `t`
"""
function tristats(p, t, tol = 1e-5)
    nt = size(t, 1) # number of triangles
    circs = zeros(nt, 2) # circumcenters
    cents = zeros(nt, 2) # centroids
    areas = zeros(nt, 1)

    for i = 1:nt

        A  = p[t[i, 1], :] # corner points
        B  = p[t[i, 2], :]
        C  = p[t[i, 3], :]

        M = (A + B)/2 # midpoint

        a = norm(A - B) # side lengths
        b = norm(B - C)
        c = norm(C - A)

        s = (a + b + c)/2

        K = sqrt(abs(s*(s - a)*(s - b)*(s - c))) #abs() needed for negative values below machine precision
        R = (a*b*c)/(4*K);
        AB = B - A

        nrmAB = norm(AB)
        nrmOM = sqrt(abs(R^2 - (nrmAB/2)^2))
        r = [0 -1; 1 0]
        OM = nrmOM .*(r*AB/nrmAB) # pm OM
        O = M + OM
        if abs(norm(M + OM - C) - R) > s*tol # flip direction if initial guess incorrect
            O = M - OM
        end

        circs[i, :] = O
        cents[i, :] = (A + B + C)/3
        areas[i] = K

    end
    return circs, cents, areas
end

"""
    p, t, e = pmesh(pv, hmax, nref)
Create an unstructured first-order triangular 2D mesh from polygon `pv`,
max initial edge length `hmax` and number of bisection refinement steps `nref`
"""
function mesh(pv, hmax, nref)
    ne = size(pv, 1) - 1 # number of edges
    p = pv[1:ne, :] # start with just corner points, remove last point
    for i = 1:ne
        disp = pv[i, :] - pv[i + 1, :] # relative displacement between subsequent polygon nodes
        l  = norm(disp)
        n = disp/l  # unit vector between corners
        d = Int64(ceil(l/hmax) - 1) # number of dividing points required
        p = vcat(p, ones(d, 1)*reshape(pv[i+1, :], 1, 2) + (l/(d + 1))*reshape(Vector(1:d), d, 1)*reshape(n, 1, 2) );
    end

    t = delaunay(p) # triangulate edge nodes
    circ, c, A = tristats(p, t)
    inside = inpolygon(c, pv)
    c = c[inside, :]; circ = circ[inside, :]; t = t[inside, :]; A = A[inside] # remove outside triangles

    while maximum(A) > (hmax^2)/2
        ind = findall(x->x == maximum(A), A) # find the biggest boy
        ind = [ind[1]]
        p = vcat(p, circ[ind, :]) # add the circumcenter of the biggest boy to p
        t = delaunay(p)
        circ, c, A = tristats(p, t)
        inside = inpolygon(c, pv)
        c = c[inside, :]; circ = circ[inside, :]; t = t[inside, :]; A = A[inside] # remove outsi
    end

    for g = 1:nref # refine the mesh
        pta = reshape([], 0, 2) # initialize empty array for points to average
        for i = 1:size(t, 1)
            newpta = [t[i, 1] t[i, 2]; t[i, 2] t[i, 3]; t[i, 3] t[i, 1]]
            pta = vcat(pta, newpta)
        end

        pta = sort(pta, dims = 2)# sort pta along second dimension so you cant miss matches that are flipped
        pta = unique(pta, dims = 1)# make pta unique along first dimension
        for j = 1:size(pta, 1)# add average of pta points to p
            p = vcat(p, reshape((p[pta[j, 1], :] + p[pta[j, 2], :])/2, 1, 2))
        end
        t = delaunay(p) # triangulate edge nodes
        circ, c, A = tristats(p, t)

        inside = inpolygon(c, pv) # remove outside triangles
        c = c[inside, :]
        circ = circ[inside, :]
        t = t[inside, :]
        A = A[inside]
    end

    gp, gt = triquad(p, t, c)

    e = boundary_nodes(t)
    return p, t, e, gp, gt
end

"""
    p2, t2, e2 = p2mesh(p, t)
Create an unstructured second-order triangular 2D mesh from first-order mesh coordinates `p`
and connectivity `t`
"""
function p2mesh(p, t)
    # adds midpoints to all edges of the mesh defined by nodal locations p and connectivity t
    ae, _, emap = all_edges(t) # find the edges and emap for the original triangulation
    np = size(p, 1) # previous number of nodes
    p2 = vcat(p, (p[ae[:, 1], :] + p[ae[:, 2], :])/2) # add all the midpoints.
    t2 = hcat(t, np .+ emap)
    e2 = []
       for s = 1:(size(p2,1))
            coords = p2[s,:]
            if coords[1] == 0 || coords[2] == 0 || coords[1] == 1 || coords[2] == 1 # if the node is on the boundary
                e2 = vcat(e2,s)
            end
        end
    return p2, t2, e2
end

"""
    gp , inds = triquad(p, t)
Determine the appropriate quadrature point locations
`gp`: (np * nt x 2) array of [x, y] coordinates of all quadrature points
`inds`: (nt x np), indices of quadrature points for each element
Currently hard-coded with second-order 3-point quadrature.

Generalization could be adapted from https://www.sciencedirect.com/science/article/pii/S0898122103900046
"""
function triquad(p, t, cent)
    nt = size(t, 1)
    np = 3
    gp = zeros(np*nt, 2)
    inds = zeros(nt , np)
    for i = 1:nt
        inds[i, :] = (1:np) .+ np * (i - 1)
        for j = 1:np
            gp[(i - 1) * np + j, :] = (cent[i, :] + p[t[i, j], :]) / 2
        end
    end
    return gp, Int64.(inds)
end

"""
    p, t = trefine(p, t; reps = 1)
Refine the triangulation p, t by adding points on each edge and retriangulating

"""
function trefine(p, t; reps = 1)
    # adds midpoints to all edges of the mesh defined by nodal locations p and connectivity t

    for _ = 1:reps
        ae, _, emap = all_edges(t) # find the edges and emap for the original triangulation
        p = vcat(p, (p[ae[:, 1], :] + p[ae[:, 2], :])/2) # add all the midpoints.
        t = delaunay(p)
    end
    return p, t
end

end # end Mesh
