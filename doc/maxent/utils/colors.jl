"""
    c = randcolor(;a = 0.5)
Make a random RGB color triplet with a given average value
"""
function randcolor(; c = nothing, a = 0.5, tol = 1e-3)
    0.0 < a < 1.0 || error("a must be between 0 and 1 exclusive")
    if c == nothing
        c = rand(3)
    else
        c = min.(c, 1.0)
        c = max.(c, 0.0)
    end

    while abs(sum(c) - 3a) > tol
        if sum(c) > 3a
            c = c.^1.1
        else
            c = c.^.95
        end
    end
    @assert all( 0.0 .<= c .<= 1.0) "One or more value of c ∉ [0, 1]"
    return tuple(c...)
end

"""
    c = colorinterp(s, f, n)
Make interpolated colors between arbitrary values.
`s::RGB triplet`, first color
`f::RGB triplet`, last color
`n::Int8`, number of total values
"""
function colorinterp(s, f, n)
    δ = f .- s
    a = range(0.0, stop = 1.0, length = n)
    c = []
    for i = 1:n
        color = s .+ a[i] .* δ
        push!(c, color)
    end
    return c
end

"""
    cs = colorarray()
Produce array of visually distinct random colors suitable for plots with related series.
"""
function colorarray(nr, ns; ci = (.05, .03, .53), cf = (.94, .98, .13))
    cs = fill(tuple(zeros(Float16, 3)...), (nr, ns))
    ca = colorinterp(ci, cf, ns)
    for j ∈ 1:ns
        c1 = ca[j]
        c2 = c1 .* .5
        cs[:, j] = colorinterp(c1, c2, nr)
    end
    return cs
end
