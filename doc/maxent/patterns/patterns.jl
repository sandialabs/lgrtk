# Example node patterns for LME test cases.

# Written by Avery Rock, <avery_rock@berkeley.edu>
# Written for Julia 1.4.2, Summer 2020

struct Pattern
    nodes::Array{Float64}
    points::Array{Float64}
    h::Float64
end

xp2 = [-15.0 15; 0 7; 15 15; 7 0; 15 -15; 0 -7; -15 -15; -7 0]

limits = (-1, 1)
xp3 = zeros(6, 3)

for i in 1:3
    global xp3[2*i - 1, i] = 1
    global xp3[2i, i] = -1
end

for x in limits, y in limits, z in limits
    global xp3 = vcat(xp3, [x y z])
end

xp3[1:6, :] *= 7
xp3[7:end, :] *= 10

star2d = Pattern(xp2, brick(minimum(xp2, dims = 1), maximum(xp2, dims = 1), (10, 10)), 20)
star3d = Pattern(xp3, brick(minimum(xp3, dims = 1), maximum(xp3, dims = 1), (10, 10, 10)), 20)
