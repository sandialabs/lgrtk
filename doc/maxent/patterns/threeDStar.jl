limits = (-1, 1)


xp = zeros(6, 3)

for i in 1:3
    global xp[2*i - 1, i] = 1
    global xp[2i, i] = -1

end

for x in limits, y in limits, z in limits
    global xp = vcat(xp, [x y z])
end

xp[1:6, :] *= 10
xp[7:end, :] *= 14

# close("all")
#
# plot3D(xp[:, 1], xp[:, 2], xp[:, 3], "ro")
