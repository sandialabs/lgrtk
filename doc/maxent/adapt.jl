# Node injection methods. 

# Written by Avery Rock, <avery_rock@berkeley.edu>
# Written for Julia 1.4.2, Summer 2020

module adapt
using LinearAlgebra
function adapt1!(x; tol = 1e-1)
    for i ∈ 1:size(x, 1)
        dmin = Inf
        d = 0
        for j ∈ i + 1:size(x, 1)
            if norm(x[i, :] - x[j, :]) < dmin
                dmin = norm(x[i, :] - x[j, :])
                d = j
            end
        end
        if dmin < tol
            x = [x; (x[i, :] .+ x[j, :])./2]
        end
    end
end

function adapt2!(x; tol = 1e-1, kmax = 1e6)
    error("unimplemented")
end

end
