# Linear system solvers for LME lagrange multiplier step.

using LinearAlgebra

"""
    x = babuska(A, b; xo = nothing, ϵ = 1e-6, maxiters = 20, tol = 1e5eps())
Iteratively solve poorly scaled Ax = b using babuska iterative method.
"""
function babuska(A, b; xo = nothing, ϵ = 1e-2, maxiters = 10, tol = 1e4eps())
    try
        x = zeros(length(b), maxiters)
        xt = zeros(length(b), maxiters)
        r = zeros(length(b), maxiters + 1)

        A_ϵ = A + ϵ*eigen(A).values[1]*eye(length(b))

        if xo == nothing
            xo = zeros(size(b))
        end

        r[:, 1] = b - A*xo

        for i = 1:maxiters
            x[:, i] = pinv(A_ϵ)*(b + sum(r, dims = 2))
            r[:, i + 1] = b - A*x[:, i]

            rh = r[:, i + 1] / norm(r[:, i + 1])
            xt[:, i] = x[:, i] - rh * dot(x[:, i], rh)

            if i > 1 && norm(xt[:, i] - xt[:, i - 1]) / norm(xt[:, i - 1]) < tol
                break
            end
        end
        return x[:, end]
    catch
        return 0*b
    end
end

"""
    x = pinvs(A, b)
Use Moore-Penrose pseudoinverse to solve Ax = b
"""
function pinvs(A, b)
    return pinv(A) * b
end

"""
    x = direct(A, b)
Use backslash to solve Ax = b. Return 0 if system is singular
"""
function direct(A, b)
    try
        return A \ b
    catch SingularException
        return 0*b
    end
end

"""
    dx = linesearch(f, df, x, dx; maxit = 20, τ = .618, c = .5)
Backtracking line search to minimize function f(x).
"""
function linesearch(f, df, x, dx; maxit = 10, τ = .618, c = .5)
    fo = f(x)
    m = dot(df, dx) # projected gradient onto search direction
    t = -c*m
    j = 0 # line search iterations
    α = 1.0
    converged = false
    for j ∈ 1:maxit
        Δf = fo - f(x + α * dx)
        Δx = α
        if Δf/Δx >= t
            converged = true
            break
        end
        α *= τ
    end
    converged ? (return α*dx) : (return dx)
end

"""
    x = gd(A, b)
perform gradient descent by returning vector b and ignoring Hessian A.
"""
function gd(A, b; α = .1)
    return α*b
end

"""
    x = guess(A, b; α = 0.1)
Return random vector in n-space
"""
function guess(A, b; α = 10.0)
    return α*(rand(length(b)) .- .5)
end

"""
    tests = test_solver(solver)
Performs sanity tests for linear system solver.
"""
function test_solver(solver)
    tests = []
    A = eye(3)
    b = ones(3)
    x = solver(A, b)
    r = A*x - b
    push!(tests, norm(r) < 1e2eps())

    A = [1 2 3; 1 3 2; 1 1 1]
    b = [0; 0; 1]
    x = solver(A, b)
    r = A*x - b
    push!(tests, norm(r) < 1e2eps())

    A = [1 1 1; 1 1 0; 1 1 1 + 1e1eps()]
    b = [1; 1; 1]
    x = solver(A, b)
    r = A*x - b
    push!(tests, norm(r) < 1e2eps())

    return tests
end
