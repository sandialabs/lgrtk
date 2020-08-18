# Support functions for Matlab-like syntax and NaN-handling.

# Written by Avery Rock, <avery_rock@berkeley.edu>
# Written for Julia 1.4.2, Summer 2020

"""
    s = nansum(x)
Sum entries of array x, ignoring NaN
"""
function nansum(x)
    sum = 0
    for i in x
        if !isnan(i)
            sum += i
        end
    end
    return sum
end

"""
    a = nanabs(x)
Absolute value of x if x is not NaN.
Return 0 if x is NaN.
"""
function nanabs(x)
    if isnan(x)
        return 0
    else
        return abs(x)
    end
end

"""
    a = nanmax(x)
Return max of x, excluding nan. -Inf by default.
"""
function nanmax(x)
    m = -Inf
    for i in x
        if i >= m
            m = i
        end
    end
    return m
end

"""
    a = nanmin(x)
Return max of x, excluding nan. Inf by default.
"""
function nanmin(x)
    m = Inf
    for i in x
        if i <= m
            m = i
        end
    end
    return m
end

"""
    A = eye(n)
n by n identity matrix. Matlab-like syntax.
"""
function eye(n)
    Matrix{Float64}(I, n, n)
end

"""
    timestamp(msg = "")
Print a message (default empty string) with the current time.
"""
function timestamp( msg = "" )
    println()
    println(msg, Dates.format(Dates.now(), "HH:MM:SS"))
end
