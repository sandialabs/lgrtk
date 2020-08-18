# Example script using key functions for visuals used in August 2020 presentation.

# Written by Avery Rock, <avery_rock@berkeley.edu>
# Written for Julia 1.4.2, Summer 2020

# Default parameters are low resolution to limit execution time. See comparison.jl and testplots.jl

Pkg.add("Plots")
Pkg.build("Plots")

include("testplots.jl")
include("comparison.jl")

# function visuals
oneDSweep()
twoDSweep()
threeDSweep()

# linear solver success rates
comparison3r(points = 1e3, reps = 5)
