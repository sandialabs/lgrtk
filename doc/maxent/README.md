# Maxent

Experimental code for Optimal Transportation Method (OTM) Meshless Framework
code with Local Max. Entropy (LME) approximation functions. This code focuses on
the interaction between the nonlinear optimization scheme used to enforce
first-order consistency and the methods for insertion and regulation of node locations. 

## Getting Started

```
main.jl
```
Contains example script used to make visuals for August 2020 presentations.
Start by trying to run this script as-is, all the key modules are called by this script.

```
matlab_comparison.jl
```
Visualizations and test of mesh used in Matlab prototype script to confirm functionality.

The most computationally expensive part of this script, evaluating the shape functions at all
requested points, is trivially parallel at the material point level. The `Threads.@threads` macro has been placed in front of the most expensive loops.


The number of active threads can be verified using:

```
julia> Threads.nthreads()
```

To change the number of available threads, you can use e.g.: 

```
export JULIA_NUM_THREADS=8
```

### Prerequisites

Required Julia packages:

```
PyPlot / Plots
PyCall
LinearAlgebra
Threads
```

Start the Julia REPL by typing "Julia" at the command line or using the REPL panel in Atom.

If the code crashes while exporting animations or figures, ensure that `Plots` has been added and built correctly. 
Add and build Plots using Pkg

```
>> julia

julia> using Pkg; Pkg.add("Plots"); Pkg.build("Plots")

```
